from ast import Str
import os
import argparse
import random
import numpy as np
from TFIDF import TFIDF_Builder

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from transformers.optimization import get_linear_schedule_with_warmup, Adafactor
import nlp
import pickle
from rouge_score import rouge_scorer
from util import sentence_processor

import pytorch_lightning as pl
from pytorch_lightning.logging import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel
from nltk.translate.bleu_score import sentence_bleu 


from longformer import LongformerEncoderDecoderForConditionalGeneration, LongformerEncoderDecoderConfig
from longformer.sliding_chunks import pad_to_window_size

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
        count = (~pad_mask).sum()
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
        count = nll_loss.numel()

    nll_loss = nll_loss.sum() / count
    smooth_loss = smooth_loss.sum() / count
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


class Seq2SeqDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_input_len, max_output_len):
        with open(file_path,'rb') as fin:
            self.data = pickle.loads(fin.read())
            if("train" in file_path and args.is_small):
                self.data = self.data[:200]
                pass
            if("dev" in file_path and args.is_small):
                self.data = self.data[:100]
                pass
            if('test' in file_path and args.is_small):
                self.data = self.data[:20]
                pass
        
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        #计算dependency tree
        # [label,context,response,tfidf_value]
        # def tok(s):
        #     return self.tokenizer.tokenize(s)
        # tokens = [self.tokenizer.cls_token] + tok(entry[1]) + [self.tokenizer.sep_token]
        input_ids = self.tokenizer.encode(entry[1])

        tf_idf_mask = []

        for score in entry[3]:
            if score > 0.2: #TODO: TD, change the number to 0-1, 0.2=use top 80% words
                tf_idf_mask.append(round(score,3))
            else:
                tf_idf_mask.append(0.0)

        if(len(input_ids)>self.max_input_len):
            first = input_ids[0]
            first_mask = tf_idf_mask[0]
            input_ids = input_ids[-self.max_input_len+1:]
            input_ids.insert(0,first)

            tf_idf_mask = tf_idf_mask[-self.max_input_len+1:]
            tf_idf_mask.insert(0,first_mask)

        input_len = len(input_ids)

        padding_length = self.max_input_len - input_len
        tf_idf_mask = tf_idf_mask + ([0.0] * padding_length)
        input_ids = input_ids + ([self.tokenizer.pad_token_id] * padding_length)
        
        output_ids = self.tokenizer.encode(entry[2], truncation=True, max_length=self.max_output_len)
        

        if self.tokenizer.bos_token_id is None:  # pegasus
            output_ids = [self.tokenizer.pad_token_id] + output_ids
        return torch.tensor(input_ids), torch.tensor(output_ids), torch.tensor(tf_idf_mask), self.convert_sparse_matrix_to_sparse_tensor(entry[4])
    
    def convert_sparse_matrix_to_sparse_tensor(self,X):
        coo = X.tocoo()
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        full_mat = torch.sparse.FloatTensor(i, v, torch.Size(coo.shape)).to_dense()
        
        return full_mat

    @staticmethod
    def collate_fn(batch):
        # A hack to know if this is bart or pegasus. DDP doesn't like global variables nor class-level memebr variables
        if batch[0][0][-1].item() == 2:
            pad_token_id = 1  # AutoTokenizer.from_pretrained('facebook/bart-base').pad_token_id
        elif batch[0][0][-1].item() == 1:
            pad_token_id = 0  # AutoTokenizer.from_pretrained('google/pegasus-large').pad_token_id
        else:
            assert False

        input_ids, output_ids, tf_idf, dt = zip(*batch)
    
        input_ids = torch.nn.utils.rnn.pad_sequence(list(input_ids), batch_first=True, padding_value=pad_token_id)
       
        output_ids = torch.nn.utils.rnn.pad_sequence(list(output_ids), batch_first=True, padding_value=pad_token_id)
        tf_idf = torch.nn.utils.rnn.pad_sequence(list(tf_idf), batch_first=True, padding_value=pad_token_id)
        dt = torch.nn.utils.rnn.pad_sequence(list(dt), batch_first=True, padding_value=pad_token_id)
        # print(input_ids.shape)
        
        return input_ids, output_ids, tf_idf, dt


class Seq2Seq(pl.LightningModule):

    def __init__(self, params):
        super().__init__()
        self.args = params
        # self.hparams = params
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer, use_fast=True)
        

        if 'long' in self.args.model_path:
            config = LongformerEncoderDecoderConfig.from_pretrained(self.args.model_path)
            config.attention_dropout = self.args.attention_dropout
            config.gradient_checkpointing = self.args.grad_ckpt
            config.attention_mode = self.args.attention_mode
            config.attention_window = [self.args.attention_window] * config.encoder_layers
            self.model = LongformerEncoderDecoderForConditionalGeneration.from_pretrained(
                self.args.model_path, config=config)
        else:
            config = AutoConfig.from_pretrained(self.args.model_path)
            config.attention_dropout = self.args.attention_dropout
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.args.model_path, config=config)
        self.train_dataloader_object = self.val_dataloader_object = self.test_dataloader_object = None

    def _prepare_input(self, input_ids):
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        attention_mask[input_ids == self.tokenizer.pad_token_id] = 0
        if isinstance(self.model, LongformerEncoderDecoderForConditionalGeneration):
            attention_mask[:, 0] = 2  # global attention on one token for all model params to be used, which is important for gradient checkpointing to work
            if self.args.attention_mode == 'sliding_chunks':
                half_padding_mod = self.model.config.attention_window[0]
            elif self.args.attention_mode == 'sliding_chunks_no_overlap':
                half_padding_mod = self.model.config.attention_window[0] / 2
            else:
                raise NotImplementedError
            input_ids, attention_mask = pad_to_window_size(  # ideally, should be moved inside the LongformerModel
                input_ids, attention_mask, half_padding_mod, self.tokenizer.pad_token_id)
        return input_ids, attention_mask

    def forward(self, input_ids, output_ids, tf_idf, dt):
        input_ids, attention_mask = self._prepare_input(input_ids)
        if(self.args.use_tfidf):
            attention_mask = torch.reshape(attention_mask,(attention_mask.shape[0],1,attention_mask.shape[1]))
            # print(attention_mask.shape)
            tf_idf = torch.reshape(tf_idf,(tf_idf.shape[0],1,tf_idf.shape[1]))
            # print(tf_idf.shape)
            mix_tensor = torch.cat((attention_mask,tf_idf),dim=1)
            
        decoder_input_ids = output_ids[:, :-1]
        decoder_attention_mask = (decoder_input_ids != self.tokenizer.pad_token_id)
        labels = output_ids[:, 1:].clone()
        # print(self.model)
        outputs = self.model(
                input_ids,
                attention_mask=mix_tensor if self.args.use_tfidf else attention_mask, #TODO: mix_tensor=using TFIDF attention_mask = not using
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                dt=dt,
                use_cache=False)
        lm_logits = outputs[0]
        # print(outputs)
        # print(kkk)
        if self.args.label_smoothing == 0:
            # Same behavior as modeling_bart.py, besides ignoring pad_token_id
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            assert lm_logits.shape[-1] == self.model.config.vocab_size
            loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
        else:
            lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, labels, self.args.label_smoothing, ignore_index=self.tokenizer.pad_token_id
            )
        return [loss]

    def training_step(self, batch, batch_nb):
        output = self.forward(*batch)
        loss = output[0]
        lr = loss.new_zeros(1) + self.trainer.optimizers[0].param_groups[0]['lr']
        tensorboard_logs = {'train_loss': loss, 'lr': lr,
                            'input_size': batch[0].numel(),
                            'output_size': batch[1].numel(),
                            'mem': torch.cuda.memory_allocated(loss.device) / 1024 ** 3 if torch.cuda.is_available() else 0}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        for p in self.model.parameters():
            p.requires_grad = False
        
        outputs = self.forward(*batch)
        vloss = outputs[0]
        # print(outputs)

        input_ids, output_ids,tf_idf,dt = batch
        input_ids, attention_mask = self._prepare_input(input_ids)
        
        if(self.args.use_tfidf):
            attention_mask = torch.reshape(attention_mask,(attention_mask.shape[0],1,attention_mask.shape[1]))
            tf_idf = torch.reshape(tf_idf,(tf_idf.shape[0],1,tf_idf.shape[1]))
            mix_tensor = torch.cat((attention_mask,tf_idf),dim=1)
        generated_ids = self.model.generate(input_ids=input_ids, dt=dt,attention_mask=mix_tensor if self.args.use_tfidf else attention_mask,#TODO: mix_tensor=using TFIDF attention_mask = not using
                                            use_cache=True, max_length=self.args.max_output_len,
                                            num_beams=1)
        generated_str = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        # generated_str = ""
        gold_str = self.tokenizer.batch_decode(output_ids.tolist(), skip_special_tokens=True)
        scorer = rouge_scorer.RougeScorer(rouge_types=['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=False)
        rouge1 = rouge2 = rougel = rougelsum = 0.0
        for ref, pred in zip(gold_str, generated_str):
            score = scorer.score(ref, pred)
            rouge1 += score['rouge1'].fmeasure
            rouge2 += score['rouge2'].fmeasure
            rougel += score['rougeL'].fmeasure
            rougelsum += score['rougeLsum'].fmeasure
        rouge1 /= len(generated_str)
        rouge2 /= len(generated_str)
        rougel /= len(generated_str)
        rougelsum /= len(generated_str)

        return {'vloss': vloss,
                'rouge1': vloss.new_zeros(1) + rouge1,
                'rouge2': vloss.new_zeros(1) + rouge2,
                'rougeL': vloss.new_zeros(1) + rougel,
                'rougeLsum': vloss.new_zeros(1) + rougelsum, }
    def predict_step(self, batch, batch_idx):
        history = ""
        while True:
            raw_text = input("USR >>> ")
            input_ids, attention_mask = self._prepare_input(input_ids)
            input_ids = self.tokenizer.encode(history, truncation=True, max_length=self.args.max_input_len)
        
            padding_length = self.args.max_input_len - len(input_ids)
            input_ids = input_ids + ([self.tokenizer.pad_token_id] * padding_length)
            # print(input_ids)
            # print(kkk)

    def validation_epoch_end(self, outputs):
        for p in self.model.parameters():
            p.requires_grad = True

        names = ['vloss', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        metrics = []
        for name in names:
            metric = torch.stack([x[name] for x in outputs]).mean()
            if self.trainer.use_ddp:
                torch.distributed.all_reduce(metric, op=torch.distributed.ReduceOp.SUM)
                metric /= self.trainer.world_size
            metrics.append(metric)
        logs = dict(zip(*[names, metrics]))
        
        return {'avg_val_loss': logs['vloss'], 'log': logs, 'progress_bar': logs}

    def test_step(self, batch, batch_nb):
        if(self.args.interact):
            history = ""
            while True:
                raw_text = input("USR >>> ")
                history += raw_text+" </s> "
                input_ids = self.tokenizer.encode(history, truncation=True, max_length=self.args.max_input_len)
                padding_length = self.args.max_input_len - len(input_ids)
                input_ids = input_ids + ([self.tokenizer.pad_token_id] * padding_length)
                input_ids = torch.tensor(input_ids)
                input_ids = torch.reshape(input_ids,(1,self.args.max_input_len)).to(batch[0].device)
                input_ids, attention_mask = self._prepare_input(input_ids)
                if(self.args.use_tfidf):
                    attention_mask = torch.reshape(attention_mask,(attention_mask.shape[0],1,attention_mask.shape[1]))
                    tf_idf = torch.reshape(tf_idf,(tf_idf.shape[0],1,tf_idf.shape[1]))
                    mix_tensor = torch.cat((attention_mask,tf_idf),dim=1)
                # tf_idf = torch.reshape(tf_idf,(tf_idf.shape[0],1,tf_idf.shape[1]))
                # mix_tensor = torch.cat((attention_mask,tf_idf),dim=1)
                generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                            use_cache=True, max_length=self.args.max_output_len,
                                            num_beams=1)
                generated_str = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
                
                print("SYS >>> ", generated_str[0])
                history += generated_str[0]+' </s> '

        for p in self.model.parameters():
            p.requires_grad = False

        outputs = self.forward(*batch)
        vloss = outputs[0]
        input_ids, output_ids,tf_idf,dt = batch
        input_ids, attention_mask = self._prepare_input(input_ids)
        if(self.args.use_tfidf):
            attention_mask = torch.reshape(attention_mask,(attention_mask.shape[0],1,attention_mask.shape[1]))
            tf_idf = torch.reshape(tf_idf,(tf_idf.shape[0],1,tf_idf.shape[1]))
            mix_tensor = torch.cat((attention_mask,tf_idf),dim=1)

        generated_ids = self.model.generate(input_ids=input_ids, attention_mask=mix_tensor if self.args.use_tfidf else attention_mask, #TODO: TFIDF
                                            use_cache=True, max_length=self.args.max_output_len,dt=dt,
                                            num_beams=1)
        generated_str = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        gold_str = self.tokenizer.batch_decode(output_ids.tolist(), skip_special_tokens=True)
        scorer = rouge_scorer.RougeScorer(rouge_types=['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=False)
        rouge1 = rouge2 = rougel = rougelsum = 0.0

        output_strs = []
        for ref, pred in zip(gold_str, generated_str):
            
            output_strs.append(str(ref)+"\t"+str(pred))
            score = scorer.score(ref, pred)
            rouge1 += score['rouge1'].fmeasure
            rouge2 += score['rouge2'].fmeasure
            rougel += score['rougeL'].fmeasure
            rougelsum += score['rougeLsum'].fmeasure
        rouge1 /= len(generated_str)
        rouge2 /= len(generated_str)
        rougel /= len(generated_str)
        rougelsum /= len(generated_str)
        print(len(output_strs))
        # print(kkk)
        with open(self.args.save_dir+"/generated_str.txt",'a',encoding='utf-8') as f2:
            f2.write("\n".join(output_strs)+'\n')

        return {'vloss': vloss,
                'rouge1': vloss.new_zeros(1) + rouge1,
                'rouge2': vloss.new_zeros(1) + rouge2,
                'rougeL': vloss.new_zeros(1) + rougel,
                'rougeLsum': vloss.new_zeros(1) + rougelsum, }

    def test_epoch_end(self, outputs):
        result = self.validation_epoch_end(outputs)
        print(result)

    def configure_optimizers(self):
        if self.args.adafactor:
            optimizer = Adafactor(self.model.parameters(), lr=self.args.lr, scale_parameter=False, relative_step=False)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        if self.args.debug:
            return optimizer  # const LR
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        num_steps = self.args.dataset_size * self.args.epochs / num_gpus / self.args.grad_accum / self.args.batch_size
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup, num_training_steps=num_steps
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def _get_dataloader(self, current_dataloader, split_name, is_train):
        if current_dataloader is not None:
            return current_dataloader
        if split_name == 'train':
            fname = self.args.train_file
        elif split_name == 'dev':
            fname = self.args.dev_file
        elif split_name == 'test':
            fname = self.args.test_file
        else:
            assert False
        dataset = Seq2SeqDataset(file_path=fname, tokenizer=self.tokenizer,
                                       max_input_len=self.args.max_input_len, max_output_len=self.args.max_output_len)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=is_train) if self.trainer.use_ddp else None
        return DataLoader(dataset, batch_size=self.args.batch_size, shuffle=(sampler is None),
                          num_workers=self.args.num_workers, sampler=sampler,
                          collate_fn=Seq2SeqDataset.collate_fn)

    @pl.data_loader
    def train_dataloader(self):
        self.train_dataloader_object = self._get_dataloader(self.train_dataloader_object, 'train', is_train=True)
        return self.train_dataloader_object

    @pl.data_loader
    def val_dataloader(self):
        self.val_dataloader_object = self._get_dataloader(self.val_dataloader_object, 'dev', is_train=False)
        return self.val_dataloader_object

    @pl.data_loader
    def test_dataloader(self):
        self.test_dataloader_object = self._get_dataloader(self.test_dataloader_object, 'test', is_train=False)
        return self.test_dataloader_object

    def configure_ddp(self, model, device_ids):
        model = LightningDistributedDataParallel(
            model,
            device_ids=device_ids,
            find_unused_parameters=False
        )
        return model

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument("--save_dir", type=str, default='seq2seq')
        parser.add_argument('--train_file')
        parser.add_argument('--dev_file')
        parser.add_argument('--test_file')
        parser.add_argument("--save_prefix", type=str, default='pl-log')
        parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
        parser.add_argument("--grad_accum", type=int, default=1, help="number of gradient accumulation steps")
        parser.add_argument("--gpus", type=int, default=1,
                            help="Number of gpus. 0 for CPU")
        parser.add_argument("--warmup", type=int, default=1000, help="Number of warmup steps")
        parser.add_argument("--lr", type=float, default=0.00003, help="Maximum learning rate")
        parser.add_argument("--val_every", type=float, default=1.0, help="Number of training steps between validations")
        parser.add_argument("--val_percent_check", default=1.00, type=float, help='Percent of validation data used')
        parser.add_argument("--num_workers", type=int, default=10, help="Number of data loader workers")
        parser.add_argument("--seed", type=int, default=1234, help="Seed")
        parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
        parser.add_argument("--disable_checkpointing", action='store_true', help="No logging or checkpointing")
        parser.add_argument("--max_output_len", type=int, default=256,
                            help="maximum num of wordpieces/summary. Used for training and testing")
        parser.add_argument("--max_input_len", type=int, default=1024,
                            help="maximum num of wordpieces/summary. Used for training and testing")
        parser.add_argument("--test", action='store_true', help="Test only, no training")
        parser.add_argument("--interact", action='store_true', help="Predict only, no training")
        parser.add_argument("--model_path", type=str, default='facebook/bart-base',
                            help="Path to the checkpoint directory or model name")
        parser.add_argument("--tokenizer", type=str, default='facebook/bart-base')
        parser.add_argument("--no_progress_bar", action='store_true', help="no progress bar. Good for printing")
        parser.add_argument("--fp32", action='store_true', help="default is fp16. Use --fp32 to switch to fp32")
        parser.add_argument("--debug", action='store_true', help="debug run")
        parser.add_argument("--resume_ckpt", type=str, help="Path of a checkpoint to resume from")
        parser.add_argument("--from_pretrained", type=str, default=None,
                            help="Path to a checkpoint to load model weights but not training state")
        parser.add_argument('--grad_ckpt', action='store_true', help='Enable gradient checkpointing to save memory')
        parser.add_argument("--attention_dropout", type=float, default=0.1, help="attention dropout")
        parser.add_argument("--attention_mode", type=str, default='sliding_chunks', help="Longformer attention mode")
        parser.add_argument("--attention_window", type=int, default=512, help="Attention window")
        parser.add_argument("--version", type=str, default='0', help="Version")
        parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
        parser.add_argument("--adafactor", action='store_true', help="Use adafactor optimizer")
        parser.add_argument('--is_small', default=False, action='store_true')
        parser.add_argument('--use_tfidf', default=False, action='store_true')
        

        return parser


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.from_pretrained is not None:
        print("load from pretrain model {}".format(args.from_pretrained))
        model = Seq2Seq.load_from_checkpoint(args.from_pretrained)
    else:
        model = Seq2Seq(args)


    logger = TestTubeLogger(
        save_dir=args.save_dir,
        name=args.save_prefix,
        version=args.version
    )

    filepath = f'{args.save_dir}/version_{logger.version}/checkpoints/'
    checkpoint_callback = ModelCheckpoint(
        filepath=filepath,
        save_top_k=5,
        verbose=True,
        monitor='avg_val_loss',
        mode='min',
        period=-1,
        prefix=''
    )

    # print(args)

    args.dataset_size = 84114  # hardcode dataset size. Needed to compute number of steps for the lr scheduler

    trainer = pl.Trainer(gpus=args.gpus, distributed_backend=None if torch.cuda.is_available() else None,
                         track_grad_norm=-1,
                         max_epochs=args.epochs if not args.debug else 100,
                         max_steps=None if not args.debug else 1,
                         replace_sampler_ddp=False,
                         accumulate_grad_batches=args.grad_accum,
                         val_check_interval=args.val_every if not args.debug else 1,
                         num_sanity_val_steps=2 if not args.debug else 0,
                         check_val_every_n_epoch=1 if not args.debug else 1,
                         val_percent_check=args.val_percent_check,
                         test_percent_check=args.val_percent_check,
                         logger=logger,
                         checkpoint_callback=checkpoint_callback if not args.disable_checkpointing else False,
                         show_progress_bar=not args.no_progress_bar,
                         use_amp=not args.fp32, amp_level='O2',
                         resume_from_checkpoint=args.resume_ckpt,
                         )
    if(args.interact):
        print("Interaction model")
        # print(kkk)
        model.args.interact = True
        model.eval()
        trainer.test(model)
    else:
        model.args.interact = False

    if not args.test:
        trainer.fit(model)
    trainer.test(model)


if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(description="summarization")
    parser = Seq2Seq.add_model_specific_args(main_arg_parser, os.getcwd())
    args = parser.parse_args()
    main(args)
