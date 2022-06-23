import os

import json
import os
import random
import argparse
import numpy as np
import glob
import json
import pickle

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import torch.distributed as dist

from longformer.longformer import Longformer, LongformerConfig
from longformer.sliding_chunks import pad_to_window_size
from transformers import RobertaTokenizer, AdamW

from torch.utils.data.dataset import IterableDataset
from tqdm.auto import tqdm
from sklearn.metrics import precision_recall_fscore_support

from transformers.optimization import (
    Adafactor,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    # '': get_constant_schedule,             # not supported for now
    # '': get_constant_schedule_with_warmup, # not supported for now
}
arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def calc_f1(y_pred:torch.Tensor, y_true:torch.Tensor) -> torch.Tensor:
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1 = f1.clamp(min=epsilon, max=1 - epsilon)
    return f1


class ClassificationDataset(Dataset):

    def __init__(self, file_path, tokenizer, seqlen, args, num_samples=None, mask_padding_with_zero=True):
        self.data = []
        with open(file_path,'rb') as fin:
            self.data = pickle.loads(fin.read())
            if("train" in file_path and args.is_small):
                # self.data = self.data[23900:]
                pass
            if("dev" in file_path and args.is_small):
                self.data = self.data[:10000]
                pass
            if('test' in file_path and args.is_small):
                # self.data = self.data[:20000]
                pass
        self.seqlen = seqlen
        self._tokenizer = tokenizer
        all_labels = list(set([e[0] for e in self.data]))
        self.label_to_idx = {e: i for i, e in enumerate(all_labels)}
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        self.mask_padding_with_zero = mask_padding_with_zero

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self._convert_to_tensors(self.data[idx])

    def _convert_to_tensors(self, instance):
        def tok(s):
            return self._tokenizer.tokenize(s, add_prefix_space=True)
        tokens = [self._tokenizer.cls_token_id] + tok(instance[1]) + [self._tokenizer.sep_token] + tok(instance[2]) + [self._tokenizer.sep_token]
        token_ids = self._tokenizer.convert_tokens_to_ids(tokens)
        tf_idf_mask = []

        for score in instance[3]:
            if score > 0:
                tf_idf_mask.append(round(score,3))
            else:
                tf_idf_mask.append(0.0)

        if(len(token_ids)>self.seqlen):
            first = token_ids[0]
            first_mask = tf_idf_mask[0]
            token_ids = token_ids[-self.seqlen+1:]
            token_ids.insert(0,first)

            tf_idf_mask = tf_idf_mask[-self.seqlen+1:]
            tf_idf_mask.insert(0,first_mask)

        input_len = len(token_ids)

        #TODO:not use tfidf
        attention_mask = [1] * input_len

        padding_length = self.seqlen - input_len
        token_ids = token_ids + ([self._tokenizer.pad_token_id] * padding_length)

        attention_mask = attention_mask + ([0 if self.mask_padding_with_zero else 1] * padding_length)
        tf_idf_mask = tf_idf_mask + ([0.0] * padding_length)
        
        mix_tensor = []
        mix_tensor.append(attention_mask)
        mix_tensor.append(tf_idf_mask)

        assert len(token_ids) == self.seqlen, "Error with input length {} vs {}".format(
            len(token_ids), self.seqlen
        )
        assert len(attention_mask) == self.seqlen, "Error with input length {} vs {}".format(
            len(attention_mask), self.seqlen
        )

        label = int(instance[0])
        # print(attention_mask)
        # print(kkk)

        return (torch.tensor(token_ids), torch.tensor(mix_tensor), torch.tensor(label))


class LongformerClassifier(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()

        config = LongformerConfig.from_pretrained(args.model)
        config.attention_mode = 'sliding_chunks'
        self.model_config = config
        self.model = Longformer.from_pretrained(args.model, config=config)
        self.tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer)
        self.tokenizer.model_max_length = self.model.config.max_position_embeddings
        self.args = args
        self.args.seqlen = args.seqlen
        self.classifier = nn.Linear(args.seqlen, args.num_labels)
        self.pool = nn.Linear(config.hidden_size,1)
        self.L_1 = nn.Linear(config.hidden_size, 384)
        self.L_2 = nn.Linear(384, 2)
        self.L_3 = nn.Linear(150, 2)
        self.softmax = nn.Softmax()
        self.dropout_1 = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, labels=None):
        input_ids, attention_mask = pad_to_window_size(
            input_ids, attention_mask, self.model_config.attention_window[0], self.tokenizer.pad_token_id)
        attention_mask[:,0, 0] = 2  # global attention for the first token
        # print(attention_mask)
        output = self.model(input_ids, attention_mask=attention_mask)[0]
        output = self.dropout_1(output)
        # pool the entire sequence into one vector (CLS token)
        output = output[:, 0, :]
        logits = self.L_1(output)
        logits = self.dropout_1(logits)
        logits = self.L_2(logits)
        # logits = self.dropout_1(logits)
        # logits = self.L_3(logits)
        # logits = self.dropout_1(logits)
        logits = self.softmax(logits)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            print(logits.view(-1, self.args.num_labels), labels.view(-1))
            loss = loss_fct(logits.view(-1, self.args.num_labels), labels.view(-1))

        return logits, loss

    def _get_loader(self, split, shuffle=True):
        if split == 'train':
            fname = self.args.train_file
        elif split == 'dev':
            fname = self.args.dev_file
        elif split == 'test':
            fname = self.args.test_file
        else:
            assert False
        is_train = split == 'train'

        dataset = ClassificationDataset(
            fname, tokenizer=self.tokenizer, seqlen=self.args.seqlen, num_samples=self.args.num_samples,args=self.args
        )

        if self.args.gpus > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
            loader = DataLoader(dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,
                sampler=sampler)
        else:
            loader = DataLoader(
                dataset, batch_size=self.args.batch_size, shuffle=shuffle,
                num_workers=self.args.num_workers)
        return loader

    def setup(self, mode):
        if mode == "fit":
            self.train_loader = self._get_loader("train")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        self.val_dataloader_obj = self._get_loader('dev')
        return self.val_dataloader_obj

    def test_dataloader(self):
        return self._get_loader('test')

    @property
    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        num_devices = max(1, self.args.gpus)  # TODO: consider num_tpu_cores
        effective_batch_size = self.args.batch_size * self.args.grad_accum * num_devices
        dataset_size = len(self.train_loader.dataset)
        return (dataset_size / effective_batch_size) * self.args.num_epochs

    def get_lr_scheduler(self):
        get_schedule_func = arg_to_scheduler[self.args.lr_scheduler]
        scheduler = get_schedule_func(
            self.opt, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.total_steps
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.args.adafactor:
            optimizer = Adafactor(
                optimizer_grouped_parameters, lr=self.args.lr, scale_parameter=False, relative_step=False
            )

        else:
            optimizer = AdamW(
                optimizer_grouped_parameters, lr=self.args.lr, eps=self.args.adam_epsilon
            )
        self.opt = optimizer

        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}

        outputs = self(**inputs)
        loss = outputs[1]

        lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
        tensorboard_logs = {"loss": loss, "rate": lr_scheduler.get_last_lr()[-1]}
        return {"loss": loss, "log": tensorboard_logs}


    def validation_step(self, batch, batch_idx):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}

        outputs = self(**inputs)
        logits, tmp_eval_loss = outputs
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs["labels"].detach().cpu().numpy()
        return {"val_loss": tmp_eval_loss.detach().cpu(), "pred": preds, "target": out_label_ids}

    def _eval_end(self, outputs) -> tuple:
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean().detach().cpu().item()
        preds = np.concatenate([x["pred"] for x in outputs], axis=0)

        preds = np.argmax(preds, axis=1)

        out_label_ids = np.concatenate([x["target"] for x in outputs], axis=0)
        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]
        # _, _, f1, _ = precision_recall_fscore_support(out_label_ids, preds)
        accuracy = (preds == out_label_ids).sum() / preds.shape[0]
        # accuracy = (preds == out_label_ids).int().sum() / torch.tensor(preds.shape[0], dtype=torch.float32, device=out_label_ids.device)
        results = {"val_loss": val_loss_mean, "acc": accuracy}

        ret = {k: v for k, v in results.items()}
        ret["log"] = results
        return ret, preds_list, out_label_list

    def validation_epoch_end(self, outputs: list) -> dict:
        ret, preds, targets = self._eval_end(outputs)
        logs = ret["log"]
        return {"val_loss": logs["val_loss"], "log": logs, "progress_bar": logs}

    def _test_end(self, outputs) -> tuple:
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean().detach().cpu().item()
        preds = np.concatenate([x["pred"] for x in outputs], axis=0)

        out_label_ids = np.concatenate([x["target"] for x in outputs], axis=0)
        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]
        # _, _, f1, _ = precision_recall_fscore_support(out_label_ids, preds)
        accuracy = (np.argmax(preds, axis=1) == out_label_ids).sum() / preds.shape[0]
        # accuracy = (preds == out_label_ids).int().sum() / torch.tensor(preds.shape[0], dtype=torch.float32, device=out_label_ids.device)
        results = {"val_loss": val_loss_mean, "acc": accuracy}
        temp = {}
        with open("model_output/predict/result.json",'w') as f1:
            temp['result'] = preds.tolist()
            # print(preds)
            save_str = json.dumps(temp)
            f1.write(save_str)

        ret = {k: v for k, v in results.items()}
        ret["log"] = results
        return ret, preds_list, out_label_list
    
    def test_epoch_end(self, outputs) -> dict:
        ret, predictions, targets = self._test_end(outputs)
        logs = ret["log"]
        # `val_loss` is the key returned by `self._eval_end()` but actually refers to `test_loss`
        return {"avg_test_loss": logs["val_loss"], "log": logs, "progress_bar": logs, "acc":logs["acc"]}

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='longformer/model/longformer-base-4096/', help='path to the model')
    parser.add_argument('--tokenizer', default='roberta-base')
    parser.add_argument('--train_file')
    parser.add_argument('--dev_file')
    parser.add_argument('--test_file')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--grad_accum', default=1, type=int)
    parser.add_argument('--gpus', default=1)
    parser.add_argument('--seed', default=1918, type=int)
    parser.add_argument('--fp16', default=False, action='store_true')
    parser.add_argument('--test_only', default=False, action='store_true')
    parser.add_argument('--test_checkpoint', default=None)
    parser.add_argument('--test_percent_check', default=1.0, type=float)
    parser.add_argument('--val_percent_check', default=1.0, type=float)
    parser.add_argument('--val_check_interval', default=1.0)
    parser.add_argument('--num_epochs', default=2, type=int)
    parser.add_argument('--seqlen', default=1024, type=int)
    parser.add_argument('--version', default=1, type=int)
    parser.add_argument('--do_predict', default=False, action='store_true')
    parser.add_argument('--is_small', default=False, action='store_true')
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--num_workers", default=4, type=int, help="kwarg passed to DataLoader")
    parser.add_argument("--adafactor", action="store_true")
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--num_labels', default=-1, type=int,
        help='if -1, it automatically finds number of labels.'
        'for larger datasets precomute this and manually set')
    parser.add_argument('--num_samples', default=None, type=int)
    parser.add_argument("--lr_scheduler",
        default="linear",
        choices=arg_to_scheduler_choices,
        metavar=arg_to_scheduler_metavar,
        type=str,
        help="Learning rate scheduler")
    args = parser.parse_args()
    return args

def get_train_params(args):
    train_params = {}
    train_params["precision"] = 16 if args.fp16 else 32
    train_params["distributed_backend"] = "ddp" if args.gpus > 1 else None
    train_params["accumulate_grad_batches"] = args.grad_accum
    train_params['track_grad_norm'] = -1
    train_params['val_percent_check'] = args.val_percent_check
    train_params['val_check_interval'] = args.val_check_interval
    train_params['gpus'] = args.gpus
    train_params['max_epochs'] = args.num_epochs
    return train_params

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if ',' in args.gpus:
        args.gpus = list(map(int, args.gpus.split(',')))
        args.total_gpus = len(args.gpus)
    else:
        args.gpus = int(args.gpus)
        args.total_gpus = args.gpus

    if args.test_only:
        print('loading model...')
        model = LongformerClassifier.load_from_checkpoint(args.test_checkpoint)
        model.args.num_gpus = 1
        model.args.total_gpus = 1
        model.args = args
        model.args.dev_file = args.dev_file
        model.args.test_file = args.test_file
        model.args.train_file = args.dev_file  # the model won't get trained, pass in the dev file instead to load faster
        trainer = pl.Trainer(gpus=1, test_percent_check=args.test_percent_check, train_percent_check=0.01, val_percent_check=0.01)
        trainer.test(model)

    else:
        if args.num_labels == -1:
            # Dataset will be constructred inside model, here we just want to read labels (seq len doesn't matter here)
            ds = ClassificationDataset(args.train_file, tokenizer=args.tokenizer, seqlen=args.seqlen,args=args)
            args.num_labels = len(ds.label_to_idx)
            del ds
        
        # model = LongformerClassifier.load_from_checkpoint(args.test_checkpoint)
        model = LongformerClassifier(args)

        # default logger used by trainer
        logger = TensorBoardLogger(
            save_dir=args.save_dir,
            version=args.version,
            name='pl-logs'
        )

        # second part of the path shouldn't be f-string
        filepath = f'{args.save_dir}/version_{logger.version}/checkpoints/' + 'ep-{epoch}_acc-{val_loss:.3f}'
        checkpoint_callback = ModelCheckpoint(
            filepath=filepath,
            save_top_k=3,
            verbose=True,
            monitor='val_loss',
            mode='min',
            prefix=''
        )

        extra_train_params = get_train_params(args)

        trainer = pl.Trainer(logger=logger,
                            checkpoint_callback=checkpoint_callback,
                            **extra_train_params)

        trainer.fit(model)

        if args.do_predict:
            # Optionally, predict and write to output_dir
            fpath = glob.glob(checkpoint_callback.dirpath + '/*.ckpt')[0]
            model = LongformerClassifier.load_from_checkpoint(fpath)
            model.args.num_gpus = 1
            model.args.total_gpus = 1
            model.args = args
            model.args.dev_file = args.dev_file
            model.args.test_file = args.test_file
            model.args.train_file = args.dev_file  # the model won't get trained, pass in the dev file instead to load faster
            trainer = pl.Trainer(gpus=1, test_percent_check=1.0, train_percent_check=0.01, val_percent_check=0.01, precision=extra_train_params['precision'])
            trainer.test(model)

if __name__ == '__main__':
    main()