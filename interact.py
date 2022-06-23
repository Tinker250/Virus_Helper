import json
from os.path import abspath, dirname, exists, join
import argparse
import logging
from tqdm import trange
import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import socket
import os, sys
from longformer import LongformerEncoderDecoderForConditionalGeneration, LongformerEncoderDecoderConfig
from longformer.sliding_chunks import pad_to_window_size
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
import logging
from functools import partial
import argparse
import subprocess as sp
import pytorch_lightning as pl

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

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


def cut_seq_to_eos(sentence, remove_id=[-1]):
    sent=[]
    for s in sentence:
        if s in remove_id:
            continue
        if s != EOS_ID:
            sent.append(s)
        else:
            break
    return sent


### FROM HUGGING FACE REPO
def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value
    return logits

def generate_next_token(model, input_ids, position_ids=None, token_type_ids=None, prev=None, temperature=1, top_k=0, top_p=0, past=None):
    with torch.no_grad():
        if not past:
            hidden_states, past = model.transformer(prev, position_ids, token_type_ids, past=past)
        else:
            hidden_states, past = model.transformer(prev, past=past)
        logits = model.lm_head(hidden_states)
        logits = logits[0, -1, :] / temperature
        logits = top_filtering(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(logits.unsqueeze(0), dim=-1)
        prev = torch.multinomial(probs, num_samples=1)
        return prev, probs[0][prev], past

def generate_sequence(model, input_ids, position_ids=None, token_type_ids=None, temperature=1, top_k=0, top_p=0, length=20, past=None, device='cuda'):
    output = input_ids.new_zeros([input_ids.size(0),0])
    prev = input_ids
    for i in range(length):
        prev, probs, past = generate_next_token(model, input_ids, position_ids, token_type_ids, prev, temperature, top_k, top_p, past)
        output = torch.cat((output, prev), dim=1)
    return output

def cut_seq_to_eos(sentence, remove_id=[-1]):
    sent=[]
    for s in sentence:
        if s in remove_id:
            continue
        if s != EOS_ID:
            sent.append(s)
        else:
            break
    return sent

def load_model(model, checkpoint):
    if checkpoint is None or checkpoint == "None":
        print("empty checkpoint")
    else:
        if not os.path.exists(checkpoint):
            raise ValueError('checkpoint %s not exist' % checkpoint)
        
        model_state_dict = torch.load(checkpoint)

        start_model = model
        start_model.load_state_dict(model_state_dict)
    return model

def run_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='', help='pretrained model name or path to local checkpoint')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_input_len", type=int, default=1024,
                            help="maximum num of wordpieces/summary. Used for training and testing")
    parser.add_argument("--model_path", type=str, default='facebook/bart-base',
                            help="Path to the checkpoint directory or model name")
    parser.add_argument("--tokenizer", type=str, default='facebook/bart-base')
    parser.add_argument("--attention_window", type=int, default=256, help="Attention window")
    parser.add_argument("--max_output_len", type=int, default=256,
                            help="maximum num of wordpieces/summary. Used for training and testing")
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument("--from_pretrained", type=str, default=None,
                            help="Path to a checkpoint to load model weights but not training state")
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)


    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    n_gpu = torch.cuda.device_count()
    args.device, args.n_gpu = device, n_gpu

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    #### load the trained model 
    config = LongformerEncoderDecoderConfig.from_pretrained(args.model_path)
    config.attention_mode = 'sliding_chunks'
    config.attention_window = [args.attention_window] * config.encoder_layers
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    model = LongformerEncoderDecoderForConditionalGeneration.from_pretrained(args.model_path, config=config)
    model = pl.LightningModule.load_from_checkpoint(args.from_pretrained)
    
    model.to(device)
    model.eval()

    history = ""
    while True:
        raw_text = input("USR >>> ")
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input("USR >>> ")
        history += raw_text+' </s> '
        input_ids = tokenizer.encode(history, truncation=True, max_length=args.max_input_len)
        
        padding_length = args.max_input_len - len(input_ids)
        input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
        input_ids = torch.tensor(input_ids)
        input_ids = torch.reshape(input_ids,(1,args.max_input_len))
        print(tokenizer.pad_token_id)
        print(input_ids)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        attention_mask[input_ids == tokenizer.pad_token_id] = 0
        attention_mask = torch.reshape(attention_mask,(1,args.max_input_len))
        print(attention_mask)
        # print(kkk)

        generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask,use_cache=True, max_length=args.max_output_len,num_beams=1)
        generated_str = tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        
        print("SYS >>> ", generated_str)
        history += generated_str+' </s> '

if __name__ == '__main__':

    PYTHON_EXE = 'python'
    MODEL_FOLDER = './models'
    DATA_FOLDER = './data'
    
    run_model()