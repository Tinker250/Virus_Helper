#!/bin/sh

python seq2seq.py \
        --train_file SeConD_data/train_tfidf_seq2seq_v2.pickle \
        --dev_file SeConD_data/dev_tfidf_seq2seq_v2.pickle \
        --test_file SeConD_data/test_tfidf_seq2seq_v2.pickle \
        --save_dir seq2seq_output \
        --num_workers 24 \
        --attention_window 256 \
        --gpus 1 \
        --batch_size 2 \ #TODO:change the size up to your GPU
        --grad_accum 2 \
        --epochs 10 \
        --version 0 \
        --model_path longformer/model/longformer-encdec-base-16384 \
        # --from_pretrained /hci/junchen_data/longformer/longformer/seq2seq_output/test/_ckpt_epoch_9.ckpt \
        # --interact \
        # --test \
        # --is_small \