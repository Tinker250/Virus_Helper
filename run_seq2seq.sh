#!/bin/sh

python seq2seq.py \
        --train_file SeConD_data/train_tfidf_seq2seq.pickle \
        --dev_file SeConD_data/dev_tfidf_seq2seq.pickle \
        --test_file SeConD_data/test_tfidf_seq2seq.pickle \
        --save_dir seq2seq_output \
        --num_workers 24 \
        --attention_window 256 \
        --gpus 1 \
        --batch_size 4 \
        --grad_accum 4 \
        --epochs 10 \
        --model_path longformer/model/longformer-encdec-base-16384 \
        --from_pretrained /hci/junchen_data/longformer/longformer/seq2seq_output/test/_ckpt_epoch_9.ckpt \
        --interact \
        # --test \
        # --is_small \