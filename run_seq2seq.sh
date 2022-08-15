#!/bin/sh
#attention_window一般情况设置为256

python seq2seq.py \
        --train_file SeConD_data/train_tfidf_seq2seq_v2.pickle \
        --dev_file SeConD_data/dev_tfidf_seq2seq_v2.pickle \
        --test_file SeConD_data/test_tfidf_seq2seq_v2.pickle \
        --save_dir persona_output_model \
        --num_workers 24 \
        --attention_window 256 \
        --max_input_len 1024 \
        --max_output_len 256 \
        --attention_mode sliding_chunks \
        --gpus 1 \
        --batch_size 4 \
        --grad_accum 4 \
        --epochs 10 \
        --version 5 \
        --use_tfidf \
        --model_path longformer/model/longformer-encdec-base-16384 \
        # --from_pretrained /hci/junchen_data/Virus_Helper/persona_output_model/version_max_input_200/_ckpt_epoch_10.ckpt \
        # --interact \
        # --test \
        # --is_small \