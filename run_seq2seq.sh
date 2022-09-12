#!/bin/sh
#attention_window一般情况设置为256

python seq2seq.py \
        --train_file SeConD_data/train_tfidf_dt.pickle \
        --dev_file SeConD_data/dev_tfidf_dt.pickle \
        --test_file SeConD_data/test_tfidf_dt.pickle \
        --save_dir seq2seq_output \
        --num_workers 24 \
        --attention_window 256 \
        --max_input_len 1024 \
        --max_output_len 256 \
        --attention_mode sliding_chunks \
        --gpus 1 \
        --batch_size 4 \
        --grad_accum 4 \
        --epochs 10 \
        --version 3 \
        --model_path longformer/model/longformer-encdec-base-16384 \
        #  --use_tfidf \
        # --from_pretrained /hci/junchen_data/Virus_Helper/persona_output_model/version_max_input_200/_ckpt_epoch_10.ckpt \
        # --interact \
        # --test \
        # --is_small \