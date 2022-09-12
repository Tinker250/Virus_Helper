#!/bin/sh
#attention_window一般情况设置为256

python seq2seq.py \
        --train_file DSTC7_AVSD/DSTC7_train_full_tfidf.pickle \
        --dev_file DSTC7_AVSD/DSTC7_dev_full_tfidf.pickle \
        --test_file DSTC7_AVSD/DSTC7_test_full_tfidf.pickle \
        --save_dir DSTC_output_model \
        --num_workers 24 \
        --attention_window 256 \
        --max_input_len 1024 \
        --max_output_len 256 \
        --attention_mode sliding_chunks \
        --gpus 1 \
        --batch_size 16 \
        --grad_accum 4 \
        --epochs 10 \
        --version 1 \
        --model_path longformer/model/longformer-encdec-base-16384 \
        --use_tfidf \
        --test \
        --from_pretrained /hci/junchen_data/Virus_Helper/DSTC_output_model/version_tfidf_top0.8_0.2/_ckpt_epoch_7_v0.ckpt \
        # --interact \
        # --test \
        # --is_small \