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
        --batch_size 4 \
        --grad_accum 4 \
        --epochs 10 \
        --version 0 \
        --model_path longformer/model/longformer-encdec-base-16384 \
        # --use_tfidf \
        # --from_pretrained /hci/junchen_data/Virus_Helper/persona_output_model/version_max_input_200/_ckpt_epoch_10.ckpt \
        # --interact \
        # --test \
        # --is_small \