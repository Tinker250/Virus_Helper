#!/bin/sh
#attention_window一般情况设置为256

python seq2seq.py \
        --train_file Daily_dialogue/train_tfidf.pickle \
        --dev_file Daily_dialogue/dev_tfidf.pickle \
        --test_file Daily_dialogue/test_tfidf.pickle \
        --save_dir daily_dialogue_output_model \
        --num_workers 24 \
        --attention_window 105 \
        --max_input_len 210 \
        --max_output_len 40 \
        --attention_mode sliding_chunks \
        --gpus 1 \
        --batch_size 36 \
        --grad_accum 5 \
        --epochs 15 \
        --version 6 \
        --model_path longformer/model/longformer-encdec-base-16384 \
        --use_tfidf \
        # --from_pretrained /hci/junchen_data/Virus_Helper/persona_output_model/version_max_input_200/_ckpt_epoch_10.ckpt \
        # --interact \
        # --test \
        # --is_small \