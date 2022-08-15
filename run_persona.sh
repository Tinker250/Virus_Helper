#!/bin/sh
#attention_window一般情况设置为256

python seq2seq.py \
        --train_file Persona_data/persona_train_full_tfidf.pickle \
        --dev_file Persona_data/persona_valid_tfidf.pickle \
        --test_file Persona_data/persona_valid_tfidf.pickle \
        --save_dir persona_output_model \
        --num_workers 24 \
        --attention_window 105 \
        --max_input_len 210 \
        --max_output_len 40 \
        --attention_mode sliding_chunks \
        --gpus 1 \
        --batch_size 16 \
        --grad_accum 4 \
        --epochs 15 \
        --version 5 \
        --use_tfidf \
        --model_path longformer/model/longformer-encdec-base-16384 \
        # --from_pretrained /hci/junchen_data/Virus_Helper/persona_output_model/version_max_input_200/_ckpt_epoch_10.ckpt \
        # --interact \
        # --test \
        # --is_small \