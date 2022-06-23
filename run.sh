#!/bin/sh

python sentence_classcification.py \
        --train_file SeConD_data/train_tfidf_v2.pickle \
        --dev_file SeConD_data/dev_tfidf.pickle \
        --test_file SeConD_data/test_tfidf.pickle \
        --save_dir model_output \
        --batch_size 4 \
        --grad_accum 4 \
        --seqlen 1024 \
        --gpus 1 \
        --num_labels 2 \
        --is_small \
        --version 9 \
        # --test_checkpoint /hci/junchen_data/longformer/longformer/model_output/version_10/checkpoints/longformer_original_mask_1.ckpt \
        # --test_only \
        
