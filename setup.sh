#!/bin/sh

conda create -n Virus_Helper python=3.8
conda activate Virus_Helper
pip install -r requirements.txt

mkdir longformer/model
wget -P longformer/model https://ai2-s2-research.s3-us-west-2.amazonaws.com/longformer/longformer-encdec-base-16384.tar.gz
tar -zxvf longformer/model/longformer-encdec-base-16384.tar.gz

mkdir SeConD_data
gdown "https://drive.google.com/uc?export=download&id=1fY5z6wxgy7lZuWAcLMhg8bB8T41HU2EX" -O "SeConD_data/dev_tfidf_seq2seq_v2.pickle"
gdown "https://drive.google.com/uc?export=download&id=1UNiM_xvbIy47eJR8KOgefn5MeQONBfnD" -O "SeConD_data/test_tfidf_seq2seq_v2.pickle"
gdown "https://drive.google.com/uc?id=1WEDvK1Cw0qlz2UNFwMUW7L_f9oXAFShr" -O "SeConD_data/train_tfidf_seq2seq_v2.pickle"

echo "SET UP FINISHED!"