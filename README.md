# Genreative-based chatbot model
Reponse generation dialogue system on SeConD data

## Use download.sh to download the longformer model and SeConD dataset

## Build a new conda enviroment and install all dependencies
```
conda create -n Virus_Helper python=3.8
conda activate Virus_Helper
pip install -r requirements_2.txt

find ~/anaconda3/envs/Virus_Helper/lib/python3.8/site-packages/transformers/modeling_bart.py"
modify line 561 to:
if encoder_padding_mask is not None:
  if(encoder_padding_mask.dim()==3):
    encoder_padding_mask = encoder_padding_mask[:,0,:]

```

## Use run_seq2seq.sh to start training
create a floder as log
bash run_seq2seq.sh >log/seq2seq_tf_idf_top100_*1.log 2>&1 &
