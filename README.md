# Genreative-based chatbot model
Reponse generation dialogue system on SeConD data

### Set up a new conda enviroment and download the data
Run the setup.sh file using `bash setup.sh`

### Modify the modeling_bart.py file

```
find ~/anaconda3/envs/Virus_Helper/lib/python3.8/site-packages/transformers/modeling_bart.py"
modify line 561 to:
if encoder_padding_mask is not None:
  if(encoder_padding_mask.dim()==3):
    encoder_padding_mask = encoder_padding_mask[:,0,:]
```
### Use run_seq2seq.sh to start training
Use `conda activate Virus_Helper` to activate the enviroment

Create a floder as log run `bash run_seq2seq.sh >log/seq2seq_tf_idf_top100_*1.log 2>&1 &`
### Change hyper-parameters
Search `#TODO: TD` in `seq2seq.py` and `longformer/longformer.py` to modify the coefficient of attention weights

Change other hypter-parameter in run_seq2seq.sh
