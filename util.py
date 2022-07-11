import os
import nltk
from nltk.grammar import DependencyGrammar
from nltk.parse import DependencyGraph, ProjectiveDependencyParser, NonprojectiveDependencyParser
from nltk.parse.corenlp import CoreNLPDependencyParser
import pickle
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
import string
from tqdm import tqdm
import multiprocessing as mp
import re
import torch
from scipy import sparse
import time

class sentence_processor():
    def __init__(self,text_sentence):
        self.sentence = text_sentence
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    def dependency_tree_build(self):
        # print(self.sentence)
        parser = CoreNLPDependencyParser(url='http://localhost:9000')
        return parser.raw_parse(self.sentence)
    
    @staticmethod    
    def rebuild_sentence_from_bart_tokens(tokenizer,sentence):
        tokens = tokenizer.tokenize(sentence)
        
        output_str = ""
        for i, token in enumerate(tokens):
            space = ' '
            clean_token = re.sub(r'Ġ','',token)
            if(i == 0):
                space = ''
            if(token == 'Ġ'):
                output_str += space+'<em>'
            elif(token == 'Ġcannot'):
                output_str += space+'<cannot>'
            elif(token == "Ġdunno"):
                output_str += space+'<dunno>'
            elif(token in ["Ġgonna","Ġgotta", "Ġaint", "Ġwanna",'aint']):
                output_str += space+'<abs>'
            elif(re.match(r'Ġ.*no*t',token)):
                output_str += space+'<dont>'
            elif(re.match(r'^[Ġ\|\?\~\`\!\@\#\$\%\^\&\*\(\)\_\-\+\=\{\}\[\]\'\"\:\;\,\<\.\>\\\/]{2,}$',token)):
                output_str += space+'<cp>' #continue punctuation
            elif(re.findall(r'[^\|\?\~\`\!\@\#\$\%\^\&\*\(\)\_\-\+\=\{\}\[\]\'\"\:\;\,\<\.\>\\\/0-9a-zA-Z]{1,}',clean_token)):
                output_str += space+'<ur>' #unrecogised
            elif(re.match(r'^[Ġ0-9]{1}\d+$',token)):
                output_str += space+'<pn>' #pure number
            elif(re.match(r'^(?:(?=.*[\|\?\~\`\!\@\#\$\%\^\&\*\(\)\_\-\+\=\{\}\[\]\'\"\:\;\,\<\.\>\\\/])(?=.*[a-zA-Z])).{2,}',token)):
                output_str += space+'<ppw>' # punctuation plus word
            else:
                output_str += space+re.sub(r'Ġ','',token)
        return output_str,tokens

    def adjacency_matrix_build(self):
        self.sentence,tokens_ori = self.rebuild_sentence_from_bart_tokens(self.tokenizer,self.sentence)
        # print(self.sentence)
        main_rel = ["nsubj", "obj", "iobj"]
        minor_rel = ["csubj", "ccomp", "xcomp", "nmod", "appos", "nummod","amod"]
        result, = self.dependency_tree_build()
        
        tokens =[]
        word_type = []
        ref_index = []
        relation = []

        conll_result = result.to_conll(4).split('\n')
        # TOKEN   TYPE    REF     RELATION
        # The     DT      4       det
        # quick   JJ      4       amod
        for  row in conll_result[:-1]:
            item = row.split('\t')
            tokens.append(item[0])
            word_type.append(item[1])
            ref_index.append(item[2])
            relation.append(item[3])
    
        dict_data = pd.DataFrame({"tokens":tokens,"type":word_type,"ref_index":ref_index,"relation":relation})
        adjacency_matrix = np.zeros((len(dict_data), len(dict_data)))
        # print(dict_data)

        test_list = self.sentence.split(' ')
        if(len(dict_data)!=len(test_list)):
            for i in range(0,len(test_list)):
                if(test_list[i] != dict_data.iat[i,0]):
                    print(test_list[i]+'----->'+dict_data.iat[i,0])
                    print(tokens_ori)
                    print(kkk)
        # print(len(dict_data))
        
        # print(kkk
        for i, row in dict_data.iterrows():
            # print(row)
            if(row[3] in main_rel):
                adjacency_matrix[i,int(row[2])-1] = 1
                adjacency_matrix[int(row[2])-1,i] = 1
            if(row[3] in minor_rel):
                adjacency_matrix[i,int(row[2])-1] = 0.5
                adjacency_matrix[int(row[2])-1,i] = 0.5
        
        # padding 和 truncate
        padding_length = 1024-adjacency_matrix.shape[0]
        if(padding_length>0):
            zero_fill_t1 = np.zeros((adjacency_matrix.shape[0],padding_length))
            adjacency_matrix = np.column_stack((adjacency_matrix,zero_fill_t1))
            zero_fill_t2 = np.zeros((padding_length,adjacency_matrix.shape[1]))
            adjacency_matrix = np.row_stack((adjacency_matrix,zero_fill_t2))
        elif(padding_length<0):
            padding_length = -padding_length
            adjacency_matrix = np.delete(adjacency_matrix,[x for x in range(padding_length)],axis=0)
            adjacency_matrix = np.delete(adjacency_matrix,[x for x in range(padding_length)],axis=1)
        
        assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1] == 1024


        #返回sci的稀疏矩阵，减小数据存储的大小和计算的难度
        return sparse.csr_matrix(adjacency_matrix)


def load_data(file_path):
    with open(file_path,'rb') as fin:
        data = pickle.loads(fin.read())
    return data

def multi_process(examples,index):
    output = []
    for i,example in enumerate(examples):
        previous_matrix = None
        sp = sentence_processor(example[1])
        temp = sp.adjacency_matrix_build()
        if(previous_matrix is not None):
            zero_fill_t1 = np.zeros((len(temp),1))
            zero_fill_t2 = np.zeros((1,1+len(temp)))
            temp = np.column_stack((temp,zero_fill_t1))
            temp = np.row_stack((temp,zero_fill_t2))
            col = len(temp)
            row = previous_matrix.shape[0]
            zero_fill = np.zeros((row,col))
            previous_matrix = np.column_stack((previous_matrix,zero_fill))
            # print(temp.shape)
            # print(previous_matrix.shape)
            temp = np.column_stack((zero_fill.T,temp))
            previous_matrix = np.row_stack((previous_matrix,temp))
            print(previous_matrix.shape)
        else:
            previous_matrix = temp
            # print(previous_matrix.shape)
            # print(np.sum(previous_matrix))
        temp_output = {}
        temp_output[index*10+i]=previous_matrix
        output.append(temp_output)
        # print(output)
    return output

def build_async_main():
    data = load_data("SeConD_data/dev_tfidf_seq2seq_v2.pickle")
    output = []
    all_tasks = []
    split_number = 10
    qbar = tqdm(total=len(data))
    num_cores = int(mp.cpu_count())
    print("amount of CPU: " + str(num_cores))
    pool = mp.Pool(num_cores*2)
    # executor = ThreadPoolExecutor(max_workers=20)
    # int(len(test_examples)/split_number)+1
    for i in tqdm(range(0,int(len(data)/split_number)+1)):
        if((i+1)*split_number > len(data)):
            temp = data[i*split_number:]
        else:
            temp = data[i*split_number:(i+1)*split_number]
        task = pool.apply_async(multi_process, args=(temp,i))
        # print(len(temp))
        # print(i*split_number)
        # task = executor.submit(multi_process,temp,bm25,corpus,output)
        all_tasks.append(task)
    
    for future in all_tasks:
        output += future.get()
        localtime = time.asctime( time.localtime(time.time()) )
        print(str(localtime)+" output length = {}".format(len(output)))
        # print(kkk)
        if(len(output)%5000 == 0 or len(output) == len(data)):
            with open("SeConD_data/dependency_tree/dev_dependency_tree_part_"+str(len(output))+".pickle",'wb') as f1:
                pickle.dump(output,f1)
                print("file saved with lenth = {}".format(len(output)))
        qbar.update(len(future.get()))
    

def build_full_metrix():
    data = load_data("SeConD_data/test_tfidf_seq2seq_v2.pickle")
    save_data = []
    for item in tqdm(data):
        previous_matrix = None
        for sentence in item[1].split('</sbs>'):
            # print(len(item[1].split('</s>')))
            # sentence ="C :\ Program Files <cp> x <pn> )\\ Cond uit"
            sp = sentence_processor(sentence)
            temp = sp.adjacency_matrix_build()
            if(previous_matrix is not None):
                zero_fill_t1 = np.zeros((len(temp),1))
                zero_fill_t2 = np.zeros((1,1+len(temp)))
                temp = np.column_stack((temp,zero_fill_t1))
                temp = np.row_stack((temp,zero_fill_t2))
                col = len(temp)
                row = previous_matrix.shape[0]
                zero_fill = np.zeros((row,col))
                previous_matrix = np.column_stack((previous_matrix,zero_fill))
                # print(temp.shape)
                # print(previous_matrix.shape)
                temp = np.column_stack((zero_fill.T,temp))
                previous_matrix = np.row_stack((previous_matrix,temp))
                print(previous_matrix.shape)
            else:
                previous_matrix = temp
                # print(previous_matrix.shape)
                # print(np.sum(previous_matrix))
        save_data.append(previous_matrix)
    with open("SeConD_data/test_dependency_tree.pickle",'wb') as f1:
        pickle.dump(save_data,f1)

def convert_matrix_to_sparse(input_matrix):
    sparse_mx = sparse.csr_matrix(input_matrix)
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


if __name__ == "__main__":
    # test = re.findall(r'Ġ*[^\|\?\~\`\!\@\#\$\%\^\&\*\(\)\_\-\+\=\{\}\[\]\'\"\:\;\,\<\.\>\\\/0-9a-zA-Z]{1,}$',"Ġhello")
    # print(test)
    # build_full_metrix()
    build_async_main()
    # test = np.array([[0,0,1],[2,0,0],[0,30,0]])
    # test = sparse.csr_matrix(test)
    # test = convert_matrix_to_sparse(test)
    
    # result = list(result)
    # result.pretty_print()
    # for head, rel, dep in result.triples():
    #     print(head,rel,dep)
    # TOKEN   TYPE    REF     RELATION
    # The     DT      4       det
    # quick   JJ      4       amod
   
    # print()