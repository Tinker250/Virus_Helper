from audioop import mul
from cgi import test
from torch.utils import data
from rank_bm25 import BM25Okapi
import os
import time
import pickle
from tqdm import tqdm
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from rouge_score import rouge_scorer

def load_data(path):
    data = []
    with open(path,'rb') as f1:
        data = pickle.loads(f1.read())
    return data
    

def BM25_comaprsion(q,BM_25_corpus,corpus):
    return BM_25_corpus.get_top_n(q, corpus, n=1)

class myThread (threading.Thread):
    def __init__(self, examples,bm25,corpus,output):
        threading.Thread.__init__(self)
        self.examples = examples
        self.bm25 = bm25
        self.corpus = corpus
        self.output = output
    def run(self):
            print(len(self.output))
            # self.qbar.update(len(self.output))
        # print("Progress: "+str(len(self.output)/9331))

def multi_process(examples,bm25,corpus):
    output = []
    for example in examples:
        similar_context = BM25_comaprsion(example[1].split(' '),bm25,corpus)
        output.append([example[1],example[2],similar_context[0]])
    return output

def main():
    data = load_data('SeConD_data/train_tfidf_seq2seq.pickle')
    corpus = [x[1] for x in data]
    # print(len(corpus))
    tokenized_corpus = [doc.split(" ") for doc in tqdm(corpus)]
    bm25 = BM25Okapi(tokenized_corpus)
    test_examples = load_data('SeConD_data/test_tfidf_seq2seq.pickle')
    
        # print(+str((i+1)/1867))
    output = []
    all_tasks = []
    split_number = 10
    qbar = tqdm(total=9331)
    num_cores = int(mp.cpu_count())
    print("amount of CPU: " + str(num_cores))
    pool = mp.Pool(num_cores*2)
    # executor = ThreadPoolExecutor(max_workers=20)
    # int(len(test_examples)/split_number)+1
    for i in tqdm(range(0,int(len(test_examples)/split_number)+1)):
        if((i+1)*split_number > len(test_examples)):
            temp = test_examples[i*split_number:]
        else:
            temp = test_examples[i*split_number:(i+1)*split_number]
        task = pool.apply_async(multi_process, args=(temp,bm25,corpus))
        # print(len(temp))
        # print(i*split_number)
        # task = executor.submit(multi_process,temp,bm25,corpus,output)
        all_tasks.append(task)

    
    for future in all_tasks:
        output += future.get()
        localtime = time.asctime( time.localtime(time.time()) )
        print(str(localtime)+" output length = {}".format(len(output)))
        # print(kkk)
        qbar.update(len(future.get()))

    for i,item in enumerate(output):
        if(item[0] in corpus):
            index = corpus.index(str(item[2]))
            selscted_response = data[index][2]
            output[i].append(selscted_response)
        else:
            output[i].append('')
        # output[original_context,gold_response,similar_context,selected_response]

    with open("SeConD_data/BM25_comparsion.pickle",'wb') as f2:
        pickle.dump(output,f2)
        print("file saved")

def load_bm25():
    test_data = []
    all_data = load_data('SeConD_data/train_tfidf_seq2seq.pickle')
    corpus = [x[1] for x in all_data]
    with open("SeConD_data/BM25_comparsion.pickle",'rb') as f1:
        test_data = pickle.loads(f1.read())
    print(test_data[0][3])
    # number = 0
    for i,item in tqdm(enumerate(test_data)):
        index = corpus.index(item[2])
        test_data[i][3] = all_data[index][2]
    with open("SeConD_data/BM25_comparsion.pickle",'wb') as f2:
        pickle.dump(test_data,f2)
        print("file saved")
    
    # print(number)
    # output[original_context,gold_response,similar_context,selected_response]
    # data = load_data('SeConD_data/train_tfidf_seq2seq.pickle')

def rouge_score():
    BM_data = load_data("SeConD_data/BM25_comparsion.pickle")
    generated_str = [x[3] for x in BM_data]
    gold_str = [x[1] for x in BM_data]
    scorer = rouge_scorer.RougeScorer(rouge_types=['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=False)
    rouge1 = rouge2 = rougel = rougelsum = 0.0

    output_strs = []
    for ref, pred in tqdm(zip(gold_str, generated_str)):
        output_strs.append([ref,pred])
        score = scorer.score(ref, pred)
        rouge1 += score['rouge1'].fmeasure
        rouge2 += score['rouge2'].fmeasure
        rougel += score['rougeL'].fmeasure
        rougelsum += score['rougeLsum'].fmeasure
    rouge1 /= len(generated_str)
    rouge2 /= len(generated_str)
    rougel /= len(generated_str)
    rougelsum /= len(generated_str)
    print("rouge_1:{}\nrouge_2:{}\nrouge_l:{}\nrouge_sum:{}\n".format(rouge1,rouge2,rougel,rougelsum))

# main()

rouge_score()