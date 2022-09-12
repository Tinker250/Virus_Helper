from mimetypes import init
from os import devnull
from subprocess import call
from nltk.parse.corenlp import CoreNLPDependencyParser
import pickle
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import multiprocessing as mp
import re
import torch
from scipy import sparse
import json
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
        # for head, rel, dep in result.triples():
        #     print(head,rel,dep)
        
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
        # print(adjacency_matrix)
        # print(kkk)
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
    # print(len(examples),index)
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
            # print(previous_matrix.shape)
        else:
            previous_matrix = temp
            # print(previous_matrix.shape)
            # print(np.sum(previous_matrix))
        temp_output = {}
        temp_output[index+i]=previous_matrix
        # print(index+i,index,i)
        output.append(temp_output)
        # print(output)

    # sp = sentence_processor(examples[1])
    # temp = sp.adjacency_matrix_build()
    # output = {}
    # output[index] = temp
    return output

def build_async_main(data_type,data_dir):
    data = load_data("{}/{}_tfidf.pickle".format(data_dir,data_type))
    # data = data[2120:]
    #callback函数的全局记录变量
    output = []
    split_number = 20
    all_tasks = []
    qbar = tqdm(total=len(data))
    num_cores = int(mp.cpu_count())
    print("amount of CPU: " + str(num_cores))
    pool = mp.Pool(num_cores*2)
    
    def callback_add_denpendency(result):
        output.extend(result)
        localtime = time.asctime( time.localtime(time.time()))
        print(str(localtime)+" output length = {}".format(len(output)))
        # print(kkk)
        if(len(output)%5000 == 0 or len(output) == len(data)):
            with open("{}/dependency_tree/{}_dependency_tree_part_".format(data_dir,data_type)+str(len(output))+".pickle",'wb') as f1:
                pickle.dump(output,f1)
                print("file saved with lenth = {}".format(len(output)))
        qbar.update(len(result))
    split_time = 0
    if(len(data)%split_number==0):
        split_time = len(data)//split_number
    else:
        split_time = len(data)//split_number + 1
    for i in range(split_time):
        # print(i)
        if((i+1)*split_number > len(data)):
            temp = data[i*split_number:]
            # print(i*split_number,'last')
        else:
            temp = data[i*split_number:(i+1)*split_number]
            # print(i*split_number,(i+1)*split_number)
        #回调
        pool.apply_async(multi_process, args=(temp,i*split_number),callback=callback_add_denpendency)
        #不使用回调
        # task = pool.apply_async(multi_process, args=(temp,i*split_number),callback=callback_add_denpendency)
        # all_tasks.append(task)

    # for i,item in enumerate(data):
    #     task = pool.apply_async(multi_process, args=(item,i))
    #     all_tasks.append(task)
    
    #不使用回调
    # for future in all_tasks:
    #     output += future.get()
    #     localtime = time.asctime( time.localtime(time.time()))
    #     print(str(localtime)+" output length = {}".format(len(output)))
    #     # print(kkk)
    #     if(len(output)%5000 == 0 or len(output) == len(data)):
    #         with open("SeConD_data/dependency_tree/dev_dependency_tree_part_"+str(len(output))+".pickle",'wb') as f1:
    #             pickle.dump(output,f1)
    #             print("file saved with lenth = {}".format(len(output)))
    #     qbar.update(len(future.get()))

    pool.close()
    pool.join()
    
    with open("{}/dependency_tree/{}_dependency_tree_part_".format(data_dir,data_type)+str(len(output))+".pickle",'wb') as f1:
        pickle.dump(output,f1)
        print("file saved with lenth = {}".format(len(output)))
    
    # for future in all_tasks:
    #     output += future.get()
    #     localtime = time.asctime( time.localtime(time.time()))
    #     print(str(localtime)+" output length = {}".format(len(output)))
    #     # print(kkk)
    #     if(len(output)%5000 == 0 or len(output) == len(data)):
    #         with open("SeConD_data/dependency_tree/dev_dependency_tree_part_"+str(len(output))+".pickle",'wb') as f1:
    #             pickle.dump(output,f1)
    #             print("file saved with lenth = {}".format(len(output)))
    #     qbar.update(len(future.get()))
    
missing_train = [1581, 1582, 1583, 1584, 1585, 1586, 1587, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1595, 1596, 1597, 1598, 1599, 1600, 4021, 4022, 4023, 4024, 4025, 4026, 4027, 4028, 4029, 4030, 4031, 4032, 4033, 4034, 4035, 4036, 4037, 4038, 4039, 4040, 4101, 4102, 4103, 4104, 4105, 4106, 4107, 4108, 4109, 4110, 4111, 4112, 4113, 4114, 4115, 4116, 4117, 4118, 4119, 4120, 4501, 4502, 4503, 4504, 4505, 4506, 4507, 4508, 4509, 4510, 4511, 4512, 4513, 4514, 4515, 4516, 4517, 4518, 4519, 4520, 4641, 4642, 4643, 4644, 4645, 4646, 4647, 4648, 4649, 4650, 4651, 4652, 4653, 4654, 4655, 4656, 4657, 4658, 4659, 4660, 4941, 4942, 4943, 4944, 4945, 4946, 4947, 4948, 4949, 4950, 4951, 4952, 4953, 4954, 4955, 4956, 4957, 4958, 4959, 4960, 5981, 5982, 5983, 5984, 5985, 5986, 5987, 5988, 5989, 5990, 5991, 5992, 5993, 5994, 5995, 5996, 5997, 5998, 5999, 6000, 10041, 10042, 10043, 10044, 10045, 10046, 10047, 10048, 10049, 10050, 10051, 10052, 10053, 10054, 10055, 10056, 10057, 10058, 10059, 10060, 15241, 15242, 15243, 15244, 15245, 15246, 15247, 15248, 15249, 15250, 15251, 15252, 15253, 15254, 15255, 15256, 15257, 15258, 15259, 15260, 18581, 18582, 18583, 18584, 18585, 18586, 18587, 18588, 18589, 18590, 18591, 18592, 18593, 18594, 18595, 18596, 18597, 18598, 18599, 18600, 22541, 22542, 22543, 22544, 22545, 22546, 22547, 22548, 22549, 22550, 22551, 22552, 22553, 22554, 22555, 22556, 22557, 22558, 22559, 22560, 23361, 23362, 23363, 23364, 23365, 23366, 23367, 23368, 23369, 23370, 23371, 23372, 23373, 23374, 23375, 23376, 23377, 23378, 23379, 23380, 24681, 24682, 24683, 24684, 24685, 24686, 24687, 24688, 24689, 24690, 24691, 24692, 24693, 24694, 24695, 24696, 24697, 24698, 24699, 24700, 28041, 28042, 28043, 28044, 28045, 28046, 28047, 28048, 28049, 28050, 28051, 28052, 28053, 28054, 28055, 28056, 28057, 28058, 28059, 28060, 28701, 28702, 28703, 28704, 28705, 28706, 28707, 28708, 28709, 28710, 28711, 28712, 28713, 28714, 28715, 28716, 28717, 28718, 28719, 28720, 32301, 32302, 32303, 32304, 32305, 32306, 32307, 32308, 32309, 32310, 32311, 32312, 32313, 32314, 32315, 32316, 32317, 32318, 32319, 32320, 36081, 36082, 36083, 36084, 36085, 36086, 36087, 36088, 36089, 36090, 36091, 36092, 36093, 36094, 36095, 36096, 36097, 36098, 36099, 36100, 36781, 36782, 36783, 36784, 36785, 36786, 36787, 36788, 36789, 36790, 36791, 36792, 36793, 36794, 36795, 36796, 36797, 36798, 36799, 36800, 39121, 39122, 39123, 39124, 39125, 39126, 39127, 39128, 39129, 39130, 39131, 39132, 39133, 39134, 39135, 39136, 39137, 39138, 39139, 39140, 39541, 39542, 39543, 39544, 39545, 39546, 39547, 39548, 39549, 39550, 39551, 39552, 39553, 39554, 39555, 39556, 39557, 39558, 39559, 39560, 42061, 42062, 42063, 42064, 42065, 42066, 42067, 42068, 42069, 42070, 42071, 42072, 42073, 42074, 42075, 42076, 42077, 42078, 42079, 42080, 44561, 44562, 44563, 44564, 44565, 44566, 44567, 44568, 44569, 44570, 44571, 44572, 44573, 44574, 44575, 44576, 44577, 44578, 44579, 44580, 44821, 44822, 44823, 44824, 44825, 44826, 44827, 44828, 44829, 44830, 44831, 44832, 44833, 44834, 44835, 44836, 44837, 44838, 44839, 44840, 47881, 47882, 47883, 47884, 47885, 47886, 47887, 47888, 47889, 47890, 47891, 47892, 47893, 47894, 47895, 47896, 47897, 47898, 47899, 47900, 48941, 48942, 48943, 48944, 48945, 48946, 48947, 48948, 48949, 48950, 48951, 48952, 48953, 48954, 48955, 48956, 48957, 48958, 48959, 48960, 51101, 51102, 51103, 51104, 51105, 51106, 51107, 51108, 51109, 51110, 51111, 51112, 51113, 51114, 51115, 51116, 51117, 51118, 51119, 51120, 51821, 51822, 51823, 51824, 51825, 51826, 51827, 51828, 51829, 51830, 51831, 51832, 51833, 51834, 51835, 51836, 51837, 51838, 51839, 51840, 57521, 57522, 57523, 57524, 57525, 57526, 57527, 57528, 57529, 57530, 57531, 57532, 57533, 57534, 57535, 57536, 57537, 57538, 57539, 57540, 57801, 57802, 57803, 57804, 57805, 57806, 57807, 57808, 57809, 57810, 57811, 57812, 57813, 57814, 57815, 57816, 57817, 57818, 57819, 57820, 61161, 61162, 61163, 61164, 61165, 61166, 61167, 61168, 61169, 61170, 61171, 61172, 61173, 61174, 61175, 61176, 61177, 61178, 61179, 61180, 62141, 62142, 62143, 62144, 62145, 62146, 62147, 62148, 62149, 62150, 62151, 62152, 62153, 62154, 62155, 62156, 62157, 62158, 62159, 62160, 66221, 66222, 66223, 66224, 66225, 66226, 66227, 66228, 66229, 66230, 66231, 66232, 66233, 66234, 66235, 66236, 66237, 66238, 66239, 66240, 66321, 66322, 66323, 66324, 66325, 66326, 66327, 66328, 66329, 66330, 66331, 66332, 66333, 66334, 66335, 66336, 66337, 66338, 66339, 66340, 66961, 66962, 66963, 66964, 66965, 66966, 66967, 66968, 66969, 66970, 66971, 66972, 66973, 66974, 66975, 66976, 66977, 66978, 66979, 66980, 67281, 67282, 67283, 67284, 67285, 67286, 67287, 67288, 67289, 67290, 67291, 67292, 67293, 67294, 67295, 67296, 67297, 67298, 67299, 67300]
missing_dev = [2120, 2121, 2122, 2123, 2124, 2125, 2126, 2127, 2128, 2129, 2130, 2131, 2132, 2133, 2134, 2135, 2136, 2137, 2138, 2139, 5080, 5081, 5082, 5083, 5084, 5085, 5086, 5087, 5088, 5089, 5090, 5091, 5092, 5093, 5094, 5095, 5096, 5097, 5098, 5099, 5760, 5761, 5762, 5763, 5764, 5765, 5766, 5767, 5768, 5769, 5770, 5771, 5772, 5773, 5774, 5775, 5776, 5777, 5778, 5779, 9280, 9281, 9282, 9283, 9284, 9285, 9286, 9287, 9288, 9289, 9290, 9291, 9292, 9293, 9294, 9295, 9296, 9297, 9298, 9299]

def build_full_metrix(data_type):
    data = load_data("SeConD_data/{}_tfidf_seq2seq_v2.pickle".format(data_type))
    save_data = []
    # for item in tqdm(data):
    
    # previous_matrix = None
    # for sentence in data[missing_dev[25]][1].split('</s>'):
    #         # print(len(item[1].split('</s>')))
    #         # sentence ="C :\ Program Files <cp> x <pn> )\\ Cond uit"
    #     sp = sentence_processor(sentence)
    #     temp = sp.adjacency_matrix_build()
    #     if(previous_matrix is not None):
    #         zero_fill_t1 = np.zeros((len(temp),1))
    #         zero_fill_t2 = np.zeros((1,1+len(temp)))
    #         temp = np.column_stack((temp,zero_fill_t1))
    #         temp = np.row_stack((temp,zero_fill_t2))
    #         col = len(temp)
    #         row = previous_matrix.shape[0]
    #         zero_fill = np.zeros((row,col))
    #         previous_matrix = np.column_stack((previous_matrix,zero_fill))
    #             # print(temp.shape)
    #             # print(previous_matrix.shape)
    #         temp = np.column_stack((zero_fill.T,temp))
    #         previous_matrix = np.row_stack((previous_matrix,temp))
    #         print(previous_matrix.shape)
    #     else:
    #         previous_matrix = temp
                # print(previous_matrix.shape)
                # print(np.sum(previous_matrix))
    
    # print(len(data[missing_dev[62]][1]))
    # sp = sentence_processor(data[missing_dev[62]][1])
    # sp.adjacency_matrix_build()

    # count = 0
    # for index in tqdm(missing_dev):
    #     if(len(data[index][1].split(' '))<3):
    #         count += 1
    # print(count)
    output_data = []
    for i, conv in tqdm(enumerate(data)):
        if(len(conv[1].split(' '))>=3):
            output_data.append(conv)
    with open("SeConD_data/{}_tfidf_seq2seq_v3.pickle".format(data_type),'wb') as f1:
        pickle.dump(output_data,f1)
        print(len(output_data))

def merge_dependency_tree(pickle_path,dependency_tree_path,out_path):
    data_original = load_data(pickle_path)
    data_dependency_tree = load_data(dependency_tree_path)
    assert len(data_original) == len(data_dependency_tree), "length different: "+str(len(data_original))+' '+str(len(data_dependency_tree))
    for i, dependency in tqdm(enumerate(data_dependency_tree)):
        index = int(list(dependency.keys())[0])
        data_original[index].append(dependency[index])
    with open(out_path,'wb') as f1:
        pickle.dump(data_original,f1)
        print("Save data with TF_IDF and dependency tree")
        # sp = sentence_processor(data[index][1])
        # save_data.append(sp.adjacency_matrix_build())
    # with open("SeConD_data/{}_dependency_tree_missing.pickle",format(data_type),'wb') as f1:
    #     pickle.dump(save_data,f1)

def convert_matrix_to_sparse(input_matrix):
    sparse_mx = sparse.csr_matrix(input_matrix)
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class persona_processor():
    def __init__(self,path) -> None:
        self.path = path
        
    @staticmethod
    def load_data(path):
        with open(path,'r') as f1:
            data = json.loads(f1.read())
        return data
    def build_pickle(self,data_type):
        data = self.load_data(self.path)
        sub_data = data[data_type]
        output = []
        for conv in tqdm(sub_data):
            for sub_conv in conv['utterances']:
                # print(' </s> '.join(sub_conv['history']),"----------",sub_conv['candidates'][-1])
                output.append(['1',' </s> '.join(sub_conv['history']),sub_conv['candidates'][-1]])
            # print(kkk)
        with open('Persona_data/persona_{}_full.pickle'.format(data_type),'wb') as f2:
            pickle.dump(output,f2)
            print("dataset size is: {}".format(len(output)))
            print("build finished")

class DSTC_processor():
    def __init__(self,path) -> None:
        self.path = path
        
    @staticmethod
    def load_data(path):
        with open(path,'r') as f1:
            data = json.loads(f1.read())
        return data
    def build_pickle(self,data_type):
        data = self.load_data(self.path)
        output = []
        for conv in tqdm(data['dialogs']):
            summary = conv['summary']
            caption = conv['caption']
            history = caption+' </s> '+ summary
            for sub_conv in conv['dialog'][:-1]:
                # print(sub_conv)
                history += ' </s> '+sub_conv['question']+' </s> '+sub_conv['answer']
            history += ' </s> '+conv['dialog'][-1]['question']
                # print(' </s> '.join(sub_conv['history']),"----------",sub_conv['candidates'][-1])
            output.append(['1',history,conv['dialog'][-1]['answer']])
        with open('DSTC7_AVSD/DSTC7_{}_full.pickle'.format(data_type),'wb') as f2:
            pickle.dump(output,f2)
            print("dataset size is: {}".format(len(output)))
            print("build finished")
    
def replace_long_path(path_in,path_out):
    data = load_data(path_in)
    output = []
    for item in tqdm(data):
        temp = []
        temp.append(item[0])
        if(list(filter(lambda x: len(x)>600,item[1].split(' ')))):
            split_words = item[1].split(' ')
            replaced_str = ' '.join([x if len(x)<200 else "[long_path]" for x in split_words ])
            # print(item[1])
            # print("*"*30)
            # print(replaced_str)
            # print(kkk)
            temp.append(replaced_str)
            temp.append(item[2])
        else:
            temp.append(item[1])
            temp.append(item[2])
        output.append(temp)
        
    with open(path_out,'wb') as f1:
        pickle.dump(output,f1)

if __name__ == "__main__":
    # data = load_data("/hci/junchen_data/Virus_Helper/SeConD_data/train_tfidf_v4.pickle")
    # del data[27999]
    # with open("/hci/junchen_data/Virus_Helper/SeConD_data/train_tfidf_v5.pickle",'wb') as f1:
    #     pickle.dump(data,f1)
    # for m in missing:
    #     print(m)
    #     sp = sentence_processor(data[m][1])
    #     temp = sp.adjacency_matrix_build()
    # replace_long_path("/hci/junchen_data/Virus_Helper/SeConD_data/train_tfidf_seq2seq_v3.pickle","/hci/junchen_data/Virus_Helper/SeConD_data/train_tfidf_v3.pickle")
    merge_dependency_tree("SeConD_data/test_tfidf.pickle","SeConD_data/dependency_tree/test_dependency_tree_part_9679.pickle","SeConD_data/test_tfidf_dt.pickle")
    # test = re.f
    # indall(r'Ġ*[^\|\?\~\`\!\@\#\$\%\^\&\*\(\)\_\-\+\=\{\}\[\]\'\"\:\;\,\<\.\>\\\/0-9a-zA-Z]{1,}$',"Ġhello")
    # print(test)
    # build_full_metrix("dev")
    # build_async_main("test",'SeConD_data')
    # test = np.array([[0,0,1],[2,0,0],[0,30,0]])
    # test = sparse.csr_matrix(test)
    # test = convert_matrix_to_sparse(test)
    # persona_data = persona_processor("Persona_data/personachat_self_original.json")
    # persona_data.build_pickle('train')
    # DSTC = DSTC_processor('DSTC7_AVSD/DSTC7-AVSD_test.json')
    # DSTC.build_pickle('test')
    # data = load_data('/hci/junchen_data/Virus_Helper/SeConD_data/train_tfidf_seq2seq_v3.pickle')
    # print(len(data[0]))

    # sp = sentence_processor("The quick brown fox jumps over the lazy dog.")
    # sp.adjacency_matrix_build()

    
    # print(data['train'][2]['utterances'][-1]['history'])
    
    # result = list(result)
    # result.pretty_print()
    # for head, rel, dep in result.triples():
    #     print(head,rel,dep)
    # TOKEN   TYPE    REF     RELATION
    # The     DT      4       det
    # quick   JJ      4       amod
   
    # print()