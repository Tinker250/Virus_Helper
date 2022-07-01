from pyparsing import NoMatch
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer
import re
import numpy as np
from sklearn import preprocessing

englis_stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

def TFIDF_Builder(filename,output_filename):
  print("begin TFIDF from file -> {}, output file -> {}".format("SeConD_data/"+filename+".pickle", output_filename))
  tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
  data = None
  with open("SeConD_data/"+filename+".pickle",'rb') as fin:
        data = pickle.load(fin)
  temp = []
  # data = data[80000:]
  for i,example in tqdm(enumerate(data), total=len(data)):
    # if(i % 10000 == 0):
    #   print(i/len(data))
    text_a = example[1]
    text_b = example[2]
    if(text_a == "" or text_b == ""):
      print(text_a)
      print(kkk)
      continue
    tokens_a = tokenizer.tokenize(text_a)
    tokens_b = tokenizer.tokenize(text_b)
    # all_tokens = [tokenizer.cls_token_id]+tokens_a+[tokenizer.sep_token]+tokens_b+[tokenizer.sep_token]
    text_a = " ".join(x for x in tokens_a)
    text_b = " ".join(x for x in tokens_b)
    #context-response pair
    # f_text = "<s> " +text_a+" </s> "+text_b+" </s>"

    #only encoder
    f_text = "<s> " +text_a+" </s>"
    f_text = re.sub('  ',' ',f_text)
    f_text = re.sub('   ',' ',f_text)
    temp.append(f_text)

  # print(temp)
  assert len(data) == len(temp)
  vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b|!|\?|\"|\'|\.|\@|\#|\$|\,|\%|\[|\]",min_df=0,lowercase=False)
  scores = vectorizer.fit_transform(temp)
  features = vectorizer.get_feature_names()
  scores = scores.toarray()
#   print(scores.toarray().shape)
  word2index = {}
  for i,feature in tqdm(enumerate(features)):
    # if(i%100 == 0):
    #   print(i/len(features))
    word2index[feature] = i
  # print(features)
  final_score = []
  for i,score in tqdm(enumerate(scores), total=len(scores)):
      # print(len(score))
      doc = temp[i].split(' ')
      score_list = []
      for word in doc:
        if(word.lower() in englis_stop_words) or len(word)==1 or word not in word2index.keys():
          score_list.append(0)
        else:
          idx = word2index[word]
          score_list.append(score[idx])
      # print(score_list)
      # print(doc)
      # print(kkk)
      final_score.append(score_list)
  output = []
  for i, item in tqdm(enumerate(data),total=len(data)):
    mat = np.array(final_score[i]).reshape(-1,1)
    scaler = preprocessing.MinMaxScaler()
    normalizedlist=scaler.fit_transform(mat)
    normalizedlist = normalizedlist.tolist()
    item_2 = []
    item_2.append(item[0])
    item_2.append(item[1])
    item_2.append(item[2])
    item_2.append([x[0] for x in normalizedlist]) #TODO:改这里就行
    output.append(item_2)
  print("len is {}".format(len(output)))
  with open('SeConD_data/'+output_filename+'.pickle','wb') as f1:
    pickle.dump(output,f1)

  print("End TFIDF")
  return word2index,scores

def merge():
  data_1 = None
  data_2 = None
  out_data = []
  with open("SeConD_data/train_tfidf_p1.pickle",'rb') as f1:
    data_1 = pickle.load(f1)
  with open("SeConD_data/train_tfidf_p2.pickle",'rb') as f2:
    data_2 = pickle.load(f2)
  for i,data in enumerate(data_2):
    data_1.append(data)
  print(len(data_1))
  with open("SeConD_data/train_tfidf.pickle",'wb') as f3:
    pickle.dump(data_1,f3)
    

def remove_stop_words(text):
  temp = text.split(' ')
  return_str = []
  for word in temp:
    if(word.lower() in englis_stop_words):
      pass
    else:
      return_str.append(word)
  return " ".join(return_str)


test_str = ['[CLS]', 'El', 'ise', '025', 'Ġand', 'ĠFab', 'ar', 'Ġwere', 'Ġable', 'Ġto', 'Ġhelp', 'Ġme', 'Ġwhen', 'ĠI', 'Ġposted', 'Ġa', 'Ġproblem', 'Ġon', 'Ġ12', '/', '31', '/', '09', 'Ġ(', 'I', 'Ġhad', 'Ġa', 'Ġroot', 'kit', 'Ġvirus', ').', 'ĠThank', 'Ġyou', 'Ġso', 'Ġmuch', 'Ġyou', 'Ġwere', 'Ġboth', 'Ġa', 'Ġbig', 'Ġhelp', '.', 'I', 'Ġam', 'Ġnot', 'Ġsure', 'Ġif', 'Ġmy', 'ĠC', 'URRENT', 'Ġproblem', 'Ġis', 'Ġrelated', '.', 'ĠBut', 'ĠI', 'Ġneed', 'ĠHELP', 'Ġagain', '.', 'When', 'ĠI', 'Ġtry', 'Ġto', 'Ġrun', 'ĠMal', 'ware', 'bytes', 'ĠAnt', 'iv', 'irus', 'ĠI', 'Ġget', 'Ġthe', 'ĠBL', 'UE', 'ĠSC', 'RE', 'EN', '.', 'ĠThe', 'Ġblue', 'Ġscreen', 'Ġmentions', 'Ġi', 'ast', 'or', '.', 'sys', 'Ġ(', 'which', 'Ġis', 'Ġpart', 'Ġof', 'Ġwhere', 'Ġthe', 'Ġproblem', 'Ġwas', 'Ġthe', 'Ġlast', 'Ġtime', ').', 'ĠMal', 'ware', 'bytes', 'Ġwill', 'Ġnot', 'Ġcomple', 't', 'ly', 'Ġrun', 'Ġthe', 'Ġblue', 'Ġscreen', 'Ġcomes', 'Ġon', 'Ġeach', 'Ġtime', '.', 'ĠI', 'Ġhave', 'Ġun', 'installed', 'Ġand', 'Ġrein', 'st', 'alled', 'Ġit', 'Ġand', 'ĠI', 'Ġget', 'Ġthe', 'Ġblue', 'Ġscreen', 'Ġeach', 'Ġtime', '.', 'I', 'Ġran', 'ĠGM', 'ER', 'Ġand', 'Ġhere', 'Ġis', 'Ġthe', 'Ġlog', ':', 'GM', 'ER', 'Ġ1', '.', '0', '.', '15', '.', '15', '281', 'Ġ-', 'Ġ[', 'link', ']', 'Root', 'kit', 'Ġscan', 'Ġ2010', '-', '01', '-', '13', 'Ġ', '</s>', 'ĠNot', 'Ġall', 'Ġhidden', 'Ġcomponents', 'Ġdetected', 'Ġby', 'ĠAR', 'K', 's', 'Ġare', 'Ġmalicious', '.', 'ĠIt', 'Ġis', 'Ġnormal', 'Ġfor', 'Ġa', 'ĠFire', 'wall', ',', 'Ġsome', 'ĠAnti', '-', 'v', 'irus', 'Ġand', 'ĠAnti', '-', 'mal', 'ware', 'Ġsoftware', 'Ġ(', 'Process', 'Guard', ',', 'ĠPrev', 'x', '1', ',', 'ĠAVG', 'ĠAS', '),', 'Ġsand', 'boxes', ',', 'Ġvirtual', 'Ġmachines', 'Ġand', 'ĠHost', 'Ġbased', 'ĠInt', 'r', 'usion', 'ĠPrevention', 'ĠSystems', 'Ġ(', 'HI', 'PS', ')', 'Ġto', 'Ġhook', 'Ġinto', 'Ġthe', 'ĠOS', 'Ġk', 'ernal', '/', 'SS', 'DT', 'Ġin', 'Ġorder', 'Ġto', 'Ġprotect', 'Ġyour', 'Ġsystem', '.', 'ĠSSD', 'T', 'Ġ(', 'System', 'ĠService', 'ĠDes', 'cript', 'or', 'ĠTable', ')', 'Ġis', 'Ġa', 'Ġtable', 'Ġthat', 'Ġstores', 'Ġaddresses', 'Ġof', 'Ġfunctions', 'Ġthat', 'Ġare', 'Ġused', 'Ġby', 'ĠWindows', '.', 'ĠWhenever', 'Ġa', 'Ġfunction', 'Ġis', 'Ġcalled', ',', 'ĠWindows', 'Ġlooks', 'Ġin', 'Ġthis', 'Ġtable', 'Ġto', 'Ġfind', 'Ġthe', 'Ġaddress', 'Ġfor', 'Ġit', '.', 'ĠBoth', 'ĠLeg', 'itimate', 'Ġprograms', 'Ġand', 'Ġroot', 'k', 'its', 'Ġcan', 'Ġhook', 'Ġinto', 'Ġand', 'Ġalter', 'Ġthis', 'Ġtable', '.', 'ĠYou', 'Ġshould', 'Ġnot', 'Ġbe', 'Ġalarmed', 'Ġif', 'Ġyou', 'Ġsee', 'Ġany', 'Ġhidden', 'Ġentries', 'Ġcreated', 'Ġby', 'Ġlegitimate', 'Ġprograms', 'Ġafter', 'Ġperforming', 'Ġa', 'Ġscan', '.', 'Some', 'Ġfiles', 'Ġare', 'Ġlocked', 'Ġby', 'Ġthe', 'Ġoperating', 'Ġsystem', 'Ġor', 'Ġrunning', 'Ġprograms', 'Ġduring', 'Ġuse', 'Ġfor', 'Ġprotection', ',', 'Ġso', 'Ġscanners', 'Ġcannot', 'Ġaccess', 'Ġthem', '.', 'ĠWhen', 'Ġthe', 'Ġscanner', 'Ġfinds', 'Ġsuch', 'Ġa', 'Ġfile', ',', 'Ġit', 'Ġmakes', 'Ġa', 'Ġnote', 'Ġand', 'Ġthen', 'Ġjust', 'Ġsk', 'ips', 'Ġto', 'Ġthe', 'Ġnext', 'Ġone', '.', 'ĠAPI', 'ĠKernel', 'Ġhooks', 'Ġare', 'Ġnot', 'Ġalways', 'Ġbad', 'Ġsince', 'Ġsome', 'Ġsystem', 'Ġmonitoring', 'Ġsoftware', 'Ġand', 'Ġsecurity', 'Ġtools', 'Ġuse', 'Ġthem', 'Ġas', 'Ġwell', '.', 'ĠIf', 'Ġno', 'Ġhooks', 'Ġare', 'Ġactive', 'Ġon', 'Ġa', 'Ġsystem', 'Ġit', 'Ġmeans', 'Ġthat', 'Ġall', 'Ġsystem', 'Ġservices', 'Ġare', 'Ġhandled', 'Ġby', 'Ġn', 't', 'os', 'kr', 'nl', '.', 'exe', 'Ġwhich', 'Ġis', 'Ġa', 'Ġbase', 'Ġcomponent', 'Ġof', 'ĠWindows', 'Ġoperating', 'Ġsystems', 'Ġand', 'Ġthe', 'Ġprocess', 'Ġused', 'Ġin', 'Ġthe', 'Ġboot', '-', 'up', 'Ġcycle', 'Ġof', 'Ġa', 'Ġcomputer', '.', 'm', 'oh', 'f', 'ilt', '.', 'sys', 'Ġis', 'Ġa', 'Ġdriver', 'Ġfor', 'ĠIntel', 'ĠCorporation', 'i', 'ast', 'or', '.', 'sys', 'Ġis', 'Ġa', 'Ġdriver', 'Ġfor', 'ĠIntel', "'s", 'ĠMatrix', 'ĠStorage', '[', 'link', ']', 'Are', 'Ġyou', 'Ġgetting', 'Ġthe', 'ĠBS', 'OD', 'Ġonly', 'Ġwhen', 'Ġrunning', 'ĠMB', 'AM', '?', '</s>', 'ĠI', 'Ġonly', 'Ġget', 'Ġthe', 'Ġblue', 'Ġscreen', 'Ġwhen', 'ĠI', 'Ġrun', 'ĠMal', 'ware', 'bytes', 'ĠAnt', 'iv', 'irus', '.', 'ĠI', 'Ġran', 'Ġthe', 'Ġsuper', 'ant', 'isp', 'y', 'ware', 'Ġfine', 'Ġ(', 'it', 'Ġonly', 'Ġshowed', 'Ġa', 'Ġfew', 'Ġtracking', 'Ġcookies', ').', 'The', 'Ġerror', 'Ġon', 'Ġthe', 'Ġblue', 'Ġscreen', 'Ġmentions', 'ĠI', 'ast', 'or', '.', 'sys', '.', 'Mal', 'ware', 'bytes', 'ĠAnt', 'iv', 'irus', 'Ġdoes', 'Ġnot', 'Ġcomplete', 'Ġrunning', '.', 'ĠThe', 'Ġblues', 'creen', 'Ġcomes', 'Ġup', 'Ġand', 'ĠI', 'Ġhave', 'Ġto', 'Ġturn', 'Ġthe', 'Ġcomputer', 'Ġoff', 'Ġand', 'Ġon', '.', 'I', 'Ġhave', 'Ġalways', 'Ġbeen', 'Ġable', 'Ġto', 'Ġrun', 'Ġit', '.', 'ĠNot', 'Ġsure', 'Ġwhy', 'ĠI', 'Ġcant', 'Ġnow', '.', 'I', 'Ġdid', 'Ġuninstall', 'Ġit', 'Ġand', 'Ġrein', 'st', 'alled', 'Ġit', 'Ġand', 'Ġstill', 'Ġthe', 'Ġsame', 'Ġproblem', '.', 'I', 'Ġam', 'Ġnot', 'Ġsure', 'Ġabout', 'Ġ"', 'run', 'Ġreg', 'edit', 'Ġand', 'Ġdelete', 'Ġall', 'Ġent', 'ires', 'Ġfor', 'ĠMB', '"', 'Ġthat', 'Ġwhite', 'ac', '2', 'k', '4', 'Ġmentions', '.', 'ĠDo', 'ĠI', 'Ġneed', 'Ġto', 'Ġdo', 'Ġthis', '..', 'Ġand', 'Ġif', 'Ġso', 'Ġhow', 'Ġwould', 'ĠI', 'Ġdo', 'Ġit', '.', 'Qu', 'iet', 'man', '7', ',', 'ĠWhat', 'Ġdo', 'ĠI', 'Ġneed', 'Ġto', 'Ġdo', '.', 'ĠPlease', 'Ġadvice', '.', 'Thanks', ',', '</s>', 'ĠSince', 'Ġthe', 'Ġproblem', 'Ġonly', 'Ġoccurs', 'Ġwhen', 'Ġusing', 'ĠMB', 'AM', ',', 'ĠI', 'Ġrecommend', 'Ġyou', 'Ġreport', 'Ġthis', 'Ġissue', 'Ġin', 'Ġthe', 'Ġ[', 'link', ']', 'Ġor', 'ĠE', '-', 'mail', 'Ġthe', 'Ġ[', 'link', ']', 'Ġso', 'Ġthe', 'Ġdevelopment', 'Ġteam', 'Ġcan', 'Ġinvestigate', 'Ġthe', 'Ġcause', '.', '[SEP]', 'Due', 'Ġto', 'Ġlack', 'Ġof', 'Ġfeedback', ',', 'Ġthis', 'Ġtopic', 'Ġis', 'Ġnow', 'Ġclosed', '.', 'If', 'Ġyou', 'Ġare', 'Ġthe', 'Ġoriginal', 'Ġtopic', 'Ġstarter', 'Ġand', 'Ġyou', 'Ġneed', 'Ġthis', 'Ġtopic', 'Ġreopened', ',', 'Ġplease', 'Ġsend', 'Ġme', 'Ġa', 'ĠPM', '.', 'Everyone', 'Ġelse', ',', 'Ġplease', 'Ġstart', 'Ġa', 'Ġnew', 'Ġtopic', '.', '[SEP]']
test_list = [0, 'ĠElise', '025', 'Ġand', 'ĠFab', 'ar', 'Ġwere', 'Ġable', 'Ġto', 'Ġhelp', 'Ġme', 'Ġwhen', 'ĠI', 'Ġposted', 'Ġa', 'Ġproblem', 'Ġon', 'Ġ12', '/', '31', '/', '09', 'Ġ(', 'I', 'Ġhad', 'Ġa', 'Ġroot', 'kit', 'Ġvirus', ').', 'ĠThank', 'Ġyou', 'Ġso', 'Ġmuch', 'Ġyou', 'Ġwere', 'Ġboth', 'Ġa', 'Ġbig', 'Ġhelp', '.', 'I', 'Ġam', 'Ġnot', 'Ġsure', 'Ġif', 'Ġmy', 'ĠC', 'URRENT', 'Ġproblem', 'Ġis', 'Ġrelated', '.', 'ĠBut', 'ĠI', 'Ġneed', 'ĠHELP', 'Ġagain', '.', 'When', 'ĠI', 'Ġtry', 'Ġto', 'Ġrun', 'ĠMal', 'ware', 'bytes', 'ĠAnt', 'iv', 'irus', 'ĠI', 'Ġget', 'Ġthe', 'ĠBL', 'UE', 'ĠSC', 'RE', 'EN', '.', 'ĠThe', 'Ġblue', 'Ġscreen', 'Ġmentions', 'Ġi', 'ast', 'or', '.', 'sys', 'Ġ(', 'which', 'Ġis', 'Ġpart', 'Ġof', 'Ġwhere', 'Ġthe', 'Ġproblem', 'Ġwas', 'Ġthe', 'Ġlast', 'Ġtime', ').', 'ĠMal', 'ware', 'bytes', 'Ġwill', 'Ġnot', 'Ġcomple', 't', 'ly', 'Ġrun', 'Ġthe', 'Ġblue', 'Ġscreen', 'Ġcomes', 'Ġon', 'Ġeach', 'Ġtime', '.', 'ĠI', 'Ġhave', 'Ġun', 'installed', 'Ġand', 'Ġrein', 'st', 'alled', 'Ġit', 'Ġand', 'ĠI', 'Ġget', 'Ġthe', 'Ġblue', 'Ġscreen', 'Ġeach', 'Ġtime', '.', 'I', 'Ġran', 'ĠGM', 'ER', 'Ġand', 'Ġhere', 'Ġis', 'Ġthe', 'Ġlog', ':', 'GM', 'ER', 'Ġ1', '.', '0', '.', '15', '.', '15', '281', 'Ġ-', 'Ġ[', 'link', ']', 'Root', 'kit', 'Ġscan', 'Ġ2010', '-', '01', '-', '13', 'Ġ', '</s>', 'ĠNot', 'Ġall', 'Ġhidden', 'Ġcomponents', 'Ġdetected', 'Ġby', 'ĠAR', 'K', 's', 'Ġare', 'Ġmalicious', '.', 'ĠIt', 'Ġis', 'Ġnormal', 'Ġfor', 'Ġa', 'ĠFire', 'wall', ',', 'Ġsome', 'ĠAnti', '-', 'v', 'irus', 'Ġand', 'ĠAnti', '-', 'mal', 'ware', 'Ġsoftware', 'Ġ(', 'Process', 'Guard', ',', 'ĠPrev', 'x', '1', ',', 'ĠAVG', 'ĠAS', '),', 'Ġsand', 'boxes', ',', 'Ġvirtual', 'Ġmachines', 'Ġand', 'ĠHost', 'Ġbased', 'ĠInt', 'r', 'usion', 'ĠPrevention', 'ĠSystems', 'Ġ(', 'HI', 'PS', ')', 'Ġto', 'Ġhook', 'Ġinto', 'Ġthe', 'ĠOS', 'Ġk', 'ernal', '/', 'SS', 'DT', 'Ġin', 'Ġorder', 'Ġto', 'Ġprotect', 'Ġyour', 'Ġsystem', '.', 'ĠSSD', 'T', 'Ġ(', 'System', 'ĠService', 'ĠDes', 'cript', 'or', 'ĠTable', ')', 'Ġis', 'Ġa', 'Ġtable', 'Ġthat', 'Ġstores', 'Ġaddresses', 'Ġof', 'Ġfunctions', 'Ġthat', 'Ġare', 'Ġused', 'Ġby', 'ĠWindows', '.', 'ĠWhenever', 'Ġa', 'Ġfunction', 'Ġis', 'Ġcalled', ',', 'ĠWindows', 'Ġlooks', 'Ġin', 'Ġthis', 'Ġtable', 'Ġto', 'Ġfind', 'Ġthe', 'Ġaddress', 'Ġfor', 'Ġit', '.', 'ĠBoth', 'ĠLeg', 'itimate', 'Ġprograms', 'Ġand', 'Ġroot', 'k', 'its', 'Ġcan', 'Ġhook', 'Ġinto', 'Ġand', 'Ġalter', 'Ġthis', 'Ġtable', '.', 'ĠYou', 'Ġshould', 'Ġnot', 'Ġbe', 'Ġalarmed', 'Ġif', 'Ġyou', 'Ġsee', 'Ġany', 'Ġhidden', 'Ġentries', 'Ġcreated', 'Ġby', 'Ġlegitimate', 'Ġprograms', 'Ġafter', 'Ġperforming', 'Ġa', 'Ġscan', '.', 'Some', 'Ġfiles', 'Ġare', 'Ġlocked', 'Ġby', 'Ġthe', 'Ġoperating', 'Ġsystem', 'Ġor', 'Ġrunning', 'Ġprograms', 'Ġduring', 'Ġuse', 'Ġfor', 'Ġprotection', ',', 'Ġso', 'Ġscanners', 'Ġcannot', 'Ġaccess', 'Ġthem', '.', 'ĠWhen', 'Ġthe', 'Ġscanner', 'Ġfinds', 'Ġsuch', 'Ġa', 'Ġfile', ',', 'Ġit', 'Ġmakes', 'Ġa', 'Ġnote', 'Ġand', 'Ġthen', 'Ġjust', 'Ġsk', 'ips', 'Ġto', 'Ġthe', 'Ġnext', 'Ġone', '.', 'ĠAPI', 'ĠKernel', 'Ġhooks', 'Ġare', 'Ġnot', 'Ġalways', 'Ġbad', 'Ġsince', 'Ġsome', 'Ġsystem', 'Ġmonitoring', 'Ġsoftware', 'Ġand', 'Ġsecurity', 'Ġtools', 'Ġuse', 'Ġthem', 'Ġas', 'Ġwell', '.', 'ĠIf', 'Ġno', 'Ġhooks', 'Ġare', 'Ġactive', 'Ġon', 'Ġa', 'Ġsystem', 'Ġit', 'Ġmeans', 'Ġthat', 'Ġall', 'Ġsystem', 'Ġservices', 'Ġare', 'Ġhandled', 'Ġby', 'Ġn', 't', 'os', 'kr', 'nl', '.', 'exe', 'Ġwhich', 'Ġis', 'Ġa', 'Ġbase', 'Ġcomponent', 'Ġof', 'ĠWindows', 'Ġoperating', 'Ġsystems', 'Ġand', 'Ġthe', 'Ġprocess', 'Ġused', 'Ġin', 'Ġthe', 'Ġboot', '-', 'up', 'Ġcycle', 'Ġof', 'Ġa', 'Ġcomputer', '.', 'm', 'oh', 'f', 'ilt', '.', 'sys', 'Ġis', 'Ġa', 'Ġdriver', 'Ġfor', 'ĠIntel', 'ĠCorporation', 'i', 'ast', 'or', '.', 'sys', 'Ġis', 'Ġa', 'Ġdriver', 'Ġfor', 'ĠIntel', "'s", 'ĠMatrix', 'ĠStorage', '[', 'link', ']', 'Are', 'Ġyou', 'Ġgetting', 'Ġthe', 'ĠBS', 'OD', 'Ġonly', 'Ġwhen', 'Ġrunning', 'ĠMB', 'AM', '?', '</s>', 'ĠI', 'Ġonly', 'Ġget', 'Ġthe', 'Ġblue', 'Ġscreen', 'Ġwhen', 'ĠI', 'Ġrun', 'ĠMal', 'ware', 'bytes', 'ĠAnt', 'iv', 'irus', '.', 'ĠI', 'Ġran', 'Ġthe', 'Ġsuper', 'ant', 'isp', 'y', 'ware', 'Ġfine', 'Ġ(', 'it', 'Ġonly', 'Ġshowed', 'Ġa', 'Ġfew', 'Ġtracking', 'Ġcookies', ').', 'The', 'Ġerror', 'Ġon', 'Ġthe', 'Ġblue', 'Ġscreen', 'Ġmentions', 'ĠI', 'ast', 'or', '.', 'sys', '.', 'Mal', 'ware', 'bytes', 'ĠAnt', 'iv', 'irus', 'Ġdoes', 'Ġnot', 'Ġcomplete', 'Ġrunning', '.', 'ĠThe', 'Ġblues', 'creen', 'Ġcomes', 'Ġup', 'Ġand', 'ĠI', 'Ġhave', 'Ġto', 'Ġturn', 'Ġthe', 'Ġcomputer', 'Ġoff', 'Ġand', 'Ġon', '.', 'I', 'Ġhave', 'Ġalways', 'Ġbeen', 'Ġable', 'Ġto', 'Ġrun', 'Ġit', '.', 'ĠNot', 'Ġsure', 'Ġwhy', 'ĠI', 'Ġcant', 'Ġnow', '.', 'I', 'Ġdid', 'Ġuninstall', 'Ġit', 'Ġand', 'Ġrein', 'st', 'alled', 'Ġit', 'Ġand', 'Ġstill', 'Ġthe', 'Ġsame', 'Ġproblem', '.', 'I', 'Ġam', 'Ġnot', 'Ġsure', 'Ġabout', 'Ġ"', 'run', 'Ġreg', 'edit', 'Ġand', 'Ġdelete', 'Ġall', 'Ġent', 'ires', 'Ġfor', 'ĠMB', '"', 'Ġthat', 'Ġwhite', 'ac', '2', 'k', '4', 'Ġmentions', '.', 'ĠDo', 'ĠI', 'Ġneed', 'Ġto', 'Ġdo', 'Ġthis', '..', 'Ġand', 'Ġif', 'Ġso', 'Ġhow', 'Ġwould', 'ĠI', 'Ġdo', 'Ġit', '.', 'Qu', 'iet', 'man', '7', ',', 'ĠWhat', 'Ġdo', 'ĠI', 'Ġneed', 'Ġto', 'Ġdo', '.', 'ĠPlease', 'Ġadvice', '.', 'Thanks', ',', '</s>', 'ĠSince', 'Ġthe', 'Ġproblem', 'Ġonly', 'Ġoccurs', 'Ġwhen', 'Ġusing', 'ĠMB', 'AM', ',', 'ĠI', 'Ġrecommend', 'Ġyou', 'Ġreport', 'Ġthis', 'Ġissue', 'Ġin', 'Ġthe', 'Ġ[', 'link', ']', 'Ġor', 'ĠE', '-', 'mail', 'Ġthe', 'Ġ[', 'link', ']', 'Ġso', 'Ġthe', 'Ġdevelopment', 'Ġteam', 'Ġcan', 'Ġinvestigate', 'Ġthe', 'Ġcause', '.', '</s>', 'ĠDue', 'Ġto', 'Ġlack', 'Ġof', 'Ġfeedback', ',', 'Ġthis', 'Ġtopic', 'Ġis', 'Ġnow', 'Ġclosed', '.', 'If', 'Ġyou', 'Ġare', 'Ġthe', 'Ġoriginal', 'Ġtopic', 'Ġstarter', 'Ġand', 'Ġyou', 'Ġneed', 'Ġthis', 'Ġtopic', 'Ġreopened', ',', 'Ġplease', 'Ġsend', 'Ġme', 'Ġa', 'ĠPM', '.', 'Everyone', 'Ġelse', ',', 'Ġplease', 'Ġstart', 'Ġa', 'Ġnew', 'Ġtopic', '.', '</s>']


def test():
  print(len(test_list))
  print(len(test_str))
  # test_split = test_str.split(' ')
  for i, token in enumerate(test_str):
    if(token != test_list[i]):
      print(token, test_list[i])

# merge()
# TFIDF_Builder('test_tfidf_seq2seq_v2','test_tfidf_seq2seq_v2')
# merge()
# test()