from Public.utils import *
from train import train_sample
import pandas as pd
from config import modes, SAVE_PATH, max_len, epoch
from test import test_sample
from Model.BERT_BILSTM_CRF import BERTBILSTMCRF
from Model.BILSTM_Attetion_CRF import BILSTMAttentionCRF
from Model.BILSTM_CRF import BILSTMCRF
from Model.IDCNN_CRF import IDCNNCRF
from Model.IDCNN5_CRF import IDCNNCRF2
from DataProcess.vocab import *
import numpy as np
from tqdm import tqdm
import re, codecs, jieba, math, random

def train_model():
    # 定义文件路径（以便记录数据）
    log_path = os.path.join(path_log_dir, 'train_log.log')
    df_path = os.path.join(path_log_dir, 'df.csv')
    log = create_log(log_path)
    # 训练同时记录数据写入的df文件中
    columns = ['model_name', 'epoch', 'loss', 'acc', 'val_loss', 'val_acc', 'f1', 'recall']
    df = pd.DataFrame(columns=columns)
    for model in modes:
        info_list = train_sample(train_model=model, epochs=epoch, log=log)
        for info in info_list:
            df = df.append([info])
        df.to_csv(df_path)

def test_model():
    # 定义文件路径（以便记录数据）
    log_path = os.path.join(path_log_dir, 'test_log.log')
    df_path = os.path.join(path_log_dir, 'test.csv')
    log = create_log(log_path)
    # 测试同时记录数据写入的df文件中
    columns = ['model_name', 'epoch', 'loss', 'acc', 'val_loss', 'val_acc', 'f1', 'recall']
    df = pd.DataFrame(columns=columns)
    for model in modes:
        info_list = test_sample(test_model=model, log=log)
        for info in info_list:
            df = df.append([info])
        df.to_csv(df_path)

class nerdetect():
    def __init__(self, model_type):         # 'IDCNNCRF', 'IDCNNCRF2', 'BILSTMAttentionCRF', 'BILSTMCRF', 'BERTBILSTMCRF'
        self.w2i = get_w2i()  # word to index
        self.i2w = {v: k for k, v in self.w2i.items()}
        self.tag2index = get_tag2index()  # tag to index
        self.index2tag = {v: k for k, v in self.tag2index.items()}
        self.vocab_size = len(self.w2i)
        self.tag_size = len(self.tag2index)
        self.unk_flag = unk_flag
        self.unk_index = self.w2i.get(unk_flag, 101)
        self.pad_index = self.w2i.get(pad_flag, 1)
        self.cls_index = self.w2i.get(cls_flag, 102)
        self.sep_index = self.w2i.get(sep_flag, 103)
        if model_type == 'BERTBILSTMCRF':
            model_class = BERTBILSTMCRF(self.vocab_size, self.tag_size, max_len=max_len)
        elif model_type == 'BILSTMAttentionCRF':
            model_class = BILSTMAttentionCRF(self.vocab_size, self.tag_size)
        elif model_type == 'BILSTMCRF':
            model_class = BILSTMCRF(self.vocab_size, self.tag_size)
        elif model_type == 'IDCNNCRF':
            model_class = IDCNNCRF(self.vocab_size, self.tag_size, max_len=max_len)
        elif model_type == 'IDCNNCRF2':
            model_class = IDCNNCRF2(self.vocab_size, self.tag_size, max_len=max_len)
        else:
            raise NotImplemented
        self.model = model_class.creat_model()
        self.model.load_weights(SAVE_PATH + model_type + ".HDF5")

    def sentence2id(self, sent):
        seg_query = []
        for e in list(jieba.cut(sent)):
            if u'\u4e00' <= e <= u'\u9fa5': seg_query.extend(list(e))
            else: seg_query.append(e)
        ids = [self.w2i.get(w, self.w2i[self.unk_flag]) for w in seg_query]
        if len(ids) < max_len:
            pad_num = max_len - len(ids)
            data_ids = [self.pad_index] * pad_num + ids
        else:
            data_ids = ids[:max_len]
        return data_ids, len(seg_query)

    def detect(self, sentence):
        res, error = [], []
        data_ids, sent_len = self.sentence2id(sentence)
        sent_len = [sent_len]
        sentids = np.array([data_ids])
        pre = self.model.predict(sentids)
        probs = np.argmax(pre, axis=2)
        for sent in range(len(sentids)):
            pred = list(probs[sent][-sent_len[sent]:])
            sentce = list(sentids[sent][-sent_len[sent]:])
            for word in range(sent_len[sent]):
                res.append((self.i2w.get(sentce[word]), self.index2tag.get(pred[word])))
        entity_name, entity_start = "", 0
        for i in range(len(res)):
            word, tag = res[i]
            if word == self.unk_flag: continue
            if tag[0] == "B" or tag[0] == "I":
                entity_name += word
            if (tag[0] == "O" or i == len(res)-1) and entity_name:
                entity_start = sentence.find(entity_name)
                error.append([entity_name, entity_start, entity_start + len(entity_name), tag.split("-")[-1]])
                entity_name = ""
        return res, error

def sample_data():
    np.random.seed(8)
    def load_same_pinyin(path, sep='\t'):  # 加载同音字
        result = dict()
        if not os.path.exists(path): return result
        with codecs.open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'): continue
                parts = line.split(sep)
                if parts:
                    key_char, same_pron_same_tone, same_pron_diff_tone = parts[0], set(), set()
                    if len(parts) > 1: same_pron_same_tone = set(list(parts[1]))
                    if len(parts) > 2: same_pron_diff_tone = set(list(parts[2]))
                    value = same_pron_same_tone.union(same_pron_diff_tone)
                    if len(key_char) > 1 or not value: continue
                    result[key_char] = value
        return result
    def load_same_stroke(path, sep='\t'):  # 加载形似字
        result = dict()
        if not os.path.exists(path): return result
        with codecs.open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if "登" in line:
                    a=1
                line = line.strip().replace(" ", "\t")
                if line.startswith('#'): continue
                parts = line.split(sep)
                if parts and len(parts) > 1:
                    for i, c in enumerate(parts):
                        result[c] = set(list(parts[:i] + parts[i + 1:]))
        return result
    common_char = {line.split()[0]: int(line.split()[1]) for line in open("data/data/common_char_set.txt", encoding="utf8").readlines()}
    def sample_number(arr, max_len=5):
        word_freq = {e: common_char.get(e, 0) for e in arr}         #sample_index = list(set(np.random.randint(low=len(arr), size=max_len)))
        sorted_freq = sorted(word_freq.items(), key=lambda d: d[1], reverse=True)
        return [e[0] for e in sorted_freq if e[1] >= 5]
    def tag_word(arr, tag):
        res = []
        for num in range(1, len(arr) + 1):   # num-gram
            for i in range(len(arr) - num + 1):
                curr, tmp = arr[i: i+num], []
                before, after = [e[0] + " O\n" for e in arr[: i]], [e[0] + " O\n" for e in arr[i+num:]]
                for j in range(len(curr)):      # tag
                    taged_word = re.sub(u"[ ]{1,}", "", curr[j][1][random.randint(0, len(curr[j][1])-1)])
                    if j == 0: tmp.append((taged_word + " B-" + tag + "\n"))
                    else: tmp.append((taged_word + " I-" + tag + "\n"))
                res.append(before + tmp + after)
        return res
    def sample(query, dicts, tag):
        candi_words = []    #; query = "java 工程师 开发C++"
        seg_query = list(jieba.cut(query))
        for cur_index in range(len(seg_query)):
            word = seg_query[cur_index]
            if not (u'\u4e00' <= word <= u'\u9fa5'): continue
            min_num, beforestr, afterstr = 1e9, [], []
            for index in range(len(seg_query)):     # 记录前后的状态
                if seg_query[index] in ['', ' '] or index == cur_index: continue
                if u'\u4e00' <= seg_query[index] <= u'\u9fa5':
                    if index < cur_index: beforestr.extend([e + " O\n" for e in seg_query[index]])
                    else: afterstr.extend([e + " O\n" for e in seg_query[index]])
                else:
                    if index < cur_index: beforestr.append(seg_query[index] + " 0\n")
                    else: afterstr.append(seg_query[index] + " 0\n")
            _dict = [(e, sample_number(list(dicts.get(e, set())))) for e in word]
            candi_words.append(''.join(beforestr + [e + " O\n" for e in word] + afterstr + ["\n"]))       # 正确的序列
            for e in _dict: min_num = min(min_num, len(e[1]))
            for _ in range(min_num * 8):    # 替换
                for tmp in tag_word(_dict, tag):
                    candi_words.append(''.join(beforestr + tmp + afterstr + ["\n"]))
        return candi_words
    same_pinyin = load_same_pinyin("data/data/same_pinyin.txt")
    same_stroke = load_same_stroke("data/data/same_stroke.txt")
    matchObj = re.compile(r'(.+)&([0-9]+)', re.M | re.I)
    path = "data/data/query_original"
    samples = []
    for i, line in enumerate(tqdm(open(path, encoding="utf8"))):
        matchRes = matchObj.match(line)
        if not matchRes: continue
        query, freq = matchRes.group(1), int(matchRes.group(2))
        query = query.lower()
        query = re.sub(u"[　  ]{1,}", "", query)
        pinyin_res = sample(query, same_pinyin, "SP")
        stoke_res = sample(query, same_stroke, "SS")
        samples.extend(list(set(pinyin_res + stoke_res)))
    random.shuffle(samples)
    #train, test = samples[: math.ceil(len(samples) * 0.9)], samples[math.ceil(len(samples) * 0.9):]
    train, test = samples[: len(samples) - 50000], samples[len(samples) - 50000:]
    with open("data/data/train.txt", "w", encoding="utf8") as fin:
        fin.write(''.join(train))
    with open("data/data/test.txt", "w", encoding="utf8") as fin:
        fin.write(''.join(test))
    a=1

def gen_vocab():
    vocab = []
    vocab_origion = open("data/chinese_L-12_H-768_A-12/vocab.txt", encoding="utf8").readlines()
    common_char = [line.strip()[0] + "\n" for line in open("data/data/common_char_set.txt", encoding="utf8").readlines()]
    english = [line.split()[0] + "\n" for line in open("data/data/english.txt", encoding="utf8").readlines()]
    for e in vocab_origion:
        if e not in vocab: vocab.append(e)
    for e in common_char:
        if e not in vocab: vocab.append(e)
    for e in english:
        if e not in vocab: vocab.append(e)
    vocab.append(" \n")
    with open("data/vocab/vocab.txt", "w", encoding="utf8") as fout:
        fout.write(''.join(vocab))
    a=1

if __name__ == "__main__":
    #gen_vocab()
    sample_data()
    #train_model()
    #test_model()
    #nd = nerdetect('IDCNNCRF2'); print(nd.detect("php 前端"))
    pass
