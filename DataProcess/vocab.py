# 获取词典

from Public.path import path_vocab
from config import TASK

unk_flag = '[UNK]'
pad_flag = '[PAD]'
cls_flag = '[CLS]'
sep_flag = '[SEP]'


# 获取 word to index 词典
def get_w2i(vocab_path = path_vocab):
    w2i = {}
    with open(vocab_path, 'r', encoding='utf8') as f:
        while True:
            text = f.readline()
            if not text:
                break
            text = text.strip()
            if text and len(text) > 0:
                w2i[text] = len(w2i) + 1
    return w2i


# 获取 tag to index 词典
def get_tag2index():
    if TASK == 0: return {"O": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4, "B-ORG": 5, "I-ORG": 6}         # NER 标注的标签
    #else: return {"O": 0, "B-R": 1, "I-R": 2, "B-M": 3, "I-M": 4, "B-S": 5, "I-S": 6, "B-W": 7, "I-W": 8}         # query 纠错标注的标签1
    else: return {"O": 0, "B-SP": 1, "I-SP": 2, "B-SS": 3, "I-SS": 4}  # query 纠错标注的标签


if __name__ == '__main__':
    get_w2i()






















