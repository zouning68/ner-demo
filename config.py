import os

TASK = 1    # 0: 原始的实体标注， 1: 纠错标注

epoch = 10
batch_size = 1024
max_len = 20           # 序列的最大长度
DATA_TYPE = "data"      # 'data', 'data2', 'msra', 'renmin'     # 数据格式
SAVE_PATH = "modeloutput/"                                     # 模型保存路径
if not os.path.exists(SAVE_PATH): os.makedirs(SAVE_PATH)

# 需要训练的模型
modes = ['IDCNNCRF2'] #   ['IDCNNCRF', 'IDCNNCRF2', 'BILSTMAttentionCRF', 'BILSTMCRF', 'BERTBILSTMCRF']