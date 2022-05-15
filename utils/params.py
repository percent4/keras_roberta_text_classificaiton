# -*- coding: utf-8 -*-
# @Time : 2022/5/14 11:32
# @Author : Jclian91
# @File : params.py
# @Place : Minghang, Shanghai
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__)).split('utils')[0]
TRAIN_FILE = os.path.join(PROJECT_DIR, 'data/train.tsv')
DEV_FILE = os.path.join(PROJECT_DIR, 'data/dev.tsv')
TEST_FILE = os.path.join(PROJECT_DIR, 'data/test.tsv')

# 模型配置
roberta_path = os.path.join(PROJECT_DIR, 'roberta-base')
tf_roberta_path = os.path.join(PROJECT_DIR, 'tf_roberta_base')
tf_ckpt_name = 'tf_roberta_base.ckpt'
vocab_path = os.path.join(PROJECT_DIR, 'keras_roberta')

CONFIG_FILE_PATH = os.path.join(tf_roberta_path, 'bert_config.json')
CHECKPOINT_FILE_PATH = os.path.join(tf_roberta_path, tf_ckpt_name)
GPT_BPE_VOCAB = os.path.join(vocab_path, 'encoder.json')
GPT_BPE_MERGE = os.path.join(vocab_path, 'vocab.bpe')
ROBERTA_DICT = os.path.join(roberta_path, 'dict.txt')

# 模型参数配置
EPOCH = 10              # 训练轮次
BATCH_SIZE = 64         # 批次数量
MAX_SEQ_LENGTH = 80     # 最大长度
