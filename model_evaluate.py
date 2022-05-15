# -*- coding: utf-8 -*-
# @Time : 2022/5/14 20:25
# @Author : Jclian91
# @File : model_evaluate.py
# @Place : Minghang, Shanghai
import json
import numpy as np
from keras.models import load_model
from keras_bert import get_custom_objects
from sklearn.metrics import classification_report


from utils.params import *
from utils.load_data import read_model_data
from keras_roberta.roberta import build_bert_model
from keras_roberta.tokenizer import RobertaTokenizer
from utils.robeberta_tokernizer import tokenizer_encode
from model_train import create_cls_model

tokenizer = RobertaTokenizer(GPT_BPE_VOCAB, GPT_BPE_MERGE, ROBERTA_DICT)

# 加载训练好的模型
model = create_cls_model()
model.load_weights('sst2.h5')


# 对单句话进行预测
def predict_single_text(text):
    # 利用BERT进行tokenize
    x1, x2 = tokenizer_encode(tokenizer, text, MAX_SEQ_LENGTH)
    # 模型预测并输出预测结果
    predicted = model.predict([[x1], [x2]])
    y = np.argmax(predicted[0])
    return y


# 模型评估
def evaluate():
    test_data = read_model_data(DEV_FILE)
    true_y_list, pred_y_list = [], []
    for i, data in enumerate(test_data):
        print("predict %d samples" % (i+1))
        true_y, text = data
        true_y = 1 if true_y == [0, 1] else 0
        pred_y = predict_single_text(text)
        true_y_list.append(true_y)
        pred_y_list.append(pred_y)

    return classification_report(true_y_list, pred_y_list, digits=4)


output_data = evaluate()
print("model evaluate result:\n")
print(output_data)
