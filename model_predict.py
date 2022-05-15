# -*- coding: utf-8 -*-
# @Time : 2022/5/15 11:53
# @Author : Jclian91
# @File : model_predict.py
# @Place : Minghang, Shanghai
import numpy as np

from utils.params import *
from keras_roberta.tokenizer import RobertaTokenizer
from utils.roberta_tokenizer import tokenizer_encode
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
    print(f"label: {y}, prob: {predicted[0][y]}")
    return y


if __name__ == '__main__':
    review_text = 'I almost balled my eyes out 5 times. Almost. Beautiful movie, very inspiring.'
    predict_single_text(review_text)
    review_text = "I felt a bit let down by this movie, probably because it was over-hyped. " \
                  "Visually it was very true to the graphic novel flavor, and the story wasn't bad."
    predict_single_text(review_text)
