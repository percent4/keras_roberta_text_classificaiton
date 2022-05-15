# -*- coding: utf-8 -*-
# @Time : 2022/5/14 19:48
# @Author : Jclian91
# @File : model_train.py
# @Place : Minghang, Shanghai
import numpy as np
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam

from utils.params import *
from utils.load_data import read_model_data
from keras_roberta.roberta import build_bert_model
from keras_roberta.tokenizer import RobertaTokenizer
from utils.robeberta_tokernizer import tokenizer_encode

tokenizer = RobertaTokenizer(GPT_BPE_VOCAB, GPT_BPE_MERGE, ROBERTA_DICT)


# data generator for model
class DataGenerator:
    def __init__(self, data, batch_size=BATCH_SIZE):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                y, text = self.data[i]
                x1, x2 = tokenizer_encode(tokenizer=tokenizer, text=text, max_seq_length=MAX_SEQ_LENGTH)
                X1.append(x1)
                X2.append(x2)
                Y.append(y)
                if len(X1) == self.batch_size or i == idxs[-1]:
                    yield [np.array(X1), np.array(X2)], np.array(Y)
                    [X1, X2, Y] = [], [], []


# 构建模型
def create_cls_model():
    # Roberta model
    roberta_model = build_bert_model(CONFIG_FILE_PATH, CHECKPOINT_FILE_PATH, roberta=True)  # 建立模型，加载权重

    for layer in roberta_model.layers:
        layer.trainable = True

    cls_layer = Lambda(lambda x: x[:, 0])(roberta_model.output)    # 取出[CLS]对应的向量用来做分类
    p = Dense(2, activation='softmax')(cls_layer)     # 多分类

    model = Model(roberta_model.input, p)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(1e-5),   # 用足够小的学习率
        metrics=['accuracy']
    )

    return model


if __name__ == '__main__':

    # 数据处理, 读取训练集和测试集
    train_data = read_model_data(TRAIN_FILE)
    test_data = read_model_data(DEV_FILE)
    print(f"finish data processing! train data number: {len(train_data)}, test data number: {len(test_data)}")

    # 模型训练
    model = create_cls_model()
    # 启用对抗训练FGM
    train_D = DataGenerator(train_data)
    test_D = DataGenerator(test_data)

    print("begin model training...")
    model.fit_generator(
        train_D.__iter__(),
        steps_per_epoch=len(train_D),
        epochs=EPOCH,
        validation_data=test_D.__iter__(),
        validation_steps=len(test_D)
    )

    model.save_weights('sst2.h5')
