# -*- coding: utf-8 -*-
# @Time : 2022/5/14 11:33
# @Author : Jclian91
# @File : load_data.py
# @Place : Minghang, Shanghai
import pandas as pd

from utils.params import TRAIN_FILE, DEV_FILE, TEST_FILE


def read_model_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [_.strip() for _ in f.readlines()]
    for i, line in enumerate(lines):
        if i:
            items = line.split('\t')
            label = [0, 1] if int(items[0]) else [1, 0]
            data.append([label, items[3], items[4]])
    return data


if __name__ == '__main__':
    train_data = read_model_data(TEST_FILE)
    print(len(train_data))
    for _ in train_data:
        print(_)
