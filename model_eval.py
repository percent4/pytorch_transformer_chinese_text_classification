# -*- coding: utf-8 -*-
# @Time : 2023/3/17 15:31
# @Author : Jclian91
# @File : model_eval.py
# @Place : Minghang, Shanghai
import torch as T
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import torch.nn.functional as F

from model import TextClassifier
from text_featuring import load_file_file, text_feature

model = T.load('sougou_mini_cls.pth')

label_dict, char_dict = load_file_file()
label_dict_rev = {v: k for k, v in label_dict.items()}
print(label_dict)
def eval(text):
    labels, contents = ['汽车'], [text]
    samples, y_true = text_feature(labels, contents, label_dict, char_dict)
    # print(samples)
    # print(len(samples[0]))
    x = T.from_numpy(np.array(samples)).long()
    y_pred = model(x)
    # print(y_pred)
    y_numpy = F.softmax(y_pred, dim=1).detach().numpy()
    # print(y_numpy)
    predict_list = np.argmax(y_numpy, axis=1).tolist()
    return label_dict_rev[predict_list[0]]


if __name__ == '__main__':
    test_df = pd.read_csv('data/test.csv')
    true_label = []
    pred_label = []
    for index, row in test_df.iterrows():
        print(index)
        true_label.append(row['label'])
        pred_label.append(eval(row['content']))

    print(classification_report(true_label, pred_label, digits=4))
