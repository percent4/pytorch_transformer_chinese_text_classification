# -*- coding: utf-8 -*-
# @Time : 2023/3/21 19:51
# @Author : Jclian91
# @File : get_pretrained_vector.py
# @Place : Gusu, Suzhou
import torch
from gensim.models import KeyedVectors

from params import NUM_WORDS
from text_featuring import load_file_file

# 读取转换后的文件
label_dict, char_dict = load_file_file()

# 加载转化后的文件
model = KeyedVectors.load_word2vec_format('./Pretrain_Vector/sgns.wiki.char.bz2',
                                          binary=False,
                                          encoding="utf-8",
                                          unicode_errors="ignore")
# 使用gensim载入word2vec词向量
pretrained_vector = torch.zeros(NUM_WORDS + 4, 300).float()
# print(model.index2word)

for char, index in char_dict.items():
    if char in model.vocab:
        vector = model.get_vector(char)
        # print(vector)
        pretrained_vector[index, :] = torch.from_numpy(vector)
