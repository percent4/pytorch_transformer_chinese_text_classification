# -*- coding: utf-8 -*-
# @Time : 2023/3/16 10:32
# @Author : Jclian91
# @File : preprocessing.py
# @Place : Minghang, Shanghai
import pandas as pd
from random import shuffle
from operator import itemgetter
from collections import Counter, defaultdict

from params import TRAIN_FILE_PATH, NUM_WORDS
from pickle_file_operaor import PickleFileOperator


class FilePreprossing(object):
    def __init__(self, n):
        # 保留前n个高频字
        self.__n = n

    def _read_train_file(self):
        train_pd = pd.read_csv(TRAIN_FILE_PATH)
        label_list = train_pd['label'].unique().tolist()
        # 统计文字频数
        character_dict = defaultdict(int)
        for content in train_pd['content']:
            for key, value in Counter(content).items():
                character_dict[key] += value
        # 不排序
        sort_char_list = [(k, v) for k, v in character_dict.items()]
        shuffle(sort_char_list)
        # 排序
        # sort_char_list = sorted(character_dict.items(), key=itemgetter(1), reverse=True)
        print(f'total {len(character_dict)} characters.')
        print('top 10 chars: ', sort_char_list[:10])
        # 保留前n个文字
        top_n_chars = [_[0] for _ in sort_char_list[:self.__n]]

        return label_list, top_n_chars

    def run(self):
        label_list, top_n_chars = self._read_train_file()
        PickleFileOperator(data=label_list, file_path='labels.pk').save()
        PickleFileOperator(data=top_n_chars, file_path='chars.pk').save()


if __name__ == '__main__':
    processor = FilePreprossing(NUM_WORDS)
    processor.run()
    # 读取pickle文件
    labels = PickleFileOperator(file_path='labels.pk').read()
    print(labels)
    content = PickleFileOperator(file_path='chars.pk').read()
    print(content)
