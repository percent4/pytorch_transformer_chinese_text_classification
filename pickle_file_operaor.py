# -*- coding: utf-8 -*-
# @Time : 2023/3/16 10:49
# @Author : Jclian91
# @File : pickle_file_operaor.py
# @Place : Minghang, Shanghai
import pickle
from abc import ABCMeta, abstractmethod


# 父类: 文件操作
class FileOperator(metaclass=ABCMeta):
    def __init__(self, data, file_path):
        self.data = data
        self.file_path = file_path

    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def save(self):
        pass


# pickle文件操作
class PickleFileOperator(FileOperator):
    def __init__(self, data=None, file_path=''):
        super(PickleFileOperator, self).__init__(data, file_path)

    def save(self):
        with open(self.file_path, 'wb') as f:
            pickle.dump(self.data, f)

    def read(self):
        with open(self.file_path, "rb") as f:
            content = pickle.load(f)
        return content


# 模型文件操作
class ModelFileOperator(FileOperator):
    def __init__(self, data=None, file_path=''):
        super(ModelFileOperator, self).__init__(data, file_path)

    def save(self):
        pass

    def read(self):
        pass
