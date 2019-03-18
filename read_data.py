# -*- coding: utf-8 -*-


import os
import os.path
import scipy.io
import pandas as pd
import numpy as np
from scipy.io import wavfile

class ReadData:
    def __init__(self, normalize = True):

        self.path           = ""
        self.label_filename = 'REFERENCE.csv'
        self.normalize      = normalize
        self.label_map      = {-1:0, 1:1}

    def read_labels(self):
        label_file_path = os.path.join(self.path, self.label_filename)
        labels = pd.read_csv(label_file_path, names = ['record', 'classes'], header = None)
        #labels.drop(columns=['record'], inplace=True)   出错：TypeError: drop() got an unexpected keyword argument 'columns'
        labels.drop('record',axis=1,inplace=True)
        labels['classes'] = labels['classes'].map(self.label_map)
        labels['classes'] = labels['classes'].astype(int)
        return labels

    def load_data(self, path, ext_len = None):
        """
        :param path: 数据的存储路径
        :return: 数据列表，每个元素是一个样本
        """
        dataset_file_names = [name for name in os.listdir(path) if name.endswith('.wav')]

        if dataset_file_names == []:
            return []

        dataset = []
        for name in dataset_file_names:
            data_path = os.path.join(path, name)     # 数据集路径
            sample = self.load_sample(data_path)
            if ext_len is not None:
                sample = self.extend_signal(sample, ext_len)
            if self.normalize:
                sample = (sample - np.mean(sample)) / np.std(sample)
            dataset.append(sample)

        return dataset

    def load_sample(self, path):
        # 读取单个样本，输入的是路径，输出的是形状为 (序列长度, 1) 的数组数据
        _, wave = wavfile.read(path)

        wave = wave.reshape((-1, 1))
        return wave

    def extend_signal(self, signal, length):
        # 输入的signal是一个样本的PCG数据，形状为(序列长度, 1)
        extend = np.zeros((length, 1))
        signal_len = np.min([length, signal.shape[0]])
        extend[-signal_len:] = signal[:signal_len]
        return extend

    def get_data(self, path, ext_len = None):
        self.path = path
        labels = self.read_labels()
        dataset = self.load_data(path, ext_len = ext_len)
        return dataset, labels







