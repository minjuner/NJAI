# -*- coding: utf-8 -*-


import numpy as np
import math
from sklearn.model_selection import train_test_split

def data_split(data, labels,folds = 5, valid_fold = 1, seed = 12306):
    """
    data: 样本数据集特征，一个列表，列表的每一个元素为一个样本（以数组形式存储）
    labels: 数据集对应的标签，形状：（样本数量, 1）
    seed: 随机数种子
    valid_fold: 返回第几折数据作为验证集，此参数配合valid_fold有效
    folds: 交叉验证的折数，默认为5
    返回：
    train_data: 用于训练的数据集，以列表形式存储，形状：（样本数量，）
    train_label:
    test_data:
    test_label:
    """
    classes = np.unique(labels['classes'])    # 样本的标签种类
    # 先找出每种类别的样本下标
    classes_index = {'index_'+str(clas): list(labels[labels['classes']==clas].index) for clas in classes}

    train_data = []; train_label = []; test_data = []; test_label = []
    # 将各个类别按比例分成训练集和测试集
    for clas in classes:
        np.random.seed(seed)
        random_index = np.random.permutation(classes_index['index_'+str(clas)])

        train_index, test_index = k_folds(random_index, folds=folds, n_fold=valid_fold)

        x_train = [data[ind] for ind in train_index]
        x_test  = [data[ind] for ind in test_index]

        y_train = np.array(labels.loc[train_index])
        y_test = np.array(labels.loc[test_index])

        train_data.extend(x_train)
        train_label.extend(y_train)
        test_data.extend(x_test)
        test_label.extend(y_test)
        seed += 1

    train_label = np.array(train_label)
    test_label = np.array(test_label)
    #dataset_train = (train_data, train_label)
    #dataset_test = (test_data, test_label)
    #return dataset_train, dataset_test
    return train_data, train_label, test_data, test_label

def k_folds(data_index, folds=5, n_fold=1):
    """
    :param data_index: 数据集的下标，数组形式
    :param folds: 交叉验证折数，默认为5
    :param n_fold: 第几折数据作为验证集，比如n_fold=1时，第一折数据作为测试集
    :return: 训练集和测试集的下标
    """
    num_data = len(data_index)
    fold_size = int(num_data / folds)
    if n_fold <=0 or n_fold > folds:
        raise ValueError("输入的n_fold错误！")
    test_index = data_index[(n_fold-1)*fold_size:n_fold * fold_size]
    train_index = np.setdiff1d(data_index, test_index)
    return train_index, test_index


def one_hot(labels, n_classes = 2):
    """One_Hot编码
    参数：
    labels: 单个样本标签
    n_classes: 样本种类数量
    返回：
    y:样本标签的One Hot编码，形状为（样本数量，1）
    """
    expansion = np.eye(n_classes)
    y = expansion[labels, :]
    #assert y.shape[1] == n_classes
    return y


def get_mini_batches(data, labels = None, batch_size = 64, shuffled = True, seed = 12345):
    """
    :param data:  输入的样本集, 列表，每一个元素代表一个样本
    :param labels:
    :return: 一个列表  样本集对应的标签, 数组：（number_of_exmaples, num_of_classes）
    :param batch_size:  批量大小
    :param seed:  随机数种子
    """
    m = len(data)    # 数据集的数量
    mini_batches = []
    if shuffled == True:
        np.random.seed(seed)
        # 随机打乱 data, label
        permutation = list(np.random.permutation(m))
        shuffled_data = [data[per] for per in permutation]
        if labels is not None:
            shuffled_label = labels[permutation, :]
    else:
        shuffled_data = data
        if labels is not None:
            shuffled_label = labels

    # 划分打乱后的数据集
    num_complete_minibatches = math.floor(m/batch_size)     # 完整的mini_batch数量
    for k in range(0, num_complete_minibatches):
        mini_batch_data = shuffled_data[k * batch_size : k * batch_size + batch_size]
        if labels is not None:
            mini_batch_label = shuffled_label[k * batch_size : k * batch_size + batch_size, :]
            mini_batch = (mini_batch_data, mini_batch_label)
        else:
            mini_batch = mini_batch_data
        mini_batches.append(mini_batch)

    # 如果数据集的样本个数不能整除batch_size,即划分后仍有剩余的样本数量小于batch_size
    if m % batch_size != 0:
        mini_batch_data = shuffled_data[num_complete_minibatches * batch_size : m]
        if labels is not None:
            mini_batch_label = shuffled_label[num_complete_minibatches * batch_size : m, :]
            mini_batch = (mini_batch_data, mini_batch_label)
        else:
            mini_batch = mini_batch_data
        mini_batches.append(mini_batch)

    return mini_batches



def padMatrix(seq_data, lengths = None):
    """
    :param seq_data: 时间序列数据，列表， 形状为（batch_size, time_steps, n_channels）
    :param lengths: 每个样本对应的时间序列长度
    :return:
    """

    if lengths == None:
        lengths = [len(s) for s in seq_data]
    n_samples = len(lengths)
    max_times = np.max(lengths)
    x = seq_data
    n_channel = seq_data[0].shape[1]
    #n_channel = 4
    x = np.zeros((n_samples, max_times, n_channel))
    for idx, s in enumerate(seq_data):
        x[idx, :lengths[idx]] = s
    return x, max_times

