# -*- coding: utf-8 -*-

import numpy as np

class Score:
    """
    标签的含义：
    0: Normal
    1: Abnormal
    注：此处的0是由原来标签（-1）映射过来的，仅仅是为了方便将标签转化为one_hot而已。

    """
    def __init__(self):

        self.labels = [0, 1]     # 一共有2种类别

        # pred_count: 存储最终预测的每种类别的样本数量，初始化为0
        self.pred_count = np.zeros(len(self.labels))

        # actual_count: 存储实际的每种类别的样本数量，初始化为0
        self.actual_count = np.zeros(len(self.labels))

        # pred_true_count: 存储预测正确的每种类别的样本数量，初始化为0
        self.pred_true_count = np.zeros(len(self.labels))

    def get_score(self, y_pred, y_true):
        """
        :param y_pred: 预测的标签类别，数组类型存储，形状为（m, 1）.其中m为样本个数
        :param y_true:  实际的样本标签类别，数组类型存储，此处为一个 (m, 1) 的数组.其中m为样本个数
        评分规则
                                                 predict label
                                           Normal(0)     Abnormal(1)
        reference label   Normal(0)           Nn              Na
                          Abnormal(1)         An              Aa

        Se = Aa / (An + Aa)
        Sp = Nn / (Nn + Na)
        overall_score = (Se + Sp) / 2

        :return: overall_score, Se, Sp
        """

        for i in range(len(y_pred)):
            yi_pred = y_pred[i]     # 第 i 个样本的预测类别标签y
            yi_true = y_true[i]     # 第 i 个样本的真实类别标签y

            if (yi_pred == yi_true):
                self.pred_true_count[yi_true] += 1
                self.pred_count[yi_true] += 1
            else:
                self.pred_count[int(yi_pred)] += 1
            self.actual_count[yi_true] += 1

        Se = self.pred_true_count[1] / self.actual_count[1]
        Sp = self.pred_true_count[0] / self.actual_count[0]
        overall_score = (Se + Sp) / 2

        return (overall_score, Se, Sp)













