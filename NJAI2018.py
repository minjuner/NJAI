# -*- coding: utf-8 -*-

from LSTM_baseline import LSTM_ECG
from data_preprocess import *
from set_params import *
import numpy as np


# 请在data_path后面输入测试集所在的文件路径，
# 比如，测试集所在路径为 D:/CPSC_2018/TestSet
# TestSet文件夹中的文件为测试集

# Please enter the file path where the test set is located.
# For example, the path of the test set is D:/CPSC_2018/TestSet
# The file in the TestSet folder is test set

train_path='G:/heartsound/data/training'

lstm_model = LSTM_ECG()
lstm_model.set_parameters(parameters)
X_train, y_train, X_valid, y_valid = lstm_model.load_inputs(train_path)

print("开始训练模型...")
lstm_model.train(X_train, y_train, X_valid, y_valid)

# X_test = lstm_model.load_inputs(data_path)
#
# classifyResult = lstm_model.prediction(X_test)
#
# # 如果要保存预测结果，将save_result设置为True，则会将测试集的预测结果保存在当前路径，默认为True
# # If you want to save the forecast result, set 'save_result' to True, it will save the test set's
# # forecast result in the current path. The dafault value is True.
#
# save_result = True
# if save_result == True:
#     np.savetxt('classifyResult.txt', classifyResult, fmt = '%d', delimiter = ' ')

