# -*- coding: utf-8 -*-

import numpy as np

max_len = 243997   # 样本最大长度
resample_rate = 0.5   # 重采样比例


# 各类别样本比例比例
class_distribution = [0.7950, 0.2050]
# 模型参数设置
parameters = {  'learning_rate': 0.0001,
                'batch_size': 64,
                'keep_prob': 0.8,
                'training_epochs': 400,
                'n_classes': 2,
                'l2_penalty':0.01,
                'class_penalty':0.01,
                'class_distribution':class_distribution,
                'gamma': 1,
                'save_result': True,
                'predict':False,

                'is_resample': False,
                'doFFT': False,
                'spectrogram': False,
                'bandpass_filter':False,
                'remove_spike': True,
                'normalize': True,
                'MFCC':True,
                'wavelet':True,

                'restore_model': False,
                'data_split': True,

                'model_params':
                {'n_hidden': 100,
                 'n_layers': 2,
                 'n_channels': 40,
                 'time_step': None,
                }
              }
