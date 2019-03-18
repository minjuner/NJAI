# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np
import os
from time import time
from data_process import *
from score import Score
from read_data import ReadData
from data_preprocess import *
import pandas as pd

class Network_base:

    # 初始化模型的基本参数
    def __init__(self):
        # 训练参数
        self.learning_rate      = None
        self.batch_szie         = None
        self.keep_prob          = None
        self.training_epochs    = None
        self.n_classes          = None
        self.l2_penalty         = None
        self.class_penalty      = None
        self.class_distribution = None
        self.gamma              = None
        self.predict            = None

        self.MFCC               = None
        self.normalize          = None
        self.remove_spike       = None
        self.save_result        = None
        self.is_resample        = None
        self.restore_model      = None
        self.data_split         = None
        self.spectrogram        = None
        self.doFFT              = None
        self.bandpass_filter    = None
        self.wavelet            = None

    def compute_cost(self, pred, y_true):

        # if self.class_penalty != 0 :
        equal_w = [1 for _ in self.class_distribution]
        penal_w = [1/d for d in self.class_distribution]
        penal = self.class_penalty
        weights = [[e * (1 - penal) + p * penal for e,p in zip(equal_w, penal_w)]]
        class_weights = tf.constant(weights, dtype=tf.float32)

        prob = tf.nn.softmax(pred)
        fl_factor = tf.pow(tf.subtract(1.0, prob), self.gamma)
        fl_factor = tf.multiply(y_true, fl_factor)
        weight_per_sample = tf.matmul(class_weights, tf.transpose(fl_factor))

        all_vars = tf.trainable_variables()
        vars = [v for v in all_vars
                if 'bias' not in v.name and
                'bias_in' not in v.name and
                'bias_out' not in v.name]
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in vars])

        with tf.name_scope('cost'):
            #softmax = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y_true) 出错：AttributeError: module 'tensorflow.python.ops.nn' has no attribute 'softmax_cross_entropy_with_logits_v2'
            softmax = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_true)

            softmax = tf.reduce_mean(tf.multiply(weight_per_sample, softmax))
            cost = tf.reduce_mean(softmax + self.l2_penalty * l2_loss)
            # softmax = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y_true))
            # cost = softmax + self.l2_penalty * l2_loss

        return cost

    def compute_accuracy(self, y_pred, true_y, name = 'accuracy'):
        correct_pred = tf.equal(y_pred, true_y)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name=name)
        return accuracy


    def load_inputs(self, load_path, ext_len = None, resample_rate = 0.2):
        label_set = None
        print("读取数据...")
        read_data = ReadData(normalize=self.normalize)
        if self.predict == True:
            data_set = read_data.load_data(path=load_path, ext_len=ext_len)
        else:
            data_set, label_set = read_data.get_data(path=load_path, ext_len = ext_len)
        print("数据完读取完毕...")
        print("数据集的样本个数：{}".format(len(data_set)))

        if self.is_resample:
            print("数据降采样...")
            data_set = [random_resample(sample.T, resample_rate = resample_rate) for sample in data_set]

        if self.spectrogram:
            data_set = [ecg_spectrogram(sample.T) for sample in data_set]

        if self.doFFT:
            print("FFT变换...")
            data_set = [doFFT(sample.T) for sample in data_set]

        if self.bandpass_filter:
            print("bandpass_filter...")
            data_set = [bandpass_filter(sample,20,400,1000,6) for sample in data_set]

        if self.wavelet:
            print("计算小波系数....")
            data_set = [pywt_swt(sample) for sample in data_set]

        if self.remove_spike:
            print("去除峰值...")
            data_set = [schmidt_spike_removal(sample.T) for sample in data_set]

        if self.MFCC:
            print("计算MFCC....")
            data_set = [MFCC(sample) for sample in data_set]

        if self.data_split and label_set is not None:
            print("划分数据集...")
            X_tr, y_tr, X_valid, y_valid = data_split(data_set, label_set, 5, 4)  # 划分训练集和测试集
            print("数据集划分完毕...")
            return  (X_tr, y_tr, X_valid, y_valid)
        else:
            if self.predict == True:
                return data_set
            else:
                return (data_set, label_set)

    def compute_score(self, pred, labels):
        score = Score()
        sc = score.get_score(pred, labels)
        return sc

    def evaluation(self, X_test, graph_path, label_test = None):
        tf.reset_default_graph()
        new_sess = self.create_session()
        try:
            # 加载图结构和参数
            saver = tf.train.import_meta_graph(graph_path)  # 加载图
            model_dir = os.path.dirname(graph_path)
            saver.restore(new_sess, tf.train.latest_checkpoint(model_dir))
        except NotImplementedError:
            print("模型加载失败...")
        if label_test is not None:
            test_labels, test_mini_batches = self.batch_onehot(X_test, label_test)
        else:
            test_mini_batches = get_mini_batches(X_test, shuffled=False)

        # 访问placeholders变量，并且创建feed-dict来作为placeholders的新值
        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name("X:0")
        if label_test is not None:
            y_true = graph.get_tensor_by_name("y_true:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        # 访问想要执行的变量
        y_pred = graph.get_tensor_by_name("y_pred:0")
        batch_maxlen = graph.get_tensor_by_name("batch_maxlen:0")
        x_length = graph.get_tensor_by_name("x_length:0")

        predictions = np.array([])
        for batch in test_mini_batches:
            lengths = [sample.shape[0] for sample in batch]  # sample.shape[0]表示样本的时间序列长度
            batch, batch_maxtime = padMatrix(batch, lengths)
            if self.predict==True and label_test is None:
                feed = {X: batch,  keep_prob: 1.0, batch_maxlen:batch_maxtime, x_length:lengths}
            else:
                feed = {X:batch[0], y:batch[1], keep_prob:1.0, batch_maxlen:batch_maxtime, x_length:lengths}
            pred = new_sess.run(y_pred, feed_dict=feed)
            predictions = np.concatenate([predictions, pred])  # 预测的样本标签
        predictions = np.squeeze(predictions.reshape((-1, 1)))
        predictions = np.where(predictions==0, -1, predictions)
        new_sess.close()
        return predictions


    def prediction(self, X_test):

        model1_path = './checkpoint_dir/model1/lstm-378.meta'
        model2_path = './checkpoint_dir/model2/lstm-71.meta'
        model3_path = './checkpoint_dir/model3/lstm-73.meta'
        model4_path = './checkpoint_dir/model4/lstm-53.meta'
        model5_path = './checkpoint_dir/model5/lstm-55.meta'
        classifyResult1 = self.evaluation(X_test, model1_path)
        classifyResult2 = self.evaluation(X_test, model2_path)
        classifyResult3 = self.evaluation(X_test, model3_path)
        classifyResult4 = self.evaluation(X_test, model5_path)
        classifyResult5 = self.evaluation(X_test, model5_path)

        result = pd.DataFrame([classifyResult1, classifyResult2, classifyResult3, classifyResult4, classifyResult5]).T
        result = np.array(result.median(axis=1))
        return result

    def create_session(self):
        # 创建新的会话窗口
        config = tf.ConfigProto(allow_soft_placement=True)  # allow_soft_placement=True:如果指定的设备不存在，允许TF自动分配设备
        config.gpu_options.allow_growth = True
        new_sess = tf.Session(config=config)
        return new_sess

    def lookup_model_params(self, graph_path, print_tensor_name = True):
        new_sess = self.create_session()
        try:
            # 加载图结构和参数
            saver = tf.train.import_meta_graph(graph_path)  # 加载图
            model_dir = os.path.dirname(graph_path)
            saver.restore(new_sess, tf.train.latest_checkpoint(model_dir))
        except NotImplementedError:
            print("模型加载失败...")
        graph = tf.get_default_graph()
        all_tensor_name = [n.name for n in graph.as_graph_def().node]
        if print_tensor_name:
            print("打印模型的所有参数名字")
            print(all_tensor_name)

        return all_tensor_name


    def batch_onehot(self, X, y):
        """输入：
        X:数据集
        y: 数据集对应的标签，每个样本有3个标签
        返回：
        labels:数据集完整的标签
        mini_batches:样本划分batch，是一个列表，其中每个元素包含（batch_x, batch_y）
        """
        mini_batches = get_mini_batches(X, y, self.batch_szie)  # 划分batch
        labels = []; batches = []
        for batch_x, batch_y in mini_batches:
            onehot_labels = one_hot(batch_y[:, 0])
            batch_labels = np.array(batch_y)
            labels.append(batch_labels)
            batch = (batch_x, onehot_labels)
            batches.append(batch)
        labels = np.vstack(np.array(labels))  # 样本完整的标签
        return labels, batches

    def set_parameters(self, dict):
        model_params = dict['model_params']
        self.set_modelparams(model_params)

        self.learning_rate      = dict['learning_rate']
        self.batch_szie         = dict['batch_size']
        self.keep_prob          = dict['keep_prob']
        self.training_epochs    = dict['training_epochs']
        self.n_classes          = dict['n_classes']
        self.l2_penalty         = dict['l2_penalty']
        self.class_penalty      = dict['class_penalty']
        self.class_distribution = dict['class_distribution']
        self.gamma              = dict['gamma']
        self.predict            = dict['predict']


        self.bandpass_filter    = dict['bandpass_filter']
        self.MFCC               = dict['MFCC']
        self.doFFT              = dict['doFFT']
        self.remove_spike       = dict['remove_spike']
        self.normalize          = dict['normalize']
        self.spectrogram        = dict['spectrogram']
        self.wavelet            = dict['wavelet']

        self.save_result        = dict['save_result']
        self.is_resample        = dict['is_resample']
        self.restore_model      = dict['restore_model']
        self.data_split         = dict['data_split']


    def set_modelparams(self, model_dict):
        raise NotImplementedError("没有特定模型，模型参数初始化失败！")
