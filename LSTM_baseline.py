# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np
from time import time
from data_process import *
from score import Score
from read_data import ReadData
from data_preprocess import *
from Network_base import Network_base

#import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

class LSTM_ECG(Network_base):

    # 初始化模型的基本参数
    def __init__(self):
        Network_base.__init__(self)       # 初始化父类中的属性
        # 模型参数
        self.n_hidden         = None
        self.n_layers         = None
        self.n_classes        = None
        self.time_step        = None

    def set_modelparams(self, dict):
        self.n_hidden         = dict['n_hidden']
        self.n_layers         = dict['n_layers']
        self.n_channels       = dict['n_channels']
        self.time_step        = dict['time_step']


    def create_placeholder(self):
        # X.shape =======> (number of samples, time_step, n_channels)
        X = tf.placeholder(tf.float32, [None, self.time_step, self.n_channels], name='X')   # self.time_step
        # Y.shape =======> (number of samples, n_classes)
        y_true = tf.placeholder(tf.float32, [None, self.n_classes], name='y_true')
        #batch_size = tf.placeholder(tf.int32, [])
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        batch_maxlen = tf.placeholder(tf.int32, name='batch_maxlen')
        x_lengths = tf.placeholder(tf.float32, [None], name='x_length')

        return X, y_true, keep_prob, batch_maxlen, x_lengths

    def _initial_weights(self):
        weights = {'in': tf.Variable(tf.random_normal([self.n_channels, self.n_hidden]), name='weights_in'),
                   'out': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]), name='weights_out')}
        return weights

    def _initial_bias(self):
        bias = {'in': tf.Variable(tf.random_normal([self.n_hidden]), name='bias_in'),
                'out': tf.Variable(tf.random_normal([self.n_classes]), name='bias_out')}
        return bias

    # def clip_gradients(self, cost):
    #     train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
    #     gradients = train_op.compute_gradients(cost)
    #     capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
    #     optimizer = train_op.apply_gradients(capped_gradients)
    #     return train_op


    def train(self, X_tr, y_tr, X_valid = None, y_valid = None):


        tf.reset_default_graph()

        X, y_true, keep_prob, batch_maxlen, x_lengths = self.create_placeholder()
        weights = self._initial_weights()
        biases = self._initial_bias()

        # 定义一个全连接层
        x_batch = tf.shape(X)[0]     # 输入batch的样本个数
        x_in = tf.reshape(X, [-1, self.n_channels])
        X_in = tf.matmul(x_in, weights['in']) + biases['in']
        X_in = tf.reshape(X_in, [x_batch, batch_maxlen, self.n_hidden])

        # # 定义LSTM模型
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden, forget_bias = 1.0, state_is_tuple = True)
        lstm_drop = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob = keep_prob)
        multi_cell = tf.contrib.rnn.MultiRNNCell([lstm_drop for _ in range(self.n_layers)], state_is_tuple = True)

        #init_state = multi_cell.zero_state(x_batch, dtype=tf.float32)

        # tf.nn.dynamic_rnn输出参数：
        # outputs: [max_time, batch_size, cell.output_size]
        # state: 最后的状态，包含每一层的最后状态
        outputs, final_state = tf.nn.dynamic_rnn(cell = multi_cell,
                                           inputs = X_in,
                                            sequence_length=x_lengths,
                                           dtype = tf.float32)

        max_output = tf.reduce_max(outputs, axis=1)
        average_output = tf.reduce_mean(outputs, axis=1)
        last_output = outputs[:, -1, :]
        con_output = tf.concat([max_output, average_output, last_output], axis=1)
        final_output = tf.contrib.layers.fully_connected(inputs=con_output,
                                                         num_outputs=self.n_hidden,
                                                         activation_fn=None)

        pred = tf.layers.dense(inputs=final_output, units=self.n_classes)


        cost = self.compute_cost(pred, y_true)

        # with tf.name_scope('train'):
        #     # Grad clipping
        #     # tf.train.AdamOptimizer函数默认参数ersilon = 1e-08
        #     train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        #     gradients = train_op.compute_gradients(cost)
        # with tf.name_scope('clip_value'):
        #     capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
        #     optimizer = train_op.apply_gradients(capped_gradients)
        with tf.name_scope('train'):
            # Grad clipping
            # tf.train.AdamOptimizer函数默认参数ersilon = 1e-08
            all_vars = tf.trainable_variables()
            train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            gradients = train_op.compute_gradients(cost, all_vars)
        with tf.name_scope('clip_value'):
            capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
            optimizer = train_op.apply_gradients(capped_gradients)

        y_pred = tf.argmax(pred, 1, name='y_pred')
        true_y = tf.argmax(y_true, 1)    # 样本的真实标签（数值）
        accuracy = self.compute_accuracy(y_pred, true_y)

        # 用于存储每个epoch的训练集和测试集的损失函数和正确率
        train_loss = []; train_acc = []; train_score = []
        valid_loss = []; valid_acc = []; valid_score = []

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        self.sess = self.create_session()
        self.sess.run(init)


        train_labels, train_mini_batches = self.batch_onehot(X_tr, y_tr)

        if (X_valid is not None) and (y_valid is not None):
            valid_labels, valid_mini_batches = self.batch_onehot(X_valid, y_valid)

        if self.restore_model:
            ckpt = tf.train.latest_checkpoint('./checkpoint_dir')  # 如果存在Checkpoint返回最新的checkpoint的路径，否则返回None
            if ckpt:
                saver.restore(self.sess, ckpt)     # 如果存在保存的模型，加载模型
            else:
                print("不存在保存的模型和参数，初始化参数并开始训练.")

        start_time = time()
        max_score = 0.70
        for e in range(self.training_epochs):
            #if e > 100:
            self.learning_rate = self.learning_rate * 0.99

            epoch_losses = []; epoch_acc = []; batch = 0

            train_pred = np.array([])
            epoch_start = time()
            for batch_x, batch_y in train_mini_batches:
                batch = batch + 1
                lengths = [sample.shape[0] for sample in batch_x]  # sample.shape[0]表示样本的时间序列长度
                batch_x, batch_maxtime = padMatrix(batch_x, lengths)
                feed = {X: batch_x,
                        y_true: batch_y,
                        keep_prob: self.keep_prob,
                        batch_maxlen: batch_maxtime,
                        x_lengths: lengths}
                loss, _, predictions, acc = self.sess.run([cost, optimizer, y_pred, accuracy], feed_dict = feed)
                epoch_losses.append(loss) ;    epoch_acc.append(acc)
                train_pred = np.concatenate([train_pred, predictions])    # 预测的样本标签

                if batch % 5 == 0:
                    print("Epoch  {}/{}:[============>]".format(e, self.training_epochs),
                          "batch {},".format(batch),
                          "training_loss:{:.4f},".format(loss),
                          "training_acc:{:.4f}".format(acc))

            train_pred = np.squeeze(train_pred.reshape((-1, 1)))

            score = self.compute_score(train_pred, train_labels)
            overall_score, Se, Sp = score

            epoch_end = time()
            print("Epoch  {}/{}:".format(e, self.training_epochs),
                  "train_loss:{:.4f},".format(np.mean(epoch_losses)),
                  "train_acc:{:.4f},".format(np.mean(epoch_acc)),
                  "overall_score:{:.4f},".format(overall_score),
                  "Se:{:.4f},".format(Se),
                  "Sp:{:.4f},".format(Sp),
                  "time:{:.2f}s".format(epoch_end - epoch_start))

            train_loss.append(np.mean(epoch_losses));  train_acc.append(np.mean(epoch_acc))
            train_score.append(list(score))

            if (X_valid is not None) and (y_valid is not None):
                valid_start = time()
                epoch_vlosses = [];  epoch_vacc = []; valid_pred = np.array([])  # 用于保存每个epoch测试集的结果
                for  x_t, y_t in valid_mini_batches:
                    lengths = [sample.shape[0] for sample in x_t]
                    x_t, batch_maxtime = padMatrix(x_t, lengths)
                    feed = {X: x_t,
                            y_true: y_t,
                            keep_prob: 1.0,
                            batch_maxlen:batch_maxtime,
                            x_lengths: lengths}         #, batch_times: batch_maxtime

                    loss, valid_predictions, acc = self.sess.run([cost, y_pred, accuracy], feed_dict=feed)
                    epoch_vlosses.append(loss) ;      epoch_vacc.append(acc)
                    valid_pred = np.concatenate([valid_pred, valid_predictions])

                valid_pred = np.squeeze(valid_pred.reshape((-1, 1)))

                v_score = self.compute_score(valid_pred, valid_labels)
                overall_score, valid_Se, valid_Sp = v_score
                valid_end = time()
                print("Epoch  {}/{}:".format(e, self.training_epochs),
                      "valid_loss:{:.4f},".format(np.mean(epoch_vlosses)),
                      "valid_acc:{:.4f},".format(np.mean(epoch_vacc)),
                      "overall_score:{:.4f},".format(overall_score),
                      "Se:{:.4f},".format(valid_Se),
                      "Sp:{:.4f},".format(valid_Sp),
                      "time:{:.2f}s".format(valid_end - valid_start))

                valid_loss.append(np.mean(epoch_vlosses)); valid_acc.append(np.mean(epoch_vacc))
                valid_score.append(list(v_score))

                if overall_score > max_score:   # 保存测试集正确率最大的模型
                    #max_acc = np.mean(epoch_vacc)
                    max_score = overall_score
                    saver.save(self.sess, "./checkpoint_dir/swt4/lstm", global_step=e+1)
            # if self.save_result:
            #     np.savetxt('./result/train_pred.txt', train_pred, fmt="%d", delimiter=" ")
            #     if (X_valid is not None) and (y_valid is not None):
            #         np.savetxt('./result/valid_pred.txt', valid_pred, fmt="%d", delimiter=" ")

        saver.save(self.sess, "./checkpoint_dir/swt4/lstm")   #保存最后一个epoch的模型

        if self.save_result:
            np.savetxt('./result/swt4/train_loss.txt', np.array(train_loss), fmt="%f", delimiter=" ")
            np.savetxt('./result/swt4/train_acc.txt', np.array(train_acc), fmt="%f", delimiter=" ")
            np.savetxt('./result/swt4/train_score.txt', np.array(train_score), fmt="%f", delimiter=" ")
            if (X_valid is not None) and (y_valid is not None):
                np.savetxt('./result/swt4/valid_loss.txt', np.array(valid_loss), fmt="%f", delimiter=" ")
                np.savetxt('./result/swt4/valid_acc.txt', np.array(valid_acc), fmt="%f", delimiter=" ")
                np.savetxt('./result/swt4/valid_score.txt', np.array(valid_score), fmt="%f", delimiter=" ")

        self.sess.close()



















