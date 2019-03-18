# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal
from scipy.fftpack import dct
import scipy as sc
from scipy.fftpack import fft
from scipy.signal import butter, lfilter
from sklearn.preprocessing import normalize
import librosa
import pywt
def ecg_spectrogram(data, fs = 1000, nperseg = 128, noverlap = 64, nfft = None):
    # 此函数是计算ECG信号经过短时傅里叶变换后时域频谱
    # data: 输入的样本ECG数据，形状为(n_channel, time_steps)
    # fs: ECG信号的频率
    # nperseg: 每个窗口的长度
    # noverlap: 相邻窗口重叠长度
    # 具体参数解释详见： https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html
    log_spectrogram = True
    _, _, Sxx = signal.spectrogram(data, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    Sxx = np.transpose(np.reshape(Sxx, (-1, Sxx.shape[2])))
    # Sxx = np.reshape(Sxx, (-1, Sxx.shape[2]))
    #Sxx = np.expand_dims(Sxx, axis=-1)

    if log_spectrogram == True:
        Sxx = abs(Sxx)
        mask = Sxx > 0
        Sxx[mask] = np.log(Sxx[mask])
    #print(Sxx.shape) :(,65)
    #格式             ：numpy.ndarray
    return Sxx

def bandpass_filter(data, lowcut, highcut, signal_freq, filter_order):
    """
    Method responsible for creating and applying Butterworth filter.
    :param deque data: raw data
    :param float lowcut: fi ,lter lowcut frequency value
    :param float highcut: filter highcut frequency value
    :param int signal_freq: signal frequency in samples per second (Hz)
    :param int filter_order: filter order
    :return array: filtered data
    """
    nyquist_freq = 0.5 * signal_freq
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b, a = butter(filter_order, [low, high], btype="band")
    y = lfilter(b, a, data)
    return y

def random_resample(data, resample_rate = 0.2, fit_length = None):
    # data : 样本的ECG数据，形状为(n_channel, time_steps)
    [n_channel, length] = data.shape
    #length = len(data)
    new_length = length * resample_rate
    interpol = sc.interpolate.interp1d(np.arange(length), data)
    resample_point = np.linspace(start=0, stop=length-1, num=new_length)
    resample = interpol(resample_point)
    return resample.T


def doFFT(data):
    fft_embedding_size = 400
    highpass_embedding_size = 200
    #data_fft = np.abs(fft(data))
    data_fft = np.abs(data)

    data_fft = data_fft[:,:fft_embedding_size].reshape(-1)


    nyquist = 1000
    cutoff_freq = 4.0
    w0, w1 = butter(5, cutoff_freq / nyquist, btype='low', analog=False)
    data_low_pass = lfilter(w0, w1, data)
    #data_low_pass_fft = np.abs(fft(data_low_pass))
    data_low_pass_fft = np.abs(data_low_pass)

    data_low_pass_fft = data_low_pass_fft[:, :highpass_embedding_size].reshape(-1)
    features = np.concatenate((data_fft, data_low_pass_fft))
    return features


def enframe(wave_data, nw, inc, winfunc):
    '''将音频信号转化为帧。
    参数含义：
    wave_data:原始音频型号
    nw:每一帧的长度(这里指采样点的长度，即采样频率乘以时间间隔)
    inc:相邻帧的间隔（同上定义）
    '''
    wlen = len(wave_data)  # 信号总长度
    if wlen <= nw:  # 若信号长度小于一个帧的长度，则帧数定义为1
        nf = 1
    else:  # 否则，计算帧的总长度
        nf = int(np.ceil((1.0 * wlen - nw + inc) / inc))
    pad_length = int((nf - 1) * inc + nw)  # 所有帧加起来总的铺平后的长度

    zeros = np.zeros((pad_length - wlen,))  # 不够的长度使用0填补，类似于FFT中的扩充数组操作
    pad_signal = np.concatenate((wave_data, zeros))  # 填补后的信号记为pad_signal
    indices = np.tile(np.arange(0, nw), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc),
                                                           (nw, 1)).T  # 相当于对所有帧的时间点进行抽取，得到nf*nw长度的矩阵
    # print(nf) :890
    indices = np.array(indices, dtype=np.int32)  # 将indices转化为矩阵
    frames = pad_signal[indices]  # 得到帧信号
    win = np.tile(winfunc, (nf, 1))  # window窗函数，这里默认取1
    return frames * win  # 返回帧信号矩阵

def max_remove(data):
    for i in range(len(data)):
        if abs(data[i])>data.max()/2:
            data[i]=0
    return data

def MFCC(data):
    data = data.reshape((-1,))
    # w = pywt.Wavelet('sym5')
    # mode = pywt.Modes.smooth
    # (a, d) = pywt.dwt(data, w, mode)
    # rec_a = pywt.waverec([a, None], w)
    # data = np.hstack((data,rec_a))
    mfcc = np.transpose(librosa.feature.mfcc(y=data.astype(np.float64), sr=2000, n_mfcc=40), [1, 0])
    return mfcc

def wavelet(data):
    data = data.reshape((-1,))
    w = pywt.Wavelet('sym5')
    mode = pywt.Modes.smooth
    (a, d) = pywt.dwt(data, w, mode)
    rec_a = pywt.waverec([a, None], w)
    return rec_a

def pywt_swt(signal):
    """
    :param signal: 形状：（time_step, 1)
    :return:
    """
    signal = np.reshape(signal, (-1,))
    if len(signal) % 2 != 0:
        signal = np.concatenate([signal, np.array([0])])
    ca = []; cd = []
    w = pywt.Wavelet('db4')
    coeffs = pywt.swt(signal, w, 1)  # [(cA1, cD1)]
    for a, d in reversed(coeffs):
        ca.append(a)
        cd.append(d)
    signal = pywt.waverec(ca, w)
    signal = np.reshape(signal, (-1, 1))
    return signal

def schmidt_spike_removal(signal, fs=2000):
    """
    :param signal: 原始信号(time_step, 1)
    :param fs: 采样频率
    :return: 移除噪音峰值的信号
    本函数参考论文：
    S. E. Schmidt et al., "Segmentation of heart sound recordings by
    duration-dependent hidden Markov model," Physiol. Meas., vol. 31,
    no. 4, pp. 513-29, Apr. 2010.
    """

    window_size = int(fs / 2)
    # 将信号按找窗口大小划分成一个个窗口，找到信号中剩余的不够一个完整窗口的采样点
    trailing_samples = len(signal) % window_size
    # 将信号转换成一些窗口,转换后的形状（window_size, -1）
    sample_frames = np.reshape(signal[:-trailing_samples], (window_size, -1))

    MAAs = np.max(abs(sample_frames), axis=0)
    is_greater = list(MAAs > (np.median(MAAs) * 3))
    while is_greater.count(True) > 0:
        max_val = np.max(MAAs)
        # 找到MAAs中最大值所在的窗口位置
        window_pos = np.argwhere(MAAs == max_val)
        # 如果等于最大值的窗口不止一个，则取第一个等于最大值的窗口
        if len(window_pos) > 1:
            window_pos = window_pos[0]
        window_pos = int(window_pos)
        # 找到窗口内的最大值，即噪音峰值
        spike_val = np.max(abs(sample_frames[:, window_pos]))
        spike_pos = np.argwhere(abs(sample_frames[:, window_pos]) == spike_val)

        if len(spike_pos) > 1:
            spike_pos = spike_pos[0]
        spike_pos = int(spike_pos)

        # 找到零交叉(这里可能没有实际的0值，只是从正到负的变化
        sign_change = abs(np.diff(np.sign(sample_frames[:, window_pos]))) > 1
        zero_crossing = np.concatenate([sign_change, np.array([False])])

        # 找到峰值的开始位置，峰值位置之前最后一个零交叉点的位置
        # 如果没有零交叉点，则取窗口开始的位置
        zero_cross_pos = np.argwhere(zero_crossing[0:spike_pos + 1] == True)
        if len(zero_cross_pos) == 0:
            spike_start = 0
        else:
            spike_start = np.squeeze(zero_cross_pos[-1])

        # 找到峰值的结束位置，峰值位置之后第一个零交叉点的位置
        # 如果没有零交叉点，则取窗口结束的位置


        zero_crossing[0:spike_pos + 1] = False
        zero_cross_pos = np.argwhere(zero_crossing == True)
        if len(zero_cross_pos) == 0:
            spike_end = window_size
        else:
            spike_end = np.squeeze(zero_cross_pos[0])
        # print("spike start pos is %d, spike pos is %d, spike end pos is %d" % (spike_start, spike_pos, spike_end))
        # 将噪音峰值设置为0
        sample_frames.flags.writeable = True
        sample_frames[spike_start:spike_end, window_pos] = 0.0001

        # 重新计算 MAAs
        MAAs = np.max(abs(sample_frames), axis=0)
        is_greater = list(MAAs > (np.median(MAAs) * 3))

    despike_signal = np.reshape(sample_frames, (-1, 1))
    signal = signal.reshape((-1,1))
    despike_signal = np.concatenate([despike_signal, signal[len(despike_signal):]])


    return despike_signal
