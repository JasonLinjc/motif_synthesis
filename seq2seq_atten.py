# -*- coding: utf-8 -*-
# @Time     :3/20/19 2:05 PM
# @Auther   :Jason Lin
# @File     :seq2seq_atten.py
# @Software :PyCharm

import warnings
warnings.filterwarnings("ignore")

from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply, Reshape
from keras.layers import RepeatVector, Dense, Activation, Lambda, Embedding
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import keras
import numpy as np
import pickle as pkl
import random
import tqdm
import matplotlib.pyplot as plt



def softmax(x, axis=1):
    """
    Softmax activation function.
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')

Tx = 18
repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor_tanh = Dense(32, activation = "tanh")
densor_relu = Dense(1, activation = "relu")
activator = Activation(softmax, name='attention_weights')
dotor = Dot(axes = 1)

def one_step_attention(a, s_prev):
    """
    Attention机制的实现，返回加权后的Context Vector

    @param a: BiRNN的隐层状态
    @param s_prev: Decoder端LSTM的上一轮隐层输出

    Returns:
    context: 加权后的Context Vector
    """
    # 将s_prev复制Tx次
    s_prev = repeator(s_prev)
    # 拼接BiRNN隐层状态与s_prev
    concat = concatenator([a, s_prev])
    # 计算energies
    e = densor_tanh(concat)
    energies = densor_relu(e)
    # 计算weights
    alphas = activator(energies)
    # 加权得到Context Vector
    context = dotor([alphas, a])

    return context

target_seq_dim = 6

n_a = 32 # The hidden size of Bi-LSTM
n_s = 128 # The hidden size of LSTM in Decoder
decoder_LSTM_cell = LSTM(n_s, return_state=True)
output_layer = Dense(target_seq_dim, activation=softmax)

# 定义网络层对象（用在model函数中）
reshapor = Reshape((1, target_seq_dim))
concator = Concatenate(axis=-1)


def model(Tx, Ty, n_a, n_s, source_seq_dim, target_seq_dim):
    """
    构造模型
    @param Tx: 输入序列的长度
    @param Ty: 输出序列的长度
    @param n_a: Encoder端Bi-LSTM隐层结点数
    @param n_s: Decoder端LSTM隐层结点数
    @param source_vocab_size: 输入（英文）语料的词典大小
    @param target_vocab_size: 输出（法语）语料的词典大小
    """
    # 定义输入层
    X = Input(shape=(Tx,))
    # Embedding层
    # embed = embedding_layer(X)
    # Decoder端LSTM的初始状态
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')

    # Decoder端LSTM的初始输入
    out0 = Input(shape=(target_seq_dim,), name='out0')
    out = reshapor(out0)

    s = s0
    c = c0

    # 模型输出列表，用来存储翻译的结果
    outputs = []

    # 定义Bi-LSTM
    a = Bidirectional(LSTM(n_a, return_sequences=True))(X)

    # Decoder端，迭代Ty轮，每轮生成一个翻译结果
    for t in range(Ty):
        # 获取Context Vector
        context = one_step_attention(a, s)

        # 将Context Vector与上一轮的翻译结果进行concat
        context = concator([context, reshapor(out)])
        s, _, c = decoder_LSTM_cell(context, initial_state=[s, c])

        # 将LSTM的输出结果与全连接层链接
        out = output_layer(s)

        # 存储输出结果
        outputs.append(out)

    model = Model([X, s0, c0, out0], outputs)

    return model



