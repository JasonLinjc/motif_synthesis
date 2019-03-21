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

def seq2seq_attention():
    Tx = 18
    repeator = RepeatVector(Tx)
    concatenator = Concatenate(axis=-1)
    densor_tanh = Dense(32, activation="tanh")
    densor_relu = Dense(1, activation="relu")
    activator = Activation(softmax, name='attention_weights')
    dotor = Dot(axes=1)