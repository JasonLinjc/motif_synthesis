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


def seq2seq_attention():
    