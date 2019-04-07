# -*- coding: utf-8 -*-
# @Time     :4/1/19 2:43 PM
# @Auther   :Jason Lin
# @File     :seperate_seq2seq.py
# @Software :PyCharm

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
import motif_encoder_decoder
import os

