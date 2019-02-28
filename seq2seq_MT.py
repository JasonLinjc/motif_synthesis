# -*- coding: utf-8 -*-
# @Time     :2/28/19 11:14 AM
# @Auther   :Jason Lin
# @File     :seq2seq_MT.py
# @Software :PyCharm

import pickle as pkl
import numpy as np
import pandas as pd
import os, sys, time, random
import h5py
from keras.models import Model
from keras.layers import Input, LSTM, Dense


batch_size = 1
epochs = 200
latent_dim = 128
