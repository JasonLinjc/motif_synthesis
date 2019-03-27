# -*- coding: utf-8 -*-
# @Time     :3/20/19 2:05 PM
# @Auther   :Jason Lin
# @File     :seq2seq_atten.py
# @Software :PyCharm

import warnings
# warnings.filterwarnings("ignore")
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

Tx = 18 + 18 # Source sequence max length
Ty = 31 + 2  # Target sequence max length
source_dim = 4
target_dim = 6

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

n_a = 32 # The hidden size of Bi-LSTM
n_s = 128 # The hidden size of LSTM in Decoder
decoder_LSTM_cell = LSTM(n_s, return_state=True)
output_layer = Dense(target_dim, activation=softmax)

# 定义网络层对象（用在model函数中）
reshapor = Reshape((1, target_dim))
concator = Concatenate(axis=-1)


def seq_model(Tx, Ty, n_a, n_s, source_seq_dim, target_seq_dim):
    """
    构造模型
    @param Tx: 输入序列的长度
    @param Ty: 输出序列的长度
    @param n_a: Encoder端Bi-LSTM隐层结点数
    @param n_s: Decoder端LSTM隐层结点数
    """
    # 定义输入层
    X = Input(shape=(Tx, source_seq_dim))
    # Embedding层
    # embed = embedding_layer(X)
    # Decoder端LSTM的初始状态
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')

    # Decoder端LSTM的初始输入
    out0 = Input(shape=(target_seq_dim, ), name='out0')
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

    # outputs = np.array(outputs)

    model = Model([X, s0, c0, out0], outputs)

    return model

model = seq_model(Tx, Ty, n_a, n_s, source_dim, target_dim)
# model.summary()
out = model.compile(optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.001),
                    metrics=['accuracy'],
                    loss='categorical_crossentropy')

def load_sequence_data(family_name = "bHLH_Homeo"):
    if family_name == "all":
        seq_info = motif_encoder_decoder.motif_encoder.get_all_dimer()
    else:
        seq_info = motif_encoder_decoder.motif_encoder.get_sequence_family_input(family_name)
    print(seq_info)
    motif1_seqs = seq_info[2]
    motif2_seqs = seq_info[3]
    dimer_seqs = seq_info[1]
    isRC_flags = seq_info[-1]

    s_seqs = []
    t_seqs = []
    for i in range(len(dimer_seqs)):
        m = motif_encoder_decoder.motif_encoder(motif1_seq=motif1_seqs[i],
                                                motif2_seq=motif2_seqs[i], dimer_seq=dimer_seqs[i])
        source_sequence = m.motif_pair_code
        target_sequence = m.dimer_code

        # print(source_sequence.shape)
        # print(target_sequence.shape)
        s_seqs.append(source_sequence)
        t_seqs.append(target_sequence)

    s_seqs = np.array(s_seqs)
    t_seqs = np.array(t_seqs)
    # print(s_seqs.shape)
    # print(t_seqs.shape)
    return s_seqs, t_seqs

def family_cv():
    # dimerfamily_dict = pkl.load(open("JiecongData/dimerMotifFamily_dict.pkl", "rb"))
    # family_set = list(set(list(dimerfamily_dict.values())))
    res_dict = dict()
    kc_data = open("./JiecongData/kc_heterodimer_family.txt", "r")
    family_set = kc_data.readlines()
    for i in range(len(family_set)):
        f = family_set[i].strip()
        mean_dist = leave_one_validation(f)
        res_dict[family_set[i]] = mean_dist
    import pickle
    pickle.dump(res_dict, open("family_locv.pkl", "wb"))



def leave_one_validation(fname):
    from sklearn.model_selection import LeaveOneOut
    source_seqs, target_seqs = load_sequence_data(family_name=fname)
    loo = LeaveOneOut()
    loo.get_n_splits(source_seqs)
    i = 1
    # print(target_seqs.shape)
    all_dist = []
    for train_index, test_index in loo.split(source_seqs):
        s_train = source_seqs[train_index]
        t_train = target_seqs[train_index]
        s_test = source_seqs[test_index]
        t_test = target_seqs[test_index]

        m = s_train.shape[0]
        s0 = np.zeros((m, n_s))
        c0 = np.zeros((m, n_s))
        out0 = np.zeros((m, target_dim))
        outputs = list(t_train.swapaxes(0, 1))
        model.fit([s_train, s0, c0, out0], outputs, epochs=100, batch_size=1)

        preds = model.predict([s_test, s0, c0, out0])
        pred_dimer = np.array(preds).reshape((33, 6))

        end_idx = 0
        for l in np.argmax(preds, axis=-1):
            if l == 5:
                 break
            else:
                end_idx += 1
        pred_dimer = pred_dimer[1:end_idx, 1:-1]

        print("-"*20, "pred_dimer", "-"*20)
        print(pred_dimer)
        t_test = t_test[0]
        end_idx = np.arange(len(t_test))[t_test[:,-1] == 1][0]
        print(end_idx)
        true_dimer = t_test[1:end_idx, 1:-1]
        print("-" * 20, "true_dimer", "-" * 20)
        print(true_dimer)
        print(true_dimer.shape)
        avg_dist = motif_encoder_decoder.motif_encoder.mean_motif_column_dist(true_dimer=true_dimer, pred_dimer=pred_dimer)
        print(avg_dist)
        all_dist.append(avg_dist)
        with open("./res_full.txt", "a") as f:
            f.write("-"*20 + str(i) + "-"*20 + "\n"+ str(avg_dist) +
                    " true_len:" + str(len(true_dimer)) + " pred_len:" + str(len(pred_dimer)) +  "\n")
        i += 1
    print(all_dist)
    print(np.mean(np.array(all_dist)))
    return all_dist


def fold10_cv():
    from sklearn.model_selection import KFold
    source_seqs, target_seqs = load_sequence_data(family_name="all")
    kf = KFold(n_splits=10, random_state=66, shuffle=True)
    fold_dict = dict()
    fold_id = 0
    for train_index, test_index in kf.split(source_seqs):
        res_dist = []
        s_train = source_seqs[train_index]
        t_train = target_seqs[train_index]
        s_test = source_seqs[test_index]
        t_test = target_seqs[test_index]
        m = s_train.shape[0]
        s0 = np.zeros((m, n_s))
        c0 = np.zeros((m, n_s))
        out0 = np.zeros((m, target_dim))
        outputs = list(t_train.swapaxes(0, 1))
        model.fit([s_train, s0, c0, out0], outputs, epochs=300, batch_size=50)
        preds = model.predict([s_test, s0, c0, out0])
        pred_dimer = np.array(preds).swapaxes(0, 1)
        print(pred_dimer.shape)
        for i in range(len(pred_dimer)):
            pdimer = pred_dimer[i]
            tdimer = t_test[i]
            end_idx = 0
            for l in np.argmax(pdimer, axis=-1):
                if l == 5:
                    break
                else:
                    end_idx += 1
            pdimer = pdimer[1:end_idx, 1:-1]
            end_idx = np.arange(len(tdimer))[tdimer[:, -1] == 1][0]
            tdimer = tdimer[1:end_idx, 1:-1]
            avg_dist = motif_encoder_decoder.motif_encoder.mean_motif_column_dist(true_dimer=tdimer,                                                                                pred_dimer=pdimer)
            # print(avg_dist)
            res_dist.append(avg_dist)
        print(res_dist)
        fold_dict[fold_id] = res_dist
        fold_id += 1

    import pickle
    pickle.dump(fold_dict, open("10_fold_cv.pkl", "wb"))

# leave_one_validation()
# fold10_cv()
family_cv()
# leave_one_validation("HomeoCUT_Fox")