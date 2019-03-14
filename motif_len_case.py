# -*- coding: utf-8 -*-
# @Time     :3/13/19 5:02 PM
# @Auther   :Jason Lin
# @File     :motif_len_case.py
# @Software :PyCharm

import numpy as np
import pandas as pd
import pickle as pkl
from keras.models import Model
from sklearn.model_selection import LeaveOneOut
import motif_encoder
from keras.losses import categorical_crossentropy
from keras.models import Sequential
import matplotlib.pyplot as plt
from itertools import cycle
from keras.layers import Dense, Conv2D, Flatten, Input
from numpy.random import seed
from sklearn.model_selection import KFold
seed(6)
from tensorflow import set_random_seed
set_random_seed(66)

def generate_input_motif_seq(family_name = "bHLH_Homeo"):
    dimer_tensor = []
    len_labels = []
    case_labels = []
    labels = []
    over_len_types = np.arange(-16, 16)
    over_case_types = np.arange(1, 5)
    dimer_data = pd.read_csv("./JiecongData/row_dimer_data.csv", index_col=[0])
    motif_data = pkl.load(open("./dimer_motif_pair.pkl", "rb"))
    # print(dimer_data)
    # print(motif_data)
    for motif in motif_data:
        dimer_dict = motif[0]
        motif1_dict = motif[1]
        motif2_dict = motif[2]
        dimer_family = motif[3]

        dimer_name = list(dimer_dict.keys())[0]
        motif1_name = list(motif1_dict.keys())[0]
        motif2_name = list(motif2_dict.keys())[0]

        dimer_seq = dimer_dict[dimer_name]
        motif1_seq = motif1_dict[motif1_name]
        motif2_seq = motif2_dict[motif2_name]
        # print(dimer_seq)
        if motif1_seq == " " or motif2_seq == " ":
            continue

        # print(dimer_name)
        dimer_info = dimer_data[dimer_data['nameOut'] == dimer_name]

        over_len = dimer_info['overlapLen'].item()
        over_case = dimer_info['case'].item()
        # print("len: ", over_len, " case: ", over_case)
        len_label = np.zeros(over_len_types.shape)
        len_label[over_len_types == over_len] = 1.0
        case_label = np.zeros(over_case_types.shape)
        case_label[over_case_types == over_case] = 1.0

        # break
        m = motif_encoder.motif_pair_encoder(motif1_seq, motif2_seq, dimer_seq)
        mp_tensor = m.motif_pair2tensor()
        # print(mp_tensor)
        # print(mp_tensor.shape)
        dimer_tensor.append(mp_tensor)
        len_labels.append(len_label)
        case_labels.append(case_label)

    dimer_tensor = np.array(dimer_tensor)
    len_labels = np.array(len_labels)
    case_labels = np.array(case_labels)
    return dimer_tensor, len_labels, case_labels
    # if dimer_family == family_name:

# mp_tensor, len_labels, case_labels = generate_input_motif_seq()
# print(mp_tensor.shape)
# print(len_labels.shape)
# print(case_labels.shape)

def multi_task_CNN(x, y_len, y_case, x_test):
    batch_size = 20
    epochs = 200
    # create model
    inputs = Input(shape=(18, 18, 16))
    # add model layers
    conv_1 = Conv2D(128, kernel_size=1, activation='relu')(inputs)
    conv_2 = Conv2D(64, kernel_size=5, activation='relu')(conv_1)
    conv_3 = Conv2D(32, kernel_size=3, activation='relu')(conv_2)
    # conv_4 = Conv2D(1, kernel_size=1, activation='relu')(conv_1)
    flatten = Flatten()(conv_3)
    dense_1 = Dense(128, activation='relu')(flatten)
    dense_1 = Dense(64, activation='relu')(dense_1)

    len_output = Dense(32, activation='softmax', name="len_out")(dense_1)
    case_output = Dense(4, activation='softmax', name="case_out")(dense_1)

    model = Model(inputs=inputs, outputs=[len_output, case_output])
    model.compile(optimizer='adam',
                  loss={
                      'len_out': 'categorical_crossentropy',
                      'case_out': 'categorical_crossentropy'},
                  loss_weights={
                      'len_out': 0.3,
                      'case_out': 0.7})
    print(model.summary())

    model.fit(x=x, y=[y_len, y_case], batch_size = batch_size, epochs = epochs)

    pred_len, pred_case = model.predict(x_test)
    # pred_len, pred_case = decode_predicted_results(pred_len, pred_case)
    return pred_len, pred_case
    # return model

def decode_predicted_results(pred_len, pred_case):
    len_ = []
    case_ = []
    over_len_types = np.arange(-16, 16)
    over_case_types = np.arange(1, 5)
    for i in range(len(pred_len)):
        tlen = over_len_types[np.argmax(pred_len[i])]
        tcase = over_case_types[np.argmax(pred_case[i])]
        len_.append(tlen)
        case_.append(tcase)
    return np.array(len_), np.array(case_)


def fold10_cross_validation():
    from sklearn.metrics import roc_curve, auc
    from scipy import interp
    kf = KFold(n_splits=10)
    mp_tensor, len_labels, case_labels = generate_input_motif_seq()
    case_n_classes = 4
    for train_idx, test_idx in kf.split(mp_tensor):
        x_train = mp_tensor[train_idx]
        ylen_train = len_labels[train_idx]
        ycase_train = case_labels[train_idx]

        x_test = mp_tensor[test_idx]
        ylen_test = len_labels[test_idx]
        ycase_test = case_labels[test_idx]

        pred_len, pred_case = multi_task_CNN(x_train, ylen_train, ycase_train, x_test)
        print(pred_case)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(case_n_classes):
            pred_case_i = pred_case[:,i]
            true_case_i = ycase_test[:,i]
            fpr[i], tpr[i], _ = roc_curve(true_case_i, pred_case_i)
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure()
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue','black'])
        lw = 2
        for i, color in zip(range(case_n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                    label='ROC curve of case {0} (area = {1:0.2f})'
                        ''.format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
        break




"""
def multi_loss(len_true, len_pred, case_true, case_pred):
    lambda_ = 0.5
    return lambda_ * categorical_crossentropy(len_true, len_pred) + \
           (1.0 - lambda_) * categorical_crossentropy(case_true, case_pred)
"""

# mp_tensor, len_labels, case_labels = generate_input_motif_seq()
# pred_len, pred_case = multi_task_CNN(mp_tensor[:50], len_labels[:50], case_labels[:50], mp_tensor[-10:])


fold10_cross_validation()