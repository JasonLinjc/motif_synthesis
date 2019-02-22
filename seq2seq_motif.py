# -*- coding: utf-8 -*-
# @Time     :2/18/19 5:16 PM
# @Auther   :Jason Lin
# @File     :seq2seq_motif.py
# @Software :PyCharm

import pickle as pkl
import numpy as np
import pandas as pd

# dimer motif max_length: 31
# two motif max length  : 32

motif_data = pkl.load(open("./dimer_motif_pair.pkl", "rb"))

# print(motif_data)
# print(len(motif_data))

def find_motif_length():
    max_dimer_len = 0
    max_motifpair_len = 0
    for motif in motif_data:
        dimer_dict = motif[0]
        motif1_dict = motif[1]
        motif2_dict = motif[2]
        # print(dimer_dict)
        dimer_name = list(dimer_dict.keys())[0]
        motif1_name = list(motif1_dict.keys())[0]
        motif2_name = list(motif2_dict.keys())[0]
        """
        print(dimer_name)
        print(dimer_dict[dimer_name])
        print(motif1_name)
        print(motif1_dict[motif1_name])
        print(motif2_name)
        print(motif2_dict[motif2_name])
        """
        d_len = len(dimer_dict[dimer_name])
        # print(d_len)
        m_len = len(motif1_dict[motif1_name]) + len(motif2_dict[motif2_name])
        print(d_len, m_len)
        if d_len > max_dimer_len:
            max_dimer_len = d_len
        if m_len > max_motifpair_len:
            max_motifpair_len = m_len

    print("dimer_len", max_dimer_len)
    print("motif_pair_len", max_motifpair_len)

def generate_input_motif_seq():
    max_dimer_len = 31
    max_motif_pair_len = 32
    data = []
    for motif in motif_data:
        dimer_dict = motif[0]
        motif1_dict = motif[1]
        motif2_dict = motif[2]
        dimer_family = motif[3]
        # print(dimer_dict)
        dimer_name = list(dimer_dict.keys())[0]
        motif1_name = list(motif1_dict.keys())[0]
        motif2_name = list(motif2_dict.keys())[0]
        motif_pair_name = motif1_name + ":" + motif2_name

        dimer_seq = dimer_dict[dimer_name]
        motif1_seq = motif1_dict[motif1_name]
        motif2_seq = motif2_dict[motif2_name]

        # print(dimer_seq)
        divied_vec = np.ones(4)
        if motif1_seq == " " or motif2_seq == " ":
            continue
        motif_pair_seq = np.concatenate((motif1_seq, [divied_vec], motif2_seq))
        # print(len(motif_pair_seq))
        motif_padding_len = max_motif_pair_len + 1 - len(motif_pair_seq)
        padding_vec = np.zeros((motif_padding_len, 4))
        motif_pair_seq = np.concatenate((motif_pair_seq, padding_vec))
        # print(motif_pair_seq.shape)
        # print(motif_pair_seq)

        dimer_padding_len = max_dimer_len - len(dimer_seq)
        padding_vec = np.zeros((dimer_padding_len, 4))
        dimer_seq = np.concatenate((dimer_seq, padding_vec))
        # print(dimer_seq.shape)
        # print(dimer_seq)
        data.append([motif_pair_name, motif_pair_seq, dimer_name, dimer_seq, dimer_family])
    return data

def get_motif_from_family(family_name = "bHLH_Homeo"):
    data = generate_input_motif_seq()
    motif = []
    x_train = []
    y_train = []
    for d in data:
        if d[-1] == "bHLH_Homeo":
            x_train.append(d[1].T)
            y_train.append(d[3].T)
            motif.append(d)

    # print(motif)
    return np.array(x_train), np.array(y_train)

def seq2seq_model():
    x_train, y_train = get_motif_from_family()
    print(x_train.shape)
    print(y_train.shape)
    


    pass



