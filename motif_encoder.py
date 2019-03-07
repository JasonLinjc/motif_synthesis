# -*- coding: utf-8 -*-
# @Time    : 6/3/2019 1:04 AM
# @Author  : Jason Lin
# @File    : motif_encoder.py
# @Software: PyCharm

import numpy as np
import pandas as pd
import pickle as pkl
import seq2seq_MT

class motif_pair_encoder:
    max_pair_len = 32
    max_dimer_len = 31
    def __init__(self, motif1_seq, motif2_seq, dimer_seq):
        self.motif1_seq = motif1_seq
        self.motif2_seq = motif2_seq
        self.dimer_seq = dimer_seq
        self.encode_motif_pair()
        self.encode_dimer()

    def encode_motif_pair(self):
        pair_code  =  np.zeros((self.max_pair_len, self.max_pair_len * 4))
        motif_pair = np.concatenate((self.motif1_seq, self.motif2_seq), axis=0)
        for i in range(len(motif_pair)):
            # code = np.zeros(self.max_pair_len * 4)
            start_idx = i * 4
            end_idx = start_idx + 4
            pair_code[i, start_idx:end_idx] = motif_pair[i]
            # pair_code.append(code)
        self.motif_pair_code = pair_code

    def encode_dimer(self):
        dimer_code = np.zeros((self.max_dimer_len, self.max_dimer_len * 4 + 2))
        for i in range(len(self.dimer_seq)):
            start_idx = i * 4
            end_idx = start_idx + 4
            dimer_code[i, start_idx+1:end_idx+1] = self.dimer_seq[i]

        dimer_code = np.array(dimer_code)
        start_code = np.zeros(self.max_dimer_len * 4 + 2)
        start_code[0] = 1.
        end_code = np.zeros(self.max_dimer_len * 4 + 2)
        end_code[-1] = 1.
        dimer_input = np.concatenate(([start_code], dimer_code, [end_code]), axis=0)
        dimer_target = np.concatenate((dimer_code, [end_code], [np.zeros(self.max_dimer_len * 4 + 2)]), axis=0)
        self.dimer_input_code = dimer_input
        self.dimer_target_code = dimer_target


class pred_dimer_decoder:
    dimer_code_len = 126
    def __init__(self, pred_dimer_code):
        self.pred_dimer_code = pred_dimer_code
        self.decode_dimer_code()

    def decode_dimer_code(self):
        four_dim_dimer = []
        for i in range(len(self.pred_dimer_code)-1):
            start_idx = i*4+1
            end_idx = i*4+5
            code = self.pred_dimer_code[i,0, start_idx:end_idx].copy()
            # code = code/sum(code)
            four_dim_dimer.append(code)
        self.four_dim_dimer = np.array(four_dim_dimer)

    def decode_dimer_code_2(self):
        pass

    def mean_motif_column_dist(self, true_dimer):
        self.true_dimer_code = true_dimer[0]
        print(self.true_dimer_code)
        # print(true_dimer_code)
        true_len = len(self.true_dimer_code)
        pred_len = len(self.four_dim_dimer)
        print("true_len:", true_len, "pred_len:", pred_len)
        len_ = min([true_len, pred_len])
        sum_dist = 0
        # print(self.true_dimer_code)
        for i in range(len_):
            sum_dist += np.sqrt(sum((self.true_dimer_code[i] - self.four_dim_dimer[i]) ** 2))
        res = sum_dist / (len_)
        return res

"""
f = open("test1.out","r")
dimer = f.read()
dimer_code = np.loadtxt("test1.out", delimiter=',')
dimer_code = dimer_code.reshape((dimer_code.shape[0], 1, dimer_code.shape[-1]))
print(dimer_code.shape)

p = pred_dimer_decoder(dimer_code)
print(p.four_dim_dimer)
# print(np.sum(p.four_dim_dimer, axis=1))


encoder_input, decoder_input, decoder_target = seq2seq_MT.get_motif_from_family()
# print(encoder_input.shape)
# print(decoder_input.shape)
# print(decoder_target.shape)
enc_in_train = encoder_input[:-1]
dec_in_train = decoder_input[:-1]
dec_tar_train = decoder_target[:-1]

enc_in_val = np.array(encoder_input[-1:])
dec_in_val = np.array(decoder_input[-1:])
dec_tar_val = np.array(decoder_target[-1:])

# print(dec_tar_val.shape)
dist = p.mean_motif_column_dist(dec_tar_val)
print(dist)

"""


"""
motif_data = pkl.load(open("./dimer_motif_pair.pkl", "rb"))

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

    # print(motif1_seq)
    m = motif_pair_encoder(motif1_seq, motif2_seq, dimer_seq)
    print(m.motif_pair_code.shape)
    print(m.dimer_input_code.shape)
    # print(m.dimer_input_code)
    # print(m.dimer_target_code)
    # print(m.motif_pair_code)

"""











