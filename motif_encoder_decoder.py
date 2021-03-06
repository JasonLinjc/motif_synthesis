# -*- coding: utf-8 -*-
# @Time     :3/18/19 8:31 PM
# @Auther   :Jason Lin
# @File     :motif_encoder_decoder.py
# @Software :PyCharm

import numpy as np
import pandas as pd
import pickle as pkl

class motif_encoder:
    dimer_motif_data = pd.read_csv("./JiecongData/row_dimer_data.csv", index_col=[0])
    single_motif_maxlen = 18
    dimer_maxlen = 31 + 2

    def __init__(self, motif1_seq, motif2_seq, dimer_seq):
        self.motif1_seq = motif1_seq
        self.motif2_seq = motif2_seq
        self.dimer_seq = dimer_seq
        self.encode_source_motif_pair()
        self.encode_target_dimer()

    def encode_source_motif_pair(self):
        motif_pair_code = np.zeros((self.single_motif_maxlen*2, 4)) + 0.25
        # motif_pair = np.concatenate((self.motif1_seq, self.motif2_seq), axis=0)
        for i in range(len(self.motif1_seq)):
            # code = np.zeros(self.max_pair_len * 4)
            motif_pair_code[i,] = self.motif1_seq[i]
            # pair_code.append(code)

        for i in range(len(self.motif2_seq)):
            motif_pair_code[i + self.single_motif_maxlen, ] = self.motif2_seq[i]
        self.motif_pair_code = motif_pair_code

    def encode_target_dimer(self):
        # Add two symbols : <GO> and <EOS>
        dimer_code = np.zeros((self.dimer_maxlen, 6))
        dimer_code[0] = np.array([1, 0, 0, 0, 0, 0])
        i = 0
        for n_code in self.dimer_seq:
            c = np.zeros((6))
            c[1:-1] = n_code
            dimer_code[i] = c
            i += 1
        dimer_code[i] = np.array([0, 0, 0, 0, 0, 1])
        self.dimer_code = dimer_code

    @classmethod
    def get_sequence_family_input(self, family_name = "bHLH_Homeo"):
        kc_dimer_info = pd.read_csv("./JiecongData/kc_dimer_info.csv")
        homomotif_seq_dict = pkl.load(open("./JiecongData/homodimerMotifDatabase_dict.pkl", "rb"))
        motif_seq_dict = pkl.load(open("JiecongData/motifDatabase_dict.pkl", "rb"))
        dimer_seq_dict = pkl.load(open("JiecongData/dimerMotifDatabase_dict.pkl", "rb"))
        dimerfamily_dict = pkl.load(open("JiecongData/dimerMotifFamily_dict.pkl", "rb"))
        dimer_list = []
        dimer_seqs = []
        motif1_seqs = []
        motif2_seqs = []
        motif_pair_seqs = []
        isRCs = []
        for name in dimerfamily_dict.keys():
            if dimerfamily_dict[name] == family_name:
                try:
                    dimer_seq = dimer_seq_dict[name]
                    isRC = kc_dimer_info[kc_dimer_info['nameOut'] == name]['isRC'].item()
                except:
                    continue
                motif1_name, motif2_name = name.split("_")[:2]
                try:
                    motif1_seq = motif_seq_dict[motif1_name]
                except:
                    motif1_seq = homomotif_seq_dict[motif1_name]
                try:
                    motif2_seq = motif_seq_dict[motif2_name]
                except:
                    motif2_seq = homomotif_seq_dict[motif2_name]

                if isRC == 1:
                    dimer_seq = get_rev_com_y(dimer_seq)
                    # continue

                motif_pair = np.concatenate((motif1_seq, motif2_seq))
                motif_pair_seqs.append(motif_pair)
                dimer_seqs.append(dimer_seq)
                motif1_seqs.append(motif1_seq)
                motif2_seqs.append(motif2_seq)
                dimer_list.append(name)
                isRCs.append(isRC)
        # print([dimer_list, dimer_seqs, motif1_seqs, motif2_seqs, motif_pair_seqs, isRCs])
        return [dimer_list, dimer_seqs, motif1_seqs, motif2_seqs, isRCs]

    @classmethod
    def get_all_dimer(self):
        dimer_list = []
        dimer_seqs = []
        motif1_seqs = []
        motif2_seqs = []
        isRCs = []
        family_name = []
        kc_dimer_info = pd.read_csv("./JiecongData/kc_dimer_info.csv")
        homomotif_seq_dict = pkl.load(open("./JiecongData/homodimerMotifDatabase_dict.pkl", "rb"))
        motif_seq_dict = pkl.load(open("JiecongData/motifDatabase_dict.pkl", "rb"))
        dimer_seq_dict = pkl.load(open("JiecongData/dimerMotifDatabase_dict.pkl", "rb"))
        dimerfamily_dict = pkl.load(open("JiecongData/dimerMotifFamily_dict.pkl", "rb"))
        for idx, d_info in kc_dimer_info.iterrows():
            olen = d_info['overlapLen']
            isRC = d_info['isRC']
            loc1 = d_info['loc1']
            loc2 = d_info['loc2']
            case = d_info['case']
            motif1_name = d_info['name1']
            motif2_name = d_info['name2']
            dimer_name = d_info['nameOut']
            dimer_seq = dimer_seq_dict[dimer_name]
            dimer_family = dimerfamily_dict[dimer_name]
            try:
                motif1_seq = motif_seq_dict[motif1_name]
            except:
                motif1_seq = homomotif_seq_dict[motif1_name]
            try:
                motif2_seq = motif_seq_dict[motif2_name]
            except:
                motif2_seq = homomotif_seq_dict[motif2_name]
            # if isRC == 1:
                # continue
            if isRC == 1:
                dimer_seq = get_rev_com_y(dimer_seq)
            dimer_seqs.append(dimer_seq)
            motif1_seqs.append(motif1_seq)
            motif2_seqs.append(motif2_seq)
            dimer_list.append(dimer_name)
            isRCs.append(isRC)
            family_name.append(dimer_family)
        return [dimer_list, dimer_seqs, motif1_seqs, motif2_seqs, isRCs, family_name]

    @classmethod
    def mean_motif_column_dist(self, true_dimer, pred_dimer):
        # print(true_dimer_code)
        true_len = len(true_dimer)
        pred_len = len(pred_dimer)
        # print("true_len:", true_len, "pred_len:", pred_len)
        # len_ = min([true_len, pred_len])
        dist_list = []
        # print(self.true_dimer_code)
        if true_len <= pred_len:
            len_ = true_len
            gap_len  =  pred_len  - true_len
            for i in range(gap_len + 1):
                tem_dist = 0
                for j in range(len_):
                    tem_dist += np.sqrt(sum((true_dimer[j] - pred_dimer[j+i]) ** 2))
                dist_list.append(tem_dist)
        else:
            len_ = pred_len
            gap_len = true_len - pred_len
            for i in range(gap_len + 1):
                tem_dist = 0
                for j in range(len_):
                    tem_dist += np.sqrt(sum((true_dimer[j+i] - pred_dimer[j]) ** 2))
                dist_list.append(tem_dist)
        dist_list = np.array(dist_list)
        res = dist_list/len_
        # print(res)
        return min(res)

def get_rev_com_y(seq_mat):

    # print(seq_mat)
    # print("-"*50)
    reversed_mat = seq_mat[::-1].copy()

    for i in range(len(reversed_mat)):
        reversed_mat[i] = np.concatenate((reversed_mat[i][:4][::-1], reversed_mat[i][4:]))
    return reversed_mat

def get_seq():

    kc_dimer_info = pd.read_csv("./JiecongData/kc_dimer_info.csv")
    homomotif_seq_dict = pkl.load(open("./JiecongData/homodimerMotifDatabase_dict.pkl", "rb"))
    motif_seq_dict = pkl.load(open("JiecongData/motifDatabase_dict.pkl", "rb"))
    dimer_seq_dict = pkl.load(open("JiecongData/dimerMotifDatabase_dict.pkl", "rb"))
    dimerfamily_dict = pkl.load(open("JiecongData/dimerMotifFamily_dict.pkl", "rb"))

    motif1_len = 0
    dimer_len = 0
    motif2_len = 0
    for idx, d_info in kc_dimer_info.iterrows():
        # if idx != 3:
        #     continue
        # print(d_info)
        olen = d_info['overlapLen']
        isRC = d_info['isRC']
        loc1 = d_info['loc1']
        loc2 = d_info['loc2']
        case = d_info['case']
        motif1_name = d_info['name1']
        motif2_name = d_info['name2']
        dimer_name = d_info['nameOut']
        dimer_seq = dimer_seq_dict[dimer_name]
        try:
            motif1_seq = motif_seq_dict[motif1_name]
        except:
            motif1_seq = homomotif_seq_dict[motif1_name]
        try:
            motif2_seq = motif_seq_dict[motif2_name]
        except:
            motif2_seq = homomotif_seq_dict[motif2_name]

        if len(motif1_seq) > motif1_len:
            motif1_len = len(motif1_seq)
        if len(motif2_seq) > motif2_len:
            motif2_len = len(motif2_seq)
        if len(dimer_seq) > dimer_len:
            dimer_len = len(dimer_seq)

        m = motif_encoder(motif1_seq, motif2_seq, dimer_seq)
        print(m.motif_pair_code)
        print(m.motif_pair_code.shape)
        print(m.dimer_code)
        print(m.dimer_code.shape)

        # break
    print(motif1_len, motif2_len, dimer_len)

# get_seq()
# d = motif_encoder.get_all_dimer()
# print(len(d[-1]))