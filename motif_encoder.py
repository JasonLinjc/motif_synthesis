# -*- coding: utf-8 -*-
# @Time    : 6/3/2019 1:04 AM
# @Author  : Jason Lin
# @File    : motif_encoder.py
# @Software: PyCharm

import numpy as np
import pandas as pd
import pickle as pkl

class motif_pair_encoder:
    max_pair_len = 31

    def __init__(self, motif1_seq, motif2_seq):
        self.motif1_seq = motif1_seq
        self.motif2_seq = motif2_seq
        self.encode_motif_pair()

    def encode_motif_pair(self):
        pair_code  =  []
        motif_pair = np.concatenate((motif1_seq, motif2_seq), axis=0)
        for i in range(len(motif_pair)):
            code = np.zeros(self.max_pair_len * 4)
            start_idx = i * 4
            end_idx = start_idx + 4
            code[start_idx, end_idx] = motif_pair[i]
            pair_code.append(code)
        pair_code = np.array(pair_code)
        self.motif_pair_code = pair_code


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

    print(motif1_seq)

    break





