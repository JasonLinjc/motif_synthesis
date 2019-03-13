# -*- coding: utf-8 -*-
# @Time     :3/13/19 5:02 PM
# @Auther   :Jason Lin
# @File     :motif_len_case.py
# @Software :PyCharm

import numpy as np
import pandas as pd
import pickle as pkl
import tensorflow as tf
from sklearn.model_selection import LeaveOneOut
import motif_encoder


def generate_input_motif_seq(family_name = "bHLH_Homeo"):
    dimer_tensor = []
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

        print(dimer_name)
        dimer_info = dimer_data[dimer_data['nameOut'] == dimer_name]

        over_len = dimer_info['overlapLen'].item()
        over_case = dimer_info['case'].item()
        print("len: ", over_len, " case: ", over_case)
        # break
        m = motif_encoder.motif_pair_encoder(motif1_seq, motif2_seq, dimer_seq)
        mp_tensor = m.motif_pair2tensor()
        print(mp_tensor)
        print(mp_tensor.shape)
        break
        # if dimer_family == family_name:

generate_input_motif_seq()








