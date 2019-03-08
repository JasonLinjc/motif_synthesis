# -*- coding: utf-8 -*-
# @Time     :3/8/19 11:37 AM
# @Auther   :Jason Lin
# @File     :build_pwm.py
# @Software :PyCharm

import numpy as np
import pickle as pkl

def generate_input_motif_seq(family_name = "bHLH_Homeo"):
    data = []
    motif_data = pkl.load(open("./dimer_motif_pair.pkl", "rb"))
    # print(motif_data)
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
        if motif1_seq == " " or motif2_seq == " ":
            continue

        if dimer_family == family_name:
            # dimer_seq = dimer_seq.T
            np.savetxt("./pwm/" +dimer_name+".pwm", dimer_seq, delimiter="\t")


generate_input_motif_seq()