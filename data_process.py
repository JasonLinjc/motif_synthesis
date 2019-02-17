# -*- coding: utf-8 -*-
# @Time     :2/17/19 3:12 PM
# @Auther   :Jason Lin
# @File     :data_process.py
# @Software :PyCharm
import re
import numpy as np

def load_motif(filename="./JiecongData/dimerMotifDatabase.txt"):
    dict = {}
    with open(filename, "r") as f:
        for line in f:
            hd_motif = re.split(" : | ", line.strip())
            # print(hd_motif)
            hd_name = hd_motif[0]
            hd_seq = hd_motif[1:]
            hd_seq = np.array([float(i) for i in hd_seq])
            print(hd_name)
            print(np.reshape(hd_seq, (-1, 4)))
            dict[hd_name] = np.reshape(hd_seq, (-1, 4))
    return dict


load_motif(filename="./motifDatabase.txt")