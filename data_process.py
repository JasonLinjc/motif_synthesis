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
            hd_name = hd_motif[0]
            hd_seq = hd_motif[1:]
            hd_seq = np.array([float(i) for i in hd_seq])
            # print(hd_name)
            # print(np.reshape(hd_seq, (4, -1)))
            dict[hd_name] = np.reshape(hd_seq, (-1, 4))
            # print(dict[hd_name].shape)
    return dict

# load_motif(filename="./JiecongData/homodimerMotifDatabase.txt")

def pair_motif_and_dimer():
    dimer_motif = load_motif(filename="./JiecongData/dimerMotifDatabase.txt")
    motif = load_motif(filename="./JiecongData/motifDatabase.txt")
    homo_motif = load_motif(filename="./JiecongData/homodimerMotifDatabase.txt")
    # print(motif)
    # print(dimer_motif)
    motifs = list(dimer_motif.keys())
    for dimer in motifs:
        motif_l = dimer.split("_")
        print(motif_l)
        motif1 = motif_l[0]
        motif2 = motif_l[1]
        


pair_motif_and_dimer()

