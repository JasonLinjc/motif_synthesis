# -*- coding: utf-8 -*-
# @Time     :2/17/19 3:12 PM
# @Auther   :Jason Lin
# @File     :data_process.py
# @Software :PyCharm
import re
import numpy as np
import pandas as pd
import pickle as pkl
import re

def load_motif(filename="./JiecongData/dimerMotifDatabase.txt"):
    dict = {}
    fname = re.split("/|\.", filename)[-2]
    #  print(fname)
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
    dict_fname = "./JiecongData/" + fname + "_dict.pkl"
    pkl.dump(dict, open(dict_fname, "wb"))
    return dict

def load_dimer_family():
    dict = {}
    filename = "./JiecongData/dimerMotifFamily.txt"
    with open(filename, "r") as f:
        for line in f:
            hd_motif = line.split(" : ")
            name = hd_motif[0].strip()
            family = hd_motif[1].strip()
            dict[name] = family
    # print(dict)
    # print(len(dict))
    # print(len(set(list(dict.values()))))
    pkl.dump(dict, open("./JiecongData/dimerMotifFamily_dict.pkl", "wb"))
    return dict
# load_motif(filename="./JiecongData/homodimerMotifDatabase.txt")


def pair_motif_and_dimer():
    dimer_motif_dict = load_motif(filename="./JiecongData/dimerMotifDatabase.txt")
    dimer_family_dict = load_dimer_family()
    # motif_dict = load_motif(filename="./JiecongData/motifDatabase.txt")
    # homo_motif_dict = load_motif(filename="./JiecongData/homodimerMotifDatabase.txt")
    # print(motif)
    # print(dimer_motif)
    # motifs = np.array(list(motif_dict.keys()))
    # hmotifs = np.array(list(homo_motif_dict.keys()))
    # print(motifs)
    # print(hmotifs)
    motifp_dimer_list = []
    dmotifs = list(dimer_motif_dict.keys())
    no_found_motif = []


    for dimer in dmotifs:
        motif_l = dimer.split("_")
        # print(motif_l)
        motif1 = motif_l[0]
        motif2 = motif_l[1]
        print(motif1, motif2)
        motif1_seq_dict = {}
        motif2_seq_dict = {}
        dimer_seq_dict = {}

        name, seq = find_motif_seq(motif1)
        motif1_seq_dict[name] = seq
        if seq == " ":
            no_found_motif.append(name)
        name, seq = find_motif_seq(motif2)
        if seq == " ":
            no_found_motif.append(name)
        motif2_seq_dict[name] = seq
        dimer_seq_dict[dimer] = dimer_motif_dict[dimer]

        print(motif1_seq_dict)
        print(motif2_seq_dict)

        motifp_dimer_list.append([dimer_seq_dict, motif1_seq_dict, motif2_seq_dict, dimer_family_dict[dimer]])

    print(no_found_motif)
    # motif_dict = load_motif(filename="./JiecongData/motifDatabase.txt")
    # homo_motif_dict = load_motif(filename="./JiecongData/homodimerMotifDatabase.txt")
    # motifs = np.array(list(motif_dict.keys()))
    # hmotifs = np.array(list(homo_motif_dict.keys()))
    # print(motifs)
    # print(hmotifs)
    print(motifp_dimer_list)
    import pickle
    pickle.dump(motifp_dimer_list, open("./dimer_motif_pair.pkl",  "wb"))

def load_kc_dimer():
    file = "./JiecongData/kc_heterodimer_family.txt"
    f = open(file, "r")
    dimer = []
    for d in f.readlines():
        dimer.append(d.strip())
    return dimer

def find_motif_seq(motif):
    # dimer_motif_dict = load_motif(filename="./JiecongData/dimerMotifDatabase.txt")
    motif_dict = load_motif(filename="./JiecongData/motifDatabase.txt")
    homo_motif_dict = load_motif(filename="./JiecongData/homodimerMotifDatabase.txt")


    motifs = np.array(list(motif_dict.keys()))
    hmotifs = np.array(list(homo_motif_dict.keys()))
    # print(motifs)
    # print(hmotifs)

    idx = find_motif_idx(motif, motifs)
    if idx > 0:
        # print(motif, motif_dict[motif])
        return motif, motif_dict[motif]
    else:
        idx = find_motif_idx(motif, hmotifs)
        if idx > 0:
            # print(motif, homo_motif_dict[motif])
            return motif, homo_motif_dict[motif]
        else:
            # print("Motif not found!")
            return motif, " "

def find_motif_idx(dimer_motif, motif_database):
    # print(motif_database)
    idx_l = np.arange(len(motif_database))
    if dimer_motif in motif_database:
        idx = idx_l[motif_database == dimer_motif]
    else:
        idx = -1
    return idx



def find_best_matching_idx():
    kc_dimer_info = pd.read_csv("./JiecongData/kc_dimer_info.csv")
    homomotif_seq_dict = pkl.load(open("./JiecongData/homodimerMotifDatabase_dict.pkl", "rb"))
    motif_seq_dict = pkl.load(open("JiecongData/motifDatabase_dict.pkl", "rb"))
    dimer_seq_dict = pkl.load(open("JiecongData/dimerMotifDatabase_dict.pkl", "rb"))
    dimerfamily_dict = pkl.load(open("JiecongData/dimerMotifFamily_dict.pkl",  "rb"))



find_best_matching_idx()

# dimer_data = pd.read_csv("./JiecongData/row_dimer_data.csv", index_col=[0])
# dimer_data.to_csv("kc_dimer_info.csv", index=False)
# find_best_matching_idx()

# load_motif()
# motif_dict = load_motif(filename="./JiecongData/motifDatabase.txt")
# homo_motif_dict = load_motif(filename="./JiecongData/homodimerMotifDatabase.txt")

# pair_motif_and_dimer()

# dict = load_dimer_family()
#  print(len(dict))
#  fam = list(dict.values())
# print(fam)
"""
f_count = {}
fam_b = []
for f in fs:
    num = fam.count(f)
    if num > 3:
        fam_b.append(f)
    f_count[f] = fam.count(f)

print(f_count)
print(len(fam_b))
print(fam_b)

dimers = load_kc_dimer()
# print(dimers)
dimer_dict = {}
for d in dimers:
    dimer_dict[d] = fam.count(d)

print(dimer_dict)
"""
# name = "bHLH_Homeo"

