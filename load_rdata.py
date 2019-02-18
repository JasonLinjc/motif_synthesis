# -*- coding: utf-8 -*-
# @Time     :2/17/19 1:42 PM
# @Auther   :Jason Lin
# @File     :load_rdata.py
# @Software :PyCharm

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import numpy as np
from rpy2.robjects import numpy2ri
importr('hash')

def build_dimermotif():
    robjects.r['load']("./JiecongData/DimerMotifDatabase.RData")
    dimer = str(robjects.r['DimerMotifDatabase'])
    family = str(robjects.r['DimerMotifFamily'])
    with open("./dimerMotifDatabase.txt", "a") as f:
        f.write(dimer)
    with open("./dimerMotifFamily.txt", "a") as f:
        f.write(family)

def build_motif():
    robjects.r['load']("./JiecongData/motifDatabase.RData")
    motif = str(robjects.r['MotifDatabase'])
    with open("./motifDatabase.txt", "a") as f:
        f.write(motif)

def build_homo_motif():
    robjects.r['load']("./JiecongData/HomodimerMotifDatabase.RData")
    motif = str(robjects.r['HomodimerMotifDatabase'])
    with open("./homodimerMotifDatabase.txt", "a") as f:
        f.write(motif)


# build_homo_motif()
build_dimermotif()