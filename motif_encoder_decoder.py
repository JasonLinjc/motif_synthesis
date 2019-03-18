# -*- coding: utf-8 -*-
# @Time     :3/18/19 8:31 PM
# @Auther   :Jason Lin
# @File     :motif_encoder_decoder.py
# @Software :PyCharm

import numpy as np
import pandas as pd

class motif_encoder:
    dimer_motif_data = pd.read_csv("./JiecongData/row_dimer_data.csv", index_col=[0])

    def __init__(self, motif1_seq, motif2_seq, dimer_seq):
        self.motif1_seq = motif1_seq
        self.motif2_seq = motif2_seq
        self.dimer_seq = dimer_seq




