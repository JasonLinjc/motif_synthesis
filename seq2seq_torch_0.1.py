# -*- coding: utf-8 -*-
# @Time     :4/8/19 4:58 PM
# @Auther   :Jason Lin
# @File     :seq2seq_torch_0.1.py
# @Software :PyCharm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import spacy

import random
import math
import os
import time
import motif_encoder_decoder
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_sequence_data(family_name = "all"):
    if family_name == "all":
        seq_info = motif_encoder_decoder.motif_encoder.get_all_dimer()
        fname = seq_info[5]
    else:
        seq_info = motif_encoder_decoder.motif_encoder.get_sequence_family_input(family_name)
    # print(seq_info)
    motif1_seqs = seq_info[2]
    motif2_seqs = seq_info[3]
    dimer_seqs = seq_info[1]
    isRC_flags = seq_info[4]

    s_seqs = []
    t_seqs = []
    for i in range(len(dimer_seqs)):
        m = motif_encoder_decoder.motif_encoder(motif1_seq=motif1_seqs[i],
                                                motif2_seq=motif2_seqs[i], dimer_seq=dimer_seqs[i])
        source_sequence = m.motif_pair_code
        target_sequence = m.dimer_code
        # print(source_sequence.shape)
        # print(target_sequence.shape)
        s_seqs.append(source_sequence)
        t_seqs.append(target_sequence)

    s_seqs = np.array(s_seqs)
    t_seqs = np.array(t_seqs)
    # print(s_seqs.shape)
    # print(t_seqs.shape)
    if family_name == "all":
        return s_seqs, t_seqs, np.array(fname)
    return s_seqs, t_seqs

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()

        self.input_dim = input_dim
        # self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout

        # self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(input_dim, enc_hid_dim, bidirectional=True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        # self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src sent len, batch size]

        # embedded = self.dropout(self.embedding(src))

        # embedded = [src sent len, batch size, emb dim]

        batch_size = src.size(0)
        src = src.view(-1, batch_size, self.input_dim)
        outputs, hidden = self.rnn(src)

        # outputs = [src sent len, batch size, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]

        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer

        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN

        # initial decoder hidden is final hidden state of the forwards and backwards encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        # outputs = [src sent len, batch size, enc hid dim * 2]
        # hidden = [batch size, dec hid dim]

        return outputs, hidden

    def initHidden(self, batch_size):
        return torch.zeros(2, batch_size, self.enc_hid_dim, device=device)


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        # self.attn = nn.Linear(96, 32)
        self.v = nn.Parameter(torch.rand(dec_hid_dim))

    def forward(self, hidden, encoder_outputs):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src sent len, batch size, enc hid dim * 2]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # repeat encoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # hidden = [batch size, src sent len, dec hid dim]
        # encoder_outputs = [batch size, src sent len, enc hid dim * 2]
        concat = torch.cat((hidden, encoder_outputs), dim=2)


        energy = torch.tanh(self.attn(concat))

        # energy = [batch size, src sent len, dec hid dim]

        energy = energy.permute(0, 2, 1)

        # energy = [batch size, dec hid dim, src sent len]

        # v = [dec hid dim]

        v = self.v.repeat(batch_size, 1).unsqueeze(1)

        # v = [batch size, 1, dec hid dim]

        attention = torch.bmm(v, energy).squeeze(1)

        # attention= [batch size, src len]

        return F.softmax(attention, dim=1)


source_seqs, target_seqs, fnames = load_sequence_data(family_name="all")
source_seqs, target_seqs = map(torch.tensor, (source_seqs, target_seqs))
print(target_seqs.size(), source_seqs.size())
source_seqs = source_seqs.to(device)
target_seqs = target_seqs.to(device)
train_ds = TensorDataset(source_seqs, target_seqs)
train_dl = DataLoader(train_ds, batch_size=10, shuffle=True)

encoder = Encoder(input_dim=4, emb_dim=16, enc_hid_dim=32, dec_hid_dim=32, dropout=0.1).to(device)
attention = Attention(enc_hid_dim=32, dec_hid_dim=32).to(device)
for source_seq, target_seq in train_dl:
    ########################Very Important#########################
    source_tensor = source_seq.float()
    target_tensor = target_seq.float()
    ###############################################################
    encoder_outputs, encoder_hidden = encoder(source_tensor)
    print(encoder_outputs.size(), encoder_hidden.size())
    atte_weight = attention(encoder_hidden, encoder_outputs)


    # src_len = encoder_outputs.size(0)
    # encoder_hidden = encoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
    # encoder_outputs = encoder_outputs.permute(1, 0, 2)
    # att = nn.Linear(96, 32).to(device)
    # t = torch.cat((encoder_hidden, encoder_outputs), dim=2)
    # print(t.size())
    # a = torch.tanh(att(t))
    print(atte_weight.size())
    break

