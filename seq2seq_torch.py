# -*- coding: utf-8 -*-
# @Time     :3/25/19 8:59 PM
# @Auther   :Jason Lin
# @File     :seq2seq_torch.py
# @Software :PyCharm

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import motif_encoder_decoder
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 36
# print(torch.cuda.is_available())

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, input, hidden):
        # embedded = self.embedding(input).view(1, 1, -1)
        # output = embedded
        input = input.view(1, 1, -1)
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, decoder_input_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.decoder_input_size = decoder_input_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.attn = nn.Linear(self.hidden_size + self.output_size, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size + self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        # embedded = self.embedding(input).view(1, 1, -1)
        # embedded = self.dropout(embedded)
        input = input.view(1, 1, -1)

        attn_weights = F.softmax(
            self.attn(torch.cat((input[0], hidden[0]), 1)), dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(0),encoder_outputs.unsqueeze(0))

        output = torch.cat((input[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


teacher_forcing_ratio = 0.5


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


def trainEpoch(epochs=100):

    learning_rate = 0.001
    encoder = EncoderRNN(4, 32).to(device)
    decoder = AttnDecoderRNN(32, 32, 6, dropout_p=0.1).to(device)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=0.001)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=0.001)
    criterion = nn.SmoothL1Loss()
    # criterion = nn.NLLLoss()

    source_seqs, target_seqs, fnames = load_sequence_data(family_name="all")
    source_seqs, target_seqs = map(torch.tensor, (source_seqs, target_seqs))
    print(target_seqs.size(), source_seqs.size())
    source_seqs = source_seqs.to(device)
    target_seqs = target_seqs.to(device)

    train_ds = TensorDataset(source_seqs, target_seqs)

    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)

    for epoch in range(100):
        loss_total = 0
        # for i in range(len(train_ds)):
        for input_tensor, target_tensor in train_dl:
            input_tensor = input_tensor.float().squeeze()
            target_tensor = target_tensor.float().squeeze()

            input_max_length = 36

            input_length = input_tensor.size(0)
            target_length = target_tensor.size(0)

            encoder_outputs = torch.zeros(input_max_length, encoder.hidden_size, device=device)
            encoder_hidden = encoder.initHidden()
            loss = 0

            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
                encoder_outputs[ei] = encoder_output[0, 0]

            # print(encoder_outputs.size())

            decoder_input = torch.tensor([[1.,0.,0.,0.,0.,0.]], device = device)
            decoder_hidden = encoder_hidden
            """
            # Testing process
            # decoder = AttnDecoderRNN(32, 32, 6, dropout_p=0.1).to(device)
            di = decoder_input.view(1,1,-1)
            print(decoder_hidden.size(), di.size())
            li = nn.Linear(32 + 6, MAX_LENGTH).to(device)
            concat = torch.cat((di[0], decoder_hidden[0]), 1)
            attn_weights = F.softmax(li(concat), dim=1)
            print(attn_weights.size())
            attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
            print(attn_applied.size())
            print(decoder_input.size(), decoder_hidden.size(), encoder_outputs.size())
            # print(decoder_hidden)
            """
            # use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            use_teacher_forcing = True
            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                for di in range(target_length):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                    decoder_output = decoder_output.view(-1)
                    target_di_tensor = target_tensor[di].view(1, -1)
                    loss += criterion(decoder_output, target_di_tensor)
                    decoder_input = target_tensor[di]  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                for di in range(target_length):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                    decoder_input = decoder_output  # detach from history as input
                    loss += criterion(decoder_output, target_tensor[di])
                    if decoder_input.argmax() == 5:
                        break

            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            loss_total += loss.item()/target_length

        print(epoch, loss_total)

# def predict():





trainEpoch()

