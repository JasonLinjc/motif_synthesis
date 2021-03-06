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
import pickle

SEED = 1

random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
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
    def __init__(self, input_dim, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.input_dim = input_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        # self.dropout = dropout


        self.rnn = nn.GRU(input_dim, enc_hid_dim, bidirectional=True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)


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

class Decoder(nn.Module):
    def __init__(self, output_dim, enc_hid_dim, dec_hid_dim, attention):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.attention = attention


        self.rnn = nn.GRU((enc_hid_dim * 2) + output_dim, dec_hid_dim)

        self.out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + output_dim, output_dim)


    def forward(self, input, hidden, encoder_outputs):
        # input = [batch size]
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src sent len, batch size, enc hid dim * 2]

        input = input.unsqueeze(0)

        # input = [1, batch size, enc_dim]

        # embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]

        a = self.attention(hidden, encoder_outputs)

        # a = [batch size, src len]

        a = a.unsqueeze(1)

        # a = [batch size, 1, src len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # encoder_outputs = [batch size, src sent len, enc hid dim * 2]

        weighted = torch.bmm(a, encoder_outputs)

        # weighted = [batch size, 1, enc hid dim * 2]

        weighted = weighted.permute(1, 0, 2)

        # weighted = [1, batch size, enc hid dim * 2]

        rnn_input = torch.cat((input, weighted), dim=2)

        # rnn_input = [1, batch size, (enc hid dim * 2) + enc_dim]

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        # output = [sent len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # sent len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        assert (output == hidden).all()

        embedded = input.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        output = F.softmax(self.out(torch.cat((output, weighted, embedded), dim=1)), dim =1)

        # output = [bsz, output dim]

        return output, hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [batch_size, src_seq_len, src_dim]
        # trg = [batch_size, trg_seq_len, trg_dim]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time


        batch_size = src.shape[0]
        max_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        output = torch.tensor([1., 0., 0., 0., 0., 0.]).repeat(batch_size).view(batch_size, -1).to(device)

        for t in range(max_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            output = (trg[:, t] if teacher_force else output)

        outputs = outputs.permute(1, 0, 2)
        # outputs = [trg_seq_len, batch_size, trg_dim]
        return outputs

def maskpad_Loss(pred_seqs, true_seqs):
    # seq = [batch_size, seq_len, seq_dim]
    batch_size = true_seqs.size(0)
    total_loss = 0

    for seq_idx in range(batch_size):
        seq_loss = 0
        for code_idx in range(len(true_seqs[seq_idx])):
            true_code = true_seqs[seq_idx, code_idx]
            pred_code = pred_seqs[seq_idx, code_idx]
            if true_code.argmax() == 5:
                seq_len = code_idx
                # print("len:" , seq_len)
                break
            seq_loss += torch.sum((pred_code - true_code) ** 2)
        seq_loss = seq_loss / seq_len
        total_loss += seq_loss

    return total_loss/batch_size


def train(seq2seq_model, criterion, train_dl):
    # BATCH_SIZE = 10
    # source_seqs, target_seqs, fnames = load_sequence_data(family_name="all")
    # source_seqs, target_seqs = map(torch.tensor, (source_seqs, target_seqs))
    # print(target_seqs.size(), source_seqs.size())
    # Set the device (cuda) and data type (float)
    # source_seqs = source_seqs.float().to(device)
    # target_seqs = target_seqs.float().to(device)
    # train_ds = TensorDataset(source_seqs, target_seqs)
    # train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # encoder = Encoder(input_dim=4, enc_hid_dim=32, dec_hid_dim=32, dropout=0.1).to(device)
    # attention = Attention(enc_hid_dim=32, dec_hid_dim=32).to(device)
    # decoder = Decoder(enc_hid_dim=32, dec_hid_dim=32, output_dim=6, attention=attention).to(device)
    # seq2seq_model = Seq2Seq(encoder, decoder, device)

    seq2seq_model.train()
    # criterion = nn.SmoothL1Loss()
    epoch_loss = 0

    optimizer = optim.Adam(seq2seq_model.parameters(), lr = 0.001, weight_decay=0.001)
    len_ = 0
    for source_seq, target_seq in train_dl:
        len_ += 1
        ########################Very Important#########################
        source_tensor = source_seq
        target_tensor = target_seq
        ###############################################################
        # ncoder_outputs, encoder_hidden = encoder(source_tensor)
        # print(encoder_outputs.size(), encoder_hidden.size())
        #
        # atte_weight = attention(encoder_hidden, encoder_outputs)
        # sos_input = torch.tensor([1., 0., 0., 0., 0., 0.]).repeat(BATCH_SIZE).view(BATCH_SIZE, -1).to(device)
        # t_trg = target_tensor[:,2]
        # print(t_trg.size())
        # decoder_output, decoder_hidden = decoder(sos_input, encoder_hidden, encoder_outputs)
        # decoder_output, decoder_hidden = decoder(decoder_output, encoder_hidden, encoder_outputs)optimizer = optim.Adam(model.parameters())

        # print(decoder_output.size())
        # print(source_tensor.size(), target_tensor.size())
        # output = seq2seq_model(source_tensor, target_tensor)
        # print(output.size())

        # src_len = encoder_outputs.size(0)
        # encoder_hidden = encoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        # encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # att = nn.Linear(96, 32).to(device)
        # t = torch.cat((encoder_hidden, encoder_outputs), dim=2)
        # print(t.size())
        # a = torch.tanh(att(t))
        # print(atte_weight.size())
        optimizer.zero_grad()

        predict_target = seq2seq_model(source_tensor, target_tensor)
        # predict_target = output.permute(1, 0, 2)
        # print(predict_target.size())
        # print(target_tensor.size())
        loss = criterion(predict_target, target_tensor)
        loss.backward()
        clip = 1
        torch.nn.utils.clip_grad_norm_(seq2seq_model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

        # print("MaskPad_loss:", maskpad_Loss(predict_target, target_tensor))


    return epoch_loss/len_


def evaluate(seq2seq_model, criterion, val_dl):
    seq2seq_model.eval()
    epoch_loss = 0
    with torch.no_grad():
        len_ = 0
        for src_tensor, trg_tensor in val_dl:
            len_ += 1
            # No use teacher forcing
            predict_target = seq2seq_model(src_tensor, trg_tensor, 0)
            # Transfer the shape of pred_trg_seq into [batch size, seq_len, seq_dim]
            # predict_target = output.permute(1, 0, 2)
            loss = criterion(predict_target, trg_tensor)
            epoch_loss += loss.item()

    return epoch_loss/len_


def predict(seq2seq_model, test_dl):
    seq2seq_model.eval()
    with torch.no_grad():
        for src_tensor, trg_tensor in test_dl:
            # No use teacher forcing
            output = seq2seq_model(src_tensor, trg_tensor, 0)
            pred_seqs =  output.to("cpu").numpy()
            true_seqs = trg_tensor.to("cpu").numpy()

    pseqs = []
    tseqs = []
    for i in range(len(pred_seqs)):
        ps = []
        ts = []
        for pred_code in pred_seqs[i]:
            if pred_code.argmax() == 5:
                break
            ps.append(pred_code[1:-1])
        ps = np.array(ps)
        pseqs.append(ps)
        for true_code in true_seqs[i]:
            if true_code.argmax() == 5:
                break
            ts.append(true_code[1:-1])
        ts = np.array(ts)
        tseqs.append(ts)

    return pseqs, tseqs

def mean_motif_column_dist(pred_seqs, true_seqs):
    dist_list = []
    for i in range(len(pred_seqs)):
        p_seq = pred_seqs[i]
        t_seq = true_seqs[i]
        avg_dist = motif_encoder_decoder.motif_encoder.mean_motif_column_dist(true_dimer=t_seq, pred_dimer=p_seq)
        dist_list.append(avg_dist)
    return dist_list


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 400
BATCH_SIZE = 50
SAVE_DIR = 'models'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'seq2seq_att_model.pt')

best_valid_loss = float('inf')

if not os.path.isdir(f'{SAVE_DIR}'):
    os.makedirs(f'{SAVE_DIR}')

from sklearn.model_selection import KFold
source_seqs, target_seqs, fnames = load_sequence_data(family_name="all")
kf = KFold(n_splits=10, random_state=66, shuffle=True)
fold_id = 0
fold_infos = dict()

for train_index, val_index in kf.split(source_seqs):
    print("-" * 10, fold_id, "-" * 10)
    src_train = source_seqs[train_index]
    trg_train = target_seqs[train_index]
    src_val = source_seqs[val_index]
    trg_val = target_seqs[val_index]
    # Training data
    src_train, trg_train = map(torch.tensor, (src_train, trg_train))
    # print(src_train.size(), trg_train.size())
    src_train = src_train.float().to(device)
    trg_train = trg_train.float().to(device)
    train_ds = TensorDataset(src_train, trg_train)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE)
    # Validating data
    val_batch_size = len(src_val)
    src_val, trg_val = map(torch.tensor, (src_val, trg_val))
    src_val = src_val.float().to(device)
    trg_val = trg_val.float().to(device)
    val_ds = TensorDataset(src_val, trg_val)
    val_dl = DataLoader(train_ds, batch_size=val_batch_size)

    encoder = Encoder(input_dim=4, enc_hid_dim=32, dec_hid_dim=32).to(device)
    attention = Attention(enc_hid_dim=32, dec_hid_dim=32).to(device)
    decoder = Decoder(enc_hid_dim=32, dec_hid_dim=32, output_dim=6, attention=attention).to(device)
    seq2seq_model = Seq2Seq(encoder, decoder, device)

    # Training and validating seq2seq model
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        criterion = maskpad_Loss
        train_loss = train(seq2seq_model, criterion, train_dl)
        valid_loss = evaluate(seq2seq_model, criterion, val_dl)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    # Tesing on validation seqs
    pred_seqs, true_seqs = predict(seq2seq_model, val_dl)
    mmc_dist = mean_motif_column_dist(pred_seqs, true_seqs)
    # print(len(pred_seqs), len(true_seqs), len(mmc_dist))
    fold_infos[fold_id] = [pred_seqs, true_seqs, mmc_dist]
    fold_id  += 1

    # print(pred_seqs)
    # print(true_seqs)
    # pred_info.append([pdimer, tdimer])

pickle.dump(fold_infos, open("10_fold_newtorch.pkl", "wb"))

    # fold_info_dict[fold_id] = [pred_info, t_fname]















