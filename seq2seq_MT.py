# -*- coding: utf-8 -*-
# @Time     :2/28/19 11:14 AM
# @Auther   :Jason Lin
# @File     :seq2seq_MT.py
# @Software :PyCharm

import pickle as pkl
import numpy as np
import pandas as pd
import os, sys, time, random
import h5py
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.optimizers import Adam

def generate_input_motif_seq():
    max_dimer_len = 31 + 2
    max_motif_pair_len = 32
    data = []
    motif_data = pkl.load(open("./dimer_motif_pair.pkl", "rb"))
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
        divied_vec = np.ones(4)
        if motif1_seq == " " or motif2_seq == " ":
            continue
        motif_pair_seq = np.concatenate((motif1_seq, [divied_vec], motif2_seq))
        # print(len(motif_pair_seq))
        motif_padding_len = max_motif_pair_len + 1 - len(motif_pair_seq)
        padding_vec = np.zeros((motif_padding_len, 4))
        motif_pair_seq = np.concatenate((motif_pair_seq, padding_vec))
        # print(motif_pair_seq.shape)
        # print(motif_pair_seq)
        start_code = np.array([[0.,0.,0.,0.,1.,0.]])
        end_code = np.array([[0.,0.,0.,0.,0.,1.]])
        dimer_input = np.concatenate((dimer_seq, np.zeros((len(dimer_seq), 2))), axis=1)
        # print(dimer_input.shape)
        # print(end_code.shape)
        dimer_target = np.concatenate((dimer_input, end_code), axis=0)
        dimer_input = np.concatenate((start_code, dimer_input, end_code), axis=0)

        dimer_padding_len = max_dimer_len - len(dimer_target)
        padding_vec = np.zeros((dimer_padding_len, 6))
        dimer_target = np.concatenate((dimer_target, padding_vec))

        dimer_padding_len = max_dimer_len - len(dimer_input)
        padding_vec = np.zeros((dimer_padding_len, 6))
        dimer_input = np.concatenate((dimer_input, padding_vec))
        # print(dimer_seq.shape)
        # print(dimer_seq)
        data.append([motif_pair_name, motif_pair_seq, dimer_name, dimer_input, dimer_target, dimer_family])
    return data

def get_motif_from_family(family_name = "bHLH_Homeo"):
    # data = generate_input_motif_seq()
    data = generate_input_motif_seq()
    motif = []
    encoder_input = []
    decoder_input = []
    decoder_target = []
    for d in data:
        if d[-1] == "bHLH_Homeo":
            encoder_input.append(d[1])
            decoder_input.append(d[3])
            decoder_target.append(d[4])
            motif.append(d)

    # print(motif)
    return np.array(encoder_input), np.array(decoder_input), np.array(decoder_target)


def seq2seq_mt_model(encoder_input_data, decoder_input_data, decoder_target_data):
    # define an input sequence and process it
    batch_size = 1
    epochs = 200
    latent_dim = 128
    input_dim = 4
    target_dim = 6
    encoder_inputs = Input(shape=(None, input_dim))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_ouputs, state_h, state_c = encoder(encoder_inputs)

    encoder_states = [state_h, state_c]

    # Set up the decoder, using 'encoder_states' as initial state
    decoder_inputs = Input(shape=(None, input_dim))

    decoder_lstm = LSTM(latent_dim, return_squence=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(input_dim, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    #  Define the model that will turn
    #  encoder_input_data & decoder_input_data into decoder_target_data
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.001), loss='categorical_crossentropy')
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2)



data = generate_input_motif_seq()
print(data[0])



