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
from sklearn.model_selection import LeaveOneOut
from numpy.random import seed
seed(6)
from tensorflow import set_random_seed
set_random_seed(666)

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

def mean_motif_column_dist(pred_seq, true_seq):
    sum_dist = 0
    rev_sum_dist = 0
    print(pred_seq)
    print(true_seq)
    # true_seq = np.reshape(true_seq, (true_seq.shape[1], 1, 6))
    print("pred_shape:",  pred_seq.shape, " true_shape:", true_seq.shape)
    pred_stop_idx = len(pred_seq) - 2
    true_stop_idx = min(np.arange(len(true_seq[0]))[true_seq[0,:,-1] == 1]) - 1
    print(pred_stop_idx, true_stop_idx)
    idx = min([pred_stop_idx, true_stop_idx])
    # print(len_)
    for i in range(idx + 1):
        sum_dist += np.sqrt(sum((true_seq[0,i,:4] - pred_seq[i,0,:4])**2))
    res = sum_dist/(idx + 1)
    print(res)
    # rev_true_seq = np.array([true_seq[0][::-1]])
    for i in range(idx+1):
        rev_sum_dist += np.sqrt(sum((true_seq[0,idx-i,:4] - pred_seq[i,0,:4])**2))

    rev_res = rev_sum_dist/(idx+1)

    return min([rev_res, res])

    # reverse_true_seq = true_seq[::-1]

def leave_one_validation():
    encoder_input, decoder_input, decoder_target = get_motif_from_family()
    print(encoder_input.shape)
    print(decoder_input.shape)
    print(decoder_target.shape)
    dist_res = []
    loo = LeaveOneOut()
    loo.get_n_splits(encoder_input)

    for train_index, test_index in loo.split(encoder_input):
        enc_in_train = encoder_input[train_index]
        dec_in_train = decoder_input[train_index]
        dec_tar_train = decoder_target[train_index]

        enc_in_test = encoder_input[test_index]
        dec_in_test = decoder_input[test_index]
        dec_tar_test = decoder_target[test_index]

        dec_out = seq2seq_mt_model(enc_in_train, dec_in_train, dec_tar_train, enc_in_test)
        res = mean_motif_column_dist(dec_out, dec_tar_test)
        dist_res.append(res)
        break
    dist_res = np.array(dist_res)
    print(np.mean(dist_res), np.std(dist_res))
    print(dist_res)



def seq2seq_mt_model(encoder_input_data, decoder_input_data, decoder_target_data, test_data):
    # define an input sequence and process it
    batch_size = 1
    epochs = 200
    latent_dim = 512
    input_dim = 4
    target_dim = 6
    encoder_inputs = Input(shape=(None, input_dim))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_ouputs, state_h, state_c = encoder(encoder_inputs)

    encoder_states = [state_h, state_c]

    # Set up the decoder, using 'encoder_states' as initial state
    decoder_inputs = Input(shape=(None, target_dim))

    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(target_dim, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    #  Define the model that will turn
    #  encoder_input_data & decoder_input_data into decoder_target_data
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.001), loss='categorical_crossentropy')
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=batch_size,
              epochs=epochs)

    # Define sampling models
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    # decode sequence
    # Encode the input as state vectors
    input_seq = test_data
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1
    target_seq = np.zeros((1, 1, target_dim))

    # Populate the first character of target sequence with the start character
    target_seq[0, 0, -2] = 1

    stop_condition = False
    decoded_seq_code = np.zeros((1,1,6))

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        # print(output_tokens.shape)
        # print(output_tokens)
        decoded_seq_code = np.concatenate((decoded_seq_code, output_tokens))
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, 0, :])
        # Exit condition: either hit max length or find stop character.
        if (sampled_token_index == 5 or len(decoded_seq_code) > 33):
            stop_condition = True

        target_seq = output_tokens
        states_value = [h, c]

    # print(decoded_seq_code)
    print(decoded_seq_code.shape)
    return decoded_seq_code[1:]
"""
encoder_input, decoder_input, decoder_target = get_motif_from_family()
# print(encoder_input.shape)
# print(decoder_input.shape)
# print(decoder_target.shape)
enc_in_train = encoder_input[:-1]
dec_in_train = decoder_input[:-1]
dec_tar_train = decoder_target[:-1]

enc_in_val = np.array(encoder_input[-1:])
dec_in_val = np.array(decoder_input[-1:])
dec_tar_val = np.array(decoder_target[-1:])

# print(dec_tar_val.shape)
# dec_out = seq2seq_mt_model(enc_in_train, dec_in_train, dec_tar_train, enc_in_val)
# mean_motif_column_dist(dec_out, dec_tar_val)

leave_one_validation()


"""
