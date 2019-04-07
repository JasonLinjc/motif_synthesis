# -*- coding: utf-8 -*-
# @Time     :4/2/19 8:49 PM
# @Auther   :Jason Lin
# @File     :test_new.py
# @Software :PyCharm
import motif_encoder_decoder
import numpy as np
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply, Reshape
from keras.layers import RepeatVector, Dense, Activation, Lambda, Embedding
import pickle as pkl
from attention_decoder import AttentionDecoder
from position_embedding import PositionEmbedding
from keras.models import Model


def load_sequence_data(family_name = "bHLH_Homeo"):
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


def encode_family_label(family_labels):
    import pickle as pkl
    fnames = pkl.load(open("./JiecongData/family_labels.pkl", "rb"))
    label_code = []
    for l in family_labels:
        code = np.zeros(len(fnames))
        code[fnames == l] = 1
        label_code.append(code)
    label_code = np.array(label_code)
    return label_code

def seq2seq_model():
    Tx = 18 + 18  # Source sequence max length
    Ty = 31 + 2  # Target sequence max length
    source_dim = 4
    target_dim = 6

    source_seq = Input(shape=(Tx, source_dim))
    target_seq = Input(shape=(Ty, target_dim))


    encoder_output = Bidirectional(LSTM(32, return_sequences=True))(source_seq)
    print(encoder_output)
    bahdanau_attention_decoder = AttentionDecoder(50, Ty)
    decoder_output = bahdanau_attention_decoder([encoder_output, target_seq])
    model = Model(inputs=[source_seq, target_seq], outputs=[decoder_output])

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adadelta')

    return  model


def training_test():
    from sklearn.model_selection import KFold
    source_seqs, target_seqs, fnames = load_sequence_data(family_name="all")
    kf = KFold(n_splits=10, random_state=66, shuffle=True)
    fold_dict = dict()
    fold_info_dict = dict()
    fold_id = 0
    family_labels = encode_family_label(fnames)



    for train_index, test_index in kf.split(source_seqs):
        print("-" * 10, fold_id, "-" * 10)
        res_dist = []
        s_train = source_seqs[train_index]
        t_train = target_seqs[train_index]
        s_test = source_seqs[test_index]
        t_test = target_seqs[test_index]
        t_fname = fnames[test_index]

        model = seq2seq_model()
        model.fit([s_train, t_train], t_train, epochs=5)
        # fam_train = family_labels[train_index]
        # fam_test = family_labels[test_index]

training_test()