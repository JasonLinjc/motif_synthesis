# -*- coding: utf-8 -*-
# @Time     :3/15/19 6:12 PM
# @Auther   :Jason Lin
# @File     :ResNet.py
# @Software :PyCharm

import keras
from keras import layers
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
import numpy as np
import os


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def ResNet(x_input, y_len, y_case, x_test):
    batch_size = 50
    epochs = 500
    bn_axis = 3
    inputs = Input(shape=(18, 18, 28))
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(inputs)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    # x = layers.Activation('relu')(x)
    # x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    # x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)

    len_output = layers.Dense(32, activation="softmax", name = "len_out")(x)
    case_output = layers.Dense(4, activation="softmax", name="case_out")(x)

    model = Model(inputs=inputs, outputs=[len_output, case_output], name='resnet')
    model.compile(optimizer='adam',
                  loss={
                      'len_out': 'mean_squared_logarithmic_error',
                      'case_out': 'mean_squared_logarithmic_error'},
                  loss_weights={
                      'len_out': 0.4,
                      'case_out': 0.6})

    print(model.summary())

    model.fit(x=x_input, y=[y_len, y_case], batch_size=batch_size, epochs=epochs)

    pred_len, pred_case = model.predict(x_test)
    # pred_len, pred_case = decode_predicted_results(pred_len, pred_case)
    return pred_len, pred_case


def one_cross_validation():
    from sklearn.metrics import roc_curve, auc
    from scipy import interp
    from sklearn.model_selection import KFold
    import motif_len_case
    from matplotlib.pylab import plt
    kf = KFold(n_splits=10, shuffle=True, random_state=16)
    mp_tensor, len_labels, case_labels = motif_len_case.generate_input_motif_seq()
    case_n_classes = 4
    tprs_4 = dict()
    aucs_4 = dict()
    mean_fpr = np.linspace(0, 1, 100)
    for i in range(case_n_classes):
        tprs_4[i] = []
        aucs_4[i] = []

    for train_idx, test_idx in kf.split(mp_tensor):
        x_train = mp_tensor[train_idx]
        ylen_train = len_labels[train_idx]
        ycase_train = case_labels[train_idx]

        x_test = mp_tensor[test_idx]
        ylen_test = len_labels[test_idx]
        ycase_test = case_labels[test_idx]

        pred_len, pred_case = ResNet(x_train, ylen_train, ycase_train, x_test)
        # print(pred_case)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        colors = ['aqua', 'darkorange', 'cornflowerblue', 'black']
        for i in range(case_n_classes):
            pred_case_i = pred_case[:,i]
            true_case_i = ycase_test[:,i]
            fpr[i], tpr[i], _ = roc_curve(true_case_i, pred_case_i)
            roc_auc[i] = auc(fpr[i], tpr[i])

            tprs_4[i].append(interp(mean_fpr, fpr[i], tpr[i]))
            tprs_4[i][-1][0] = 0.0
            aucs_4[i].append(roc_auc[i])

            plt.plot(fpr[i], tpr[i], color=colors[i],
                     label=r'Mean ROC Case %d (AUC = %0.3f )' % (i + 1, roc_auc[i]), lw=2, alpha=.8)

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

        break


one_cross_validation()