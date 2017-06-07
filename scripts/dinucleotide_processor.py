import os
import sys
import numpy as np
import copy
import matplotlib.pyplot as plt
import pandas as pd
import time
import pickle
import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.nnet.nnet import sigmoid, softmax, relu, binary_crossentropy, categorical_crossentropy
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStream
from theano.compile.nanguardmode import NanGuardMode
from theano.compile.debugmode import DebugMode
from collections import OrderedDict
from sklearn.metrics import auc, log_loss, precision_recall_curve, roc_auc_score, roc_curve
from prg.prg import create_prg_curve, calc_auprg
from theano import pp
from sklearn.neural_network import MLPClassifier
from sklearn.grid_search import ParameterGrid
from sklearn.preprocessing import scale, StandardScaler 
np.random.seed(42)
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from keras.models import Sequential
from keras.engine.training import Model
from keras.callbacks import EarlyStopping
from keras.layers.core import (
    Activation, Dense, Dropout, Flatten,
    Permute, Reshape
)
from keras.layers.merge import Add
from keras.layers import Input, Dense, Flatten, merge
from keras.engine.topology import Layer, Container
from keras import activations, regularizers, constraints
import keras
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import GRU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2, l1_l2
from keras.utils import np_utils
from keras import backend as K
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
from scipy.stats import norm
plt.style.use('seaborn-whitegrid')
from common_utils import ClassificationResult
from common_utils import BinaryClassProcessor
from keras.models import model_from_json

def process_th_dinuc_tensor(X):
    bn = X.shape[0]
    nucs = X.shape[2]
    width = X.shape[3]

    W = np.zeros((bn, 1, 16, width-1))
    for b in range(bn):
        for n in range(width-1):
            i = np.argmax(X[b,0,:,n])
            j = np.argmax(X[b,0,:,n+1])
            mat = np.zeros((4,4))
            mat[i,j] = 1
            vec = mat.flatten()
            W[b,0,:,n] = vec
    return W

def process_th_nth_tensor(X):
    pass

def process_tf_nuc_tensor(X):
    bn = X.shape[0]
    nucs = X.shape[1]
    width = X.shape[2]

    W = np.zeros((bn, 16, width-1, 1))
    for b in range(bn):
        for n in range(width-1):
            i = np.argmax(X[b,:,n,0])
            j = np.argmax(X[b,:,n+1,0])
            mat = np.zeros((4,4))
            mat[i,j] = 1
            vec = mat.flatten()
            W[b,:,n,0] = vec
    return W

class AA_processor(object):
    def __init__(self):
        pass

    def process_theano_tensor(self):
        pass

    def process_tf_tensor(self):
        pass

class SimpleDiCNNModel(object):
    def __init__(self, X, X_RC, debug=False):
        self.X_input = X
        self.debug = debug
        self.motif_shape = (200,1,16,18)

    def build_model(self, dropout=0.1, L1=0.0001, L2=0.0001, units=2, pool_width=10, lr=0.00001, merge_mode='sum', conv_filters=200, extra_conv=False, weighted=True):
        self.X = Input(shape=self.X_input.shape[1:], name="Input")
        self.X_RC = Input(shape=self.X_input.shape[1:], name="RC Input")
        self.conv_layer = Convolution2D(nb_filter=conv_filters, nb_row=16, nb_col=18, activation=relu, W_regularizer=l1_l2(L1,L2)) 
        self.X_conv = self.conv_layer(self.X)
        self.X_RC_conv = self.conv_layer(self.X_RC)
        self.pool_size = (1,int(pool_width))
        self.X_max_out = MaxPooling2D(pool_size=self.pool_size)(self.X_conv)
        self.X_RC_max_out = MaxPooling2D(pool_size=self.pool_size)(self.X_RC_conv)
        #self.X_merged = merge([self.X, self.X_RC], mode=custom_merge,output_shape=custom_merge_shape)
        #Batch size, filters, 0, 133/max pool size
        if merge_mode == 'sum':
            self.X_merged = merge([self.X_max_out, self.X_RC_max_out], mode=merge_mode)
        else:
            pass

        if extra_conv:
            self.X_merged_out = Convolution2D(nb_filter = 15, nb_row=1, nb_col=5, activation='relu', W_regularizer=l1l2(L1,L2))(self.X_merged)
        else:
            self.X_merged_out = self.X_merged
        self.X_reduced = Flatten()(self.X_merged_out)
        self.X_feat_concat = self.X_reduced
        self.X_feat_concat_dropout = Dropout(dropout)(self.X_feat_concat)
        self.X_feat_new = Dense(units, activation ='relu', W_regularizer=l1_l2(L1,L2))(self.X_feat_concat_dropout)
        self.predictions = Dense(1, activation='sigmoid', W_regularizer=l1_l2(L1,L2))(self.X_feat_new)
        self.model = keras.engine.training.Model(input=[self.X, self.X_RC],output=self.predictions)
        self.optimizer = keras.optimizers.Adam(lr=lr)
        self.model.compile(optimizer=self.optimizer, loss='binary_crossentropy')

    def train(self, X, X_RC, y, neg_weight, pos_weight, nb_epoch=60, batch_size=2000, early_stopping=True, patience = 10, outside_eval=None, val_split=0.10, save=False, verbose=0):
        callbacks = []
        if early_stopping:
            self.EarlyStopping = keras.callbacks.EarlyStopping('val_loss', patience=patience)
            callbacks.append(self.EarlyStopping)
        if save:
            filepath="{epoch:02d}-{val_acc:.2f}.hdf5"
            self.ModelCheckpoint = keras.callbacks.ModelCheckpoint(filepath,
                                                                   monitor="val_loss",
                                                                   save_best_only=True,
                                                                   mode="min",
                                                                   period=5)
            callbacks.append(self.ModelCheckpoint)
        #print history
        if outside_eval is not None:
            history = self.model.fit(x=[X, X_RC], y=y, batch_size = batch_size, nb_epoch = nb_epoch, callbacks=callbacks,validation_data=outside_eval, shuffle=True, class_weight={0:neg_weight, 1:pos_weight}, verbose=verbose)
        else:
            history = self.model.fit(x=[X, X_RC], y=y, batch_size = batch_size, nb_epoch = nb_epoch, callbacks=callbacks,validation_split=val_split, shuffle=True, class_weight={0:neg_weight, 1:pos_weight}, verbose=verbose)
        return history

    def predict(self, X, X_RC, batch_size=2000, verbose = 0):
        return self.model.predict([X, X_RC], batch_size=batch_size)
