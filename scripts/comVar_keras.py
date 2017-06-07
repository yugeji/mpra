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
from theano.tensor.signal.downsample import max_pool_2d
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
    Permute, Reshape, TimeDistributedDense
)
from keras.layers import Input, Dense, Flatten
from keras.engine.topology import Layer, merge, Container
from keras import activations, initializations, regularizers, constraints
import keras
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import GRU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2, l1l2
from keras.utils import np_utils
from keras import backend as K
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
from scipy.stats import norm
plt.style.use('seaborn-whitegrid')
from common_utils import ClassificationResult
from common_utils import BinaryClassProcessor
from keras.models import model_from_json 


def dump_model(model_obj, name):
    model_obj.save("%s.h5" % (name))
    print('Done dumping %s'%(name))


def save_models(model_obj, name=""):
    model_json = model_obj.model.to_json()
    with open("%s_model.json"%(name), 'w') as json_file:
        json_file.write(model_json)
    model_obj.model.save_weights("%s_weights.h5"%(name))
    print("Saved model to disk")

def matrix_grad(self, X, W, H, lr):
    WH = K.dot(W,H)
    obj = K.square((X - WH).norm(2))
    dW, dH = K.gradient(obj, W, H)
    Wnew = W - lr*dW
    Hnew = H - lr*dH
    updates = {W: Wnew, H:Hnew}
    return updates

def save_models(model_obj, name=""):
    model_json = model_obj.model.to_json()
    with open("%s_model.json"%(name), 'w') as json_file:
        json_file.write(model_json)
    model_obj.model.save_weights("%s_weights.h5"%(name))
    print("Saved model to disk")

class MatrixFactorizationLayer(Layer):
    def __init__(self, LK, output_dim, input_dim, **kwargs):
        self.LK = LK #factorized dimension
        self.output_dim = output_dim #H for now
        self.input_dim = input_dim #W for now
        self.L1 = 0
        self.L2 = 0
        self.iters = iters
        self.lr = 0.00001
        self.iters = 3
        self.iter_placeholder = K.variable(np.ones((self.iters,)))
        self.convg = 1e-4
        super(MatrixFactorizationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        mean = 0
        std = 1
        random_seed = 42
        self.W_shape = (self.input_dim[0], self.LK)
        self.H_shape = (self.LK, self.input_dim[1])
        self.inital_W = norm(loc=mean, scale=std, shape=self.W_shape)
        self.inital_H = norm(loc=mean, scale=std, shape=self.H_shape)
        self.W = K.variable(self.inital_W)
        self.H = K.variable(self.inital_H)
        self.trainable_weights = []
        super(MatrixFactorizationLayer, self).build(input_shape=input_shape)

    def call(self, X, mask=None):
        if self.iters > 1:
            self.WH = K.dot(self.W, self.H)
            self.obj = K.square((X - self.WH).norm(2))
            self.dW, self.dH = K.gradients(self.obj, [self.W, self.H])
            self.W = self.W - self.lr * self.dW
            self.H = self.H - self.lr * self.dH
        else:
            values, updates = theano.scan(fn = matrix_grad, outputs_info=None, sequences=[self.iter_placeholder], non_sequences=[self.X, self.W, self.H, self.lr], updates=updates, n_steps=self.iters)
        return self.H

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.H_shape[0], self.H_shape[1])

def theano_dot(X, CW_T):
    return T.dot(CW_T, X)

class MotifConnectionsLayer(Layer):
    def __init__(self, motif_connections, output_dim, L2, weighted=True, **kwargs):
        self.motif_connections = motif_connections
        self.shape = self.motif_connections.shape[-1]
        self.output_dim = output_dim
        self.L2 = L2
        self.activations = activations.relu
        self.weighted = weighted
        super(MotifConnectionsLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.weighted:
            self.W_shape = (self.shape,)
            self.W = self.add_weight(self.W_shape,
                                    initializer=initializations.glorot_uniform,
                                    name='{}_MC_W'.format(self.name),
                                    regularizer=l2(self.L2))
        self.C = K.variable(self.motif_connections)
        super(MotifConnectionsLayer, self).build(input_shape=input_shape)
        self.built = True

    def call(self, X, mask=None):
        if self.weighted:
            self.W_tiled = K.tile(self.W,(self.motif_connections.shape[0],1))
            self.CW = self.C * self.W_tiled #1953 x 70
        else:
            self.CW = self.C

        self.values, self.updates = theano.scan(fn=theano_dot, sequences=[X[:,:,0,:]], non_sequences=[self.CW.T])
        return self.values
        #return K.dot(self.CW.T, X[:,:,0,:]) #(X, CW)=

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim[0], self.output_dim[1])

class UnweightedMotifLayer(Layer):
    def __init__(self, motif_tensor, output_dim, L1, L2, **kwargs):
        self.motif_shape = motif_tensor.shape
        self.M = K.variable(motif_tensor)
        self.num_motifs = self.motif_shape[0]
        self.output_dim = output_dim
        self.num_filters = self.motif_shape[0]
        self.activation = activations.relu
        self.L1 = L1
        self.L2 = L2
        super(UnweightedMotifLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(UnweightedMotifLayer, self).build(input_shape=input_shape)
        self.built=True

    def call(self, X, mask=None):
        output = T.nnet.conv2d(X, self.M, border_mode="valid", filter_flip=False)
        output=self.activation(output)
        return output

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim[0], self.output_dim[1], self.output_dim[2])

class MotifLayer(Layer):
    def __init__(self, motif_tensor, output_dim, L1, L2, **kwargs):
        #print(_dict)

        self.motif_shape = motif_tensor.shape
        self.M = K.variable(motif_tensor)
        self.num_motifs = self.motif_shape[0]
        self.output_dim = output_dim
        self.num_filters = self.motif_shape[0]
        self.L1 = L1
        self.L2 = L2
        self.activation = activations.relu
        #InputSpec should specify dtype = dtype, shape = shape, ndim = ndim
        # self.activation = activations.get(activation)
        #self.b_regularizer = regularizers.get(b_regularizer)
        # self.activity_regularizer = regularizers.get(activity_regularizer)
        # self.input_spec = InputSpec(ndim=3)

        # self.bias = bias
        # self.inital_weights = weights
        # self.input
        super(MotifLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W_shape = (self.num_filters,)
        self.W = self.add_weight(self.W_shape,
                                 initializer=initializations.glorot_uniform,
                                 name='{}_ML_W'.format(self.name),
                                 regularizer=l1l2(self.L1, self.L2))
        #super(MotifLayer, self).build()
        super(MotifLayer, self).build(input_shape=input_shape)
        self.built = True

    def call(self, X, mask=None):
        self.MW = self.W.dimshuffle(0, 'x', 'x', 'x') * self.M
        output = T.nnet.conv2d(X, self.MW, border_mode='valid', filter_flip=False)
        # output = conv2d(input=x,filters=self.weightedMotifs,
        #                 border_mode = 'valid', fliter_flip=True)
        #conv2d takes input image (batch, input channel, rows, column)
        #conv2d takes filters (output_channels, input channels, filter rows, filter columns)
        #conv2d produces output (batch, output channels, output rows, output columns)
        output = self.activation(output)
        #print(output)
        return output

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim[0], self.output_dim[1], self.output_dim[2])

def get_layer_output(model, layer_index):
    weights = model.layers[layer_index].get_weights()[0]
    weights.shape = weights.shape
    return weights

class MetaKerasModel(object):
    def __init__(self, DataObj, motif, motif_connections, debug=False):
        self.DataObj = DataObj
        self.motifs = motif
        self.motif_shape = self.motifs.shape
        self.X_input = self.DataObj.X
        self.debug = debug
        self.motif_connections = motif_connections

    def build_model(self, dropout=0.01, L2=0.0001, L1=0.00001, pool_width=4, lr=0.00001, meta=True, weighted=True, merge_mode="sum", extra_conv = False, batch_norm=False):
        self.X = Input(shape=self.X_input.shape[1:], name="Sequence Input")
        self.X_RC = Input(shape=self.X_input.shape[1:], name="RC Input")
        self.dim = self.DataObj.X.shape[-1] - self.motifs.shape[-1]+1
        shared_motif_output = (self.motif_shape[0], 1, self.dim)
        # self.shared_motifs = MotifLayer(self.motifs, input_shape=self.X.shape[1:], output_shape=shared_motif_output,activation="relu", W_regularizer=l1l2(L1,L2))
        if weighted:
            self.shared_motifs = MotifLayer(motif_tensor=self.motifs,output_dim=shared_motif_output,input_shape=(1,4,150), L1=L1, L2=L2,name='shared_motif_layer',trainable=True)
        else:
            self.shared_motifs = UnweightedMotifLayer(motif_tensor=self.motifs, output_dim=shared_motif_output,input_shape=(1,4,150), L1=L1, L2=L2,name='unweighted_motif_layer',trainable=False)
        self.X_out = self.shared_motifs(self.X)
        self.X_RC_out = self.shared_motifs(self.X_RC)
        self.pool_size = (1,int(pool_width))
        self.X_max_out = MaxPooling2D(pool_size=self.pool_size)(self.X_out)
        self.X_RC_max_out = MaxPooling2D(pool_size=self.pool_size)(self.X_RC_out)
        #self.X_merged = merge([self.X, self.X_RC], mode=custom_merge,output_shape=custom_merge_shape)
        #Batch size, filters, 0, 133/max pool size
        self.pool_out_dim,_ = divmod(self.dim,int(pool_width))
        #print(self.pool_out_dim)
        self.motif_connections_output = (self.motif_connections.shape[-1],
                                         self.pool_out_dim)
        #print(self.motif_connections_output)
        self.motif_connections_layer = MotifConnectionsLayer(self.motif_connections,
                                                     self.motif_connections_output,
                                                     L2,
                                                     name="motif_connections_layer")
        
        self.X_MF_out = self.motif_connections_layer(self.X_max_out)
        self.X_RC_MF_out = self.motif_connections_layer(self.X_RC_max_out)
        self.X_merged = merge([self.X_MF_out, self.X_RC_MF_out], mode=merge_mode)
        if extra_conv:
            self.X_merged_out = Convolution2D(nb_filter = 15, nb_row=1, nb_col=5, activation='relu', W_regularizer=l1l2(L1,L2))(self.X_merged.dimshuffle(0,1,'x',2))
        else:
            self.X_merged_out = self.X_merged

        self.X_reduced = Flatten()(self.X_merged_out)
        self.X_feat_concat = self.X_reduced
        self.X_feat_concat_dropout = Dropout(dropout)(self.X_feat_concat)
        if batch_norm:
            self.X_feat_final = BatchNormalization()(self.X_feat_concat_dropout)
        else:
            self.X_feat_final = self.X_feat_concat_dropout
        self.predictions = Dense(1, activation='sigmoid', W_regularizer=l1l2(L1,L2))(self.X_feat_final)
        self.model = keras.engine.training.Model(input=[self.X, self.X_RC],output=self.predictions)
        self.optimizer = keras.optimizers.Adam(lr=lr)
        self.model.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=['binary_crossentropy'])

    
    def train(self, X, X_RC, y, neg_weight, pos_weight, nb_epoch=60, batch_size=250, early_stopping=True, outside_eval = None, patience = 10, val_split=0.07, save=False, verbose=0):
        callbacks = []
        if early_stopping:
            self.EarlyStopping = keras.callbacks.EarlyStopping('val_loss', patience=patience)
            callbacks.append(self.EarlyStopping)
        if save:
            filepath="{epoch:02d}-{val_acc:.2f}_metamotif.hdf5"
            self.ModelCheckpoint = keras.callbacks.ModelCheckpoint(filepath,
                                                                   monitor="val_loss",
                                                                   save_best_only=True,
                                                                   mode="min",
                                                                   period=5)
            callbacks.append(self.ModelCheckpoint)
        if outside_eval is not None:
            history = self.model.fit(x=[X, X_RC], y=y, batch_size = batch_size, nb_epoch = nb_epoch, callbacks=callbacks,validation_data=outside_eval, shuffle=True, class_weight={0:neg_weight, 1:pos_weight}, verbose=verbose)
        else:
            history = self.model.fit(x=[X, X_RC], y=y, batch_size = batch_size, nb_epoch = nb_epoch, callbacks=callbacks,validation_split=val_split, shuffle=True, class_weight={0:neg_weight, 1:pos_weight}, verbose=verbose)
        #print history
        return history 

    def predict(self, X, X_RC, batch_size=200, verbose = 0):
        return self.model.predict([X, X_RC], batch_size=batch_size)

class DoubleKerasModel(object):
    def __init__(self, DataObj, motif, debug=False):
        self.DataObj = DataObj
        self.motifs = motif
        self.motif_shape = self.motifs.shape
        self.X_input = self.DataObj.X
        self.debug = debug 

    def build_model(self, dropout=0.01, L2=0.0001, L1=0, pool_width=10, lr=0.00001, meta=True, weighted=True, merge_mode="sum", extra_conv = False, batch_norm=False):
        self.X = Input(shape=self.X_input.shape[1:], name ="Sequence Input")
        self.X_RC = Input(shape=self.X_input.shape[1:], name="RC Input")
        dim = self.DataObj.X.shape[-1] - self.motifs.shape[-1]+1
        shared_motif_output = (self.motif_shape[0], 1, dim)
        # self.shared_motifs = MotifLayer(self.motifs, input_shape=self.X.shape[1:], output_shape=shared_motif_output,activation="relu", W_regularizer=l1l2(L1,L2))
        if weighted:
            self.shared_motifs = MotifLayer(motif_tensor=self.motifs,output_dim=shared_motif_output,input_shape=(1,4,150), L1=L1, L2=L2,name='shared_motif_layer',trainable=True)
        else:
            self.shared_motifs = UnweightedMotifLayer(motif_tensor=self.motifs, output_dim=shared_motif_output,input_shape=(1,4,150), L1=L1, L2=L2,name='unweighted_motif_layer',trainable=False)
        self.X_out = self.shared_motifs(self.X)
        self.X_RC_out = self.shared_motifs(self.X_RC)
        self.pool_size = (1,int(pool_width))
        self.X_max_out = MaxPooling2D(pool_size=self.pool_size)(self.X_out)
        self.X_RC_max_out = MaxPooling2D(pool_size=self.pool_size)(self.X_RC_out)
        #self.X_merged = merge([self.X, self.X_RC], mode=custom_merge,output_shape=custom_merge_shape)
        #Batch size, filters, 0, 133/max pool size
        self.X_merged = merge([self.X_max_out, self.X_RC_max_out], mode=merge_mode)

        if extra_conv:
            self.X_merged_out = Convolution2D(nb_filter = 15, nb_row=1, nb_col=5, activation='relu', W_regularizer=l1l2(L1,L2))(self.X_merged)
        else:
            self.X_merged_out = self.X_merged

        
        self.X_reduced = Flatten()(self.X_merged_out)
        self.X_feat_concat = self.X_reduced
        self.X_feat_concat_dropout = Dropout(dropout)(self.X_feat_concat)
        if batch_norm:
            self.X_feat_final = BatchNormalization()(self.X_feat_concat_dropout)
        else:
            self.X_feat_final = self.X_feat_concat_dropout
        self.predictions = Dense(1, activation='sigmoid', W_regularizer=l1l2(L1,L2))(self.X_feat_final)
        self.model = keras.engine.training.Model(input=[self.X, self.X_RC],output=self.predictions)
        self.optimizer = keras.optimizers.Adam(lr=lr)
        self.model.compile(optimizer=self.optimizer, loss='binary_crossentropy')

    def train(self, X, X_RC, y, neg_weight, pos_weight, nb_epoch=60, batch_size=250, early_stopping=True, patience = 10, outside_eval = None, val_split=0.07, save=False, verbose=0):
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
        if outside_eval is not None:
            history = self.model.fit(x=[X, X_RC], y=y, batch_size = batch_size, nb_epoch = nb_epoch, callbacks=callbacks,validation_data=outside_eval, shuffle=True, class_weight={0:neg_weight, 1:pos_weight}, verbose=verbose)
        else:
            history = self.model.fit(x=[X, X_RC], y=y, batch_size = batch_size, nb_epoch = nb_epoch, callbacks=callbacks,validation_split=val_split, shuffle=True, class_weight={0:neg_weight, 1:pos_weight}, verbose=verbose)
        #print history
        return history 

    def predict(self, X, X_RC, batch_size=200, verbose = 0):
        return self.model.predict([X, X_RC], batch_size=batch_size)


class SimpleKerasModel(object):
    def __init__(self, DataObj, debug=False):
        self.DataObj = DataObj
        self.X_input = self.DataObj.X
        self.debug = debug
        self.motif_shape = (200,1,4,18)

    def build_model(self, dropout=0.01, L1=0, L2=0.0001, pool_width=10, lr=0.00001, merge_mode='sum', conv_filters=150, extra_conv=False, weighted=True):
        self.X = Input(shape=self.X_input.shape[1:], name="Input")
        self.X_RC = Input(shape=self.X_input.shape[1:], name="RC Input")
        dim = self.DataObj.X.shape[-1] - self.motif_shape[-1] + 1
        self.conv_layer = Convolution2D(nb_filter=conv_filters, nb_row=4, nb_col=18, activation=relu, W_regularizer=l1l2(L1,L2)) 
        self.X_conv = self.conv_layer(self.X)
        self.X_RC_conv = self.conv_layer(self.X_RC)
        self.pool_size = (1,int(pool_width))
        self.X_max_out = MaxPooling2D(pool_size=self.pool_size)(self.X_conv)
        self.X_RC_max_out = MaxPooling2D(pool_size=self.pool_size)(self.X_RC_conv)
        #self.X_merged = merge([self.X, self.X_RC], mode=custom_merge,output_shape=custom_merge_shape)
        #Batch size, filters, 0, 133/max pool size
        self.X_merged = merge([self.X_max_out, self.X_RC_max_out], mode=merge_mode)

        if extra_conv:
            self.X_merged_out = Convolution2D(nb_filter = 15, nb_row=1, nb_col=5, activation='relu', W_regularizer=l1l2(L1,L2))(self.X_merged)
        else:
            self.X_merged_out = self.X_merged

        
        self.X_reduced = Flatten()(self.X_merged_out)
        self.X_feat_concat = self.X_reduced
        self.X_feat_concat_dropout = Dropout(dropout)(self.X_feat_concat)
        self.predictions = Dense(1, activation='sigmoid', W_regularizer=l1l2(L1,L2))(self.X_feat_concat)
        self.model = keras.engine.training.Model(input=[self.X, self.X_RC],output=self.predictions)
        self.optimizer = keras.optimizers.Adam(lr=lr)
        self.model.compile(optimizer=self.optimizer, loss='binary_crossentropy')

    def train(self, X, X_RC, y, neg_weight, pos_weight, nb_epoch=60, batch_size=250, early_stopping=True, patience = 10, outside_eval=None, val_split=0.07, save=False, verbose=0):
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

    def predict(self, X, X_RC, batch_size=200, verbose = 0):
        return self.model.predict([X, X_RC], batch_size=batch_size)


class SimpleCNNModel(object):
    def __init__(self, X, debug=False):
        self.X_input = X
        self.debug = debug
        self.motif_shape = (200,1,4,18)

    def build_model(self, dropout=0.01, L1=0, L2=0.0001, pool_width=10, lr=0.00001, merge_mode='sum', conv_filters=150, extra_conv=False, weighted=True):
        self.X = Input(shape=self.X_input.shape[1:], name="Input")
        self.X_RC = Input(shape=self.X_input.shape[1:], name="RC Input")
        self.conv_layer = Convolution2D(nb_filter=conv_filters, nb_row=4, nb_col=18, activation=relu, W_regularizer=l1l2(L1,L2)) 
        self.X_conv = self.conv_layer(self.X)
        self.X_RC_conv = self.conv_layer(self.X_RC)
        self.pool_size = (1,int(pool_width))
        self.X_max_out = MaxPooling2D(pool_size=self.pool_size)(self.X_conv)
        self.X_RC_max_out = MaxPooling2D(pool_size=self.pool_size)(self.X_RC_conv)
        #self.X_merged = merge([self.X, self.X_RC], mode=custom_merge,output_shape=custom_merge_shape)
        #Batch size, filters, 0, 133/max pool size
        self.X_merged = merge([self.X_max_out, self.X_RC_max_out], mode=merge_mode)

        if extra_conv:
            self.X_merged_out = Convolution2D(nb_filter = 15, nb_row=1, nb_col=5, activation='relu', W_regularizer=l1l2(L1,L2))(self.X_merged)
        else:
            self.X_merged_out = self.X_merged
        self.X_reduced = Flatten()(self.X_merged_out)
        self.X_feat_concat = self.X_reduced
        self.X_feat_concat_dropout = Dropout(dropout)(self.X_feat_concat)
        self.predictions = Dense(1, activation='sigmoid', W_regularizer=l1l2(L1,L2))(self.X_feat_concat)
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
