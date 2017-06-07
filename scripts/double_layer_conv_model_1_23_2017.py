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
plt.style.use('seaborn-whitegrid')
from common_utils import ClassificationResult
from common_utils import BinaryClassProcessor

def get_weights(Y):
    pos=np.sum(np.asarray(Y))
    neg=len(Y)-pos
    return float(len(Y))/pos, float(len(Y))/neg

class ConvLayer(object):
    def __init__(self, motifs):
        self.motifs = np.asarray(motifs, dtype=theano.config.floatX)
        self.motif_shape = self.motifs.shape
        self.W = theano.shared(value=self.motifs, name='W', borrow=True)
        self.X = T.ftensor4('X')
        self.conv_out = conv2d(input=self.X, filters=self.W,
                               border_mode="valid", filter_flip=False)
        self.conv_func = theano.function([self.X], self.conv_out, on_unused_input='warn')

# class ConvLayer(object):
#     def __init__(self, motifs):
#         self.motifs = motifs
#         self.motif_shape = self.motifs.shape
#         self.W = theano.shared(value=self.motifs, name='W', borrow=True)
#         self.X = T.ftensor4('X')
#         self.conv_out = conv2d(input=self.X, filters=self.W,
#                                border_mode="valid", filter_flip=False)
#         self.conv_func = theano.function([self.X, self.conv_out])

class ConvPredictor(object):
    def __init__(self, DataObj, name = '', debug = False):
        self.DataObj = DataObj
        self.Y = self.DataObj.Y
        self.X_conv = self.DataObj.X_conv
        self.X_comb_conv = self.DataObj.X_comb_conv
        self.X_comb_conv_shape = self.X_comb_conv.shape
        self.X_comb_conv_width = self.X_comb_conv_shape[1]
        self.classifier_dict = {}

    def Correct_RC_index(self,index):
        index_copy = np.copy(index)
        index_RC = self.X_conv.shape[0] + index_copy
        return np.hstack((index_copy, index_RC))

    def LogisticRegPredict(self, chrom_indices, other_indices, penalty='l2', solver='sag', tol=0.001, C=0.1, max_iter=100, should_scale=True, train_only = False):
        if self.RC and not self.combine_RC:
            chrom_indices = self.Correct_RC_index(chrom_indices)
            other_indices = self.Correct_RC_index(other_indices)
        self.LogisticRegObj = LogisticRegression(class_weight='balanced', penalty=penalty, solver=solver, tol=tol, C=C, max_iter=max_iter, random_state=42)
        if should_scale:
            scaler = StandardScaler()
            scaler.fit(self.X_comb_conv[other_indices])
            X_train= scaler.transform(self.X_comb_conv[other_indices])
            if not train_only:
                X_test = scaler.transform(self.X_comb_conv[chrom_indices])
        else:
            X_train = self.X_comb_conv[other_indices]
            if not train_only:
                X_test = self.X_comb_conv[chrom_indices]
        self.LogisticRegObj.fit(X_train, self.Y[other_indices])
        self.classifier_dict['LogisticReg'] = self.LogisticRegObj
        if train_only:
            return self.LogisticRegObj
        else:
            return self.LogisticRegObj.predict_proba(self.X_test)[:, 1],\
            self.Y[chrom_indices].astype(bool)

    def LogisticRegCVPredict(self, chrom_indices, other_indices, penalty = 'l2', solver = 'lbfgs', tol = 0.0001, n_jobs = 4, max_iter = 150, should_scale = True, train_only = False):
        if self.RC and not self.combine_RC:
            chrom_indices = self.Correct_RC_index(chrom_indices)
            other_indices = self.Correct_RC_index(other_indices)
        self.LogisticRegCVObj = LogisticRegressionCV(class_weight = 'balanced', penalty = penalty, solver = solver, n_jobs = n_jobs,  tol = tol, max_iter = max_iter, random_state = 42)
        if should_scale:
            scaler = StandardScaler()
            scaler.fit(self.X_comb_conv[other_indices])
            X_train= scaler.transform(self.X_comb_conv[other_indices])
            if not train_only:
                X_test = scaler.transform(self.X_comb_conv[chrom_indices])
        else:
            X._train = self.X_comb_conv[other_indices]
            if not train_only:
                X_test = self.X_comb_conv[chrom_indices]
        self.LogisticRegCVObj.fit(X_train, self.Y[other_indices])
        self.classifier_dict['LogisticRegCV'] = self.LogisticRegCVObj
        if train_only:
            return self.LogisticRegCVObj
        else:
            return self.LogisticRegCVObj.predict_proba(X_test)[:,1], self.Y[chrom_indices].astype(bool)

    def ElasticNet(self, chrom_indices, other_indices, alpha = 0.0001, l1_ratio = 0.15, loss = 'log', learning_rate = "optimal", penalty = "elasticnet", should_scale = True, train_only = False):
        if self.RC and not self.combine_RC:
            chrom_indices = self.Correct_RC_index(chrom_indices)
            other_indices = self.Correct_RC_index(other_indices)
        self.ElasticNetObj = SGDClassifier(loss = loss, alpha = alpha, l1_ratio = l1_ratio, penalty = penalty, class_weight = 'balanced', learning_rate = learning_rate, random_state = 42)
        if should_scale:
            scaler = StandardScaler()
            scaler.fit(self.X_comb_conv[other_indices])
            X_train= scaler.transform(self.X_comb_conv[other_indices])
            if not train_only:
                X_test = scaler.transform(self.X_comb_conv[chrom_indices])
        else:
            X_train = self.X_comb_conv[other_indices]
            if not train_only:
                X_test = self.X_comb_conv[chrom_indices]
        self.ElasticNetObj.fit(X_train, self.Y[other_indices])
        self.classifier_dict['ElasticNet']=self.ElasticNetObj
        if train_only:
            return self.ElasticNetObj
        else:
            return self.ElasticNetObj.predict_proba(X_test)[:,1], self.Y[chrom_indices].astype(bool)
    
    def RandomForest(self, chrom_indices, other_indices, n_estimators=500, n_jobs = 4, should_scale = True, train_only = False):
        if self.RC and not self.combine_RC:
            chrom_indices = self.Correct_RC_index(chrom_indices)
            other_indices = self.Correct_RC_index(other_indices)
        self.RandomForestObj = RandomForestClassifier(n_estimators=n_estimators, n_jobs = n_jobs, class_weight = 'balanced')
        if should_scale:
            scaler = StandardScaler()
            scaler.fit(self.X_comb_conv[other_indices])
            X_train= scaler.transform(self.X_comb_conv[other_indices])
            if not train_only:
                X_test = scaler.transform(self.X_comb_conv[chrom_indices])
        else:
            X_train = self.X_comb_conv[other_indices]
            if not train_only:
                X_test = self.X_comb_conv[chrom_indices]
        self.RandomForestObj.fit(X_train, self.Y[other_indices])
        self.classifier_dict['RandomForest'] = self.RandomForestObj
        if train_only:
            return self.RandomForestObj
        else:
            return self.RandomForestObj.predict_proba(X_test)[:,1], self.Y[chrom_indices].astype(bool)

    def GradientBoosting(self, chrom_indices, other_indices, n_estimators=500, should_scale = True, train_only = False):
        if self.RC and not self.combine_RC:
            chrom_indices = self.Correct_RC_index(chrom_indices)
            other_indices = self.Correct_RC_index(other_indices)
        self.GradientBoostingObj = GradientBoostingClassifier(n_estimators=n_estimators)
        if should_scale:
            scaler = StandardScaler()
            scaler.fit(self.X_comb_conv[other_indices])
            X_train= scaler.transform(self.X_comb_conv[other_indices])
            if not train_only:
                X_test = scaler.transform(self.X_comb_conv[chrom_indices])
        else:
            X_train = self.X_comb_conv[other_indices]
            if not train_only:
                X_test = self.X_comb_conv[chrom_indices]
        self.GradientBoostingObj.fit(X_train, self.Y[other_indices])
        self.classifier_dict['GradientBoostingObj'] = self.GradientBoostingObj
        if train_only:
            return self.GradientBoostingObj
        else:
            return self.GradientBoostingObj.predict_proba(X_test)[:,1], self.Y[chrom_indices].astype(bool)

    def MLP(self, chrom_indices, other_indices, hidden_layer_sizes = (100,), should_scale = True, train_only = False):
        if self.RC and not self.combine_RC:
            chrom_indices = self.Correct_RC_index(chrom_indices)
            other_indices = self.Correct_RC_index(other_indices)
        self.MLPObj = MLPClassifier(hidden_layer_sizes = hidden_layer_sizes)
        if should_scale:
            scaler = StandardScaler()
            scaler.fit(self.X_comb_conv[other_indices])
            X_train= scaler.transform(self.X_comb_conv[other_indices])
            if not train_only:
                X_test = scaler.transform(self.X_comb_conv[chrom_indices])
        else:
            X_train = self.X_comb_conv[other_indices]
            if not train_only:
                X_test = self.X_comb_conv[chrom_indices]
        self.MLPObj.fit(X_train, self.Y[other_indices])
        self.classifier_dict['MLP'] = self.MLPObj
        if train_only:
            return self.MLPObj
        else:
            return self.MLPObj.predict_proba(X_test)[:,1], self.Y[chrom_indices].astype(bool)

    def get_class_weights(self, tar_vec):
        counts = np.zeros(2)
        for idx, val in enumerate(tar_vec):
            if val == 0:
                counts[0] += 1
            elif val == 1: 
                counts[1] += 1
        _sum = np.sum(counts)
        return _sum / float(counts[0]), _sum / float(counts[1])

    def FC_1layer_model(self, chrom_indices, other_indices, patience=40, L1=1, L2=1, dropout_input =0, dropout_hidden=0, n_hidden=5):
        input_shape = self.X_comb_conv.shape[-1]
        self.model = Sequential()
        self.model.add(Dropout(dropout_input, input_shape=(input_shape,)))
        self.model.add(Dense(n_hidden, init='glorot_uniform',
                             W_regularizer=l1l2(L1, L2)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(dropout_hidden))
        self.model.add(Dense(n_hidden, init = 'glorot_uniform', W_regularizer=l1l2(L1,L2)))
        self.model.add(Dropout(dropout_hidden))
        self.model.add(Dense(output_dim=1, init='glorot_uniform',
                             W_regularizer=l1l2(L1, L2)))
        self.model.add(Activation('sigmoid'))
        num_epochs = 130
        learning_rate = 0.0001
        self.adam = keras.optimizers.Adam(lr = learning_rate)
        self.model.compile(optimizer=self.adam, loss = 'binary_crossentropy', metrics=['binary_crossentropy'])
        self.earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=True,  mode='auto')
        neg_weight, pos_weight = self.get_class_weights(self.Y[other_indices])
        self.model.fit(x=self.X_comb_conv[other_indices], y=self.Y[other_indices],
                       batch_size=128, nb_epoch=num_epochs, validation_split=0.1,
                       callbacks=[self.earlyStopping],
                       class_weight={0: neg_weight, 1: pos_weight}, verbose=False)
        #self.model.fit(x = self.X_comb_conv[other_indices], y = self.Y[other_indices], batch_size = 2500, nb_epoch = 150, validation_split = 0.10, callbacks = [self.earlyStopping], shuffle = True, class_weight= 'auto', verbose = False)
        #self.model.fit(x = self.X_comb_conv[other_indices], y = self.Y[other_indices], batch_size = 2500, nb_epoch = 125, shuffle = True, class_weight= {0: neg_weight, 1: pos_weight}, verbose = False)
        pred = self.model.predict(self.X_comb_conv[chrom_indices], batch_size=1000)
        return pred.flatten(), self.Y[chrom_indices].astype(bool)

    def Conv_model(self, chrom_indices, other_indices, input_shape = 133, patience = 20, L1 = 1, L2 = 1, dropout_input = 0.2, dropout_hidden = 0.2, n_hidden = 5):
        self.model = Sequential()
        self.model.add(Dropout(dropout_input, input_shape(input_shape, )))
        #self.model.add(Convolution())

def Rectifier(x):
    return np.maximum(x,0)

class ToyTester(object):
    def __init__(self, num_motifs = 15, pwm_len = 10, num_dna = 100, dna_length = 20):
        self.motifs = self.simulate_motifs(num_motifs = num_motifs, pwm_len = pwm_len)
        self.dna = self.simulate_dna(num_dna = num_dna, dna_length = dna_length)

    def simulate_motifs(self, num_motifs = 5, pwm_len = 5):
        motifs = np.zeros((num_motifs,1,4,pwm_len))
        for i in range(num_motifs):

            motifs[i,0,:,:] = np.random.rand(4, pwm_len)
        return motifs

    def simulate_dna(self,num_dna = 5, dna_length = 10):
        dna = np.zeros((num_dna,1,4,dna_length))
        for i in range(num_dna):
            dna[i,0,:,:] = np.random.rand(4, dna_length)
        return dna 

    def calculate_actual_convs(self, w = 5, b = 0):
        output_mat = np.zeros((self.dna.shape[0], self.motifs.shape[0], 1, self.dna.shape[-1]-self.motifs.shape[-1]+1))
        for n in range(self.dna.shape[0]):
            for k in range(self.motifs.shape[0]):
                for i in range(self.dna.shape[-1] - self.motifs.shape[-1] + 1):
                    output_mat[n,k,0,i] = Rectifier(w*np.sum((self.dna[n,0,:,i:i+self.motifs.shape[3]]*self.motifs[k,0,:,:]).flatten())+ b)
        return output_mat

def adam(loss, all_params, learning_rate=0.0005, b1=0.9, b2=0.999, e=1e-8,
         gamma=0):
    """
    ADAM update rules
    Default values are taken from [Kingma2014]
    References:
    [Kingma2014] Kingma, Diederik, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."
    arXiv preprint arXiv:1412.6980 (2014).
    http://arxiv.org/pdf/1412.6980v4.pdf
    """
    updates = []
    all_grads = theano.grad(loss, all_params)
    alpha = learning_rate
    t = theano.shared(np.float32(1))
    b1_t = b1*gamma**(t-1)   #(Decay the first moment running average coefficient)

    for theta_previous, g in zip(all_params, all_grads):
        m_previous = theano.shared(np.zeros(theta_previous.get_value().shape,
                                            dtype=theano.config.floatX))
        v_previous = theano.shared(np.zeros(theta_previous.get_value().shape,
                                            dtype=theano.config.floatX))

        m = b1_t*m_previous + (1 - b1_t)*g                             # (Update biased first moment estimate)
        v = b2*v_previous + (1 - b2)*g**2                              # (Update biased second raw moment estimate)
        m_hat = m / (1-b1**t)                                          # (Compute bias-corrected first moment estimate)
        v_hat = v / (1-b2**t)                                          # (Compute bias-corrected second raw moment estimate)
        theta = theta_previous - (alpha * m_hat) / (T.sqrt(v_hat) + e) #(Update parameters)

        updates.append((m_previous, m))
        updates.append((v_previous, v))
        updates.append((theta_previous, theta) )
    updates.append((t, t + 1.))
    return updates

def RMSprop(cost, params, lr=0.0001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def custom_merge1(inputs):
    l = inputs[0]
    r = inputs[1]
    l_reduced = l[:,:,0,:].flatten(ndim=2)
    r_reduced = r[:,:,0,:].flatten(ndim=2)
    return T.largest(l_reduced,r_reduced)

def custom_merge2(inputs):
    l = inputs[0]
    r = inputs[1]
    l_reduced = l[:,:,0,:].flatten(ndim=2)
    r_reduced = r[:,:,0,:].flatten(ndim=2)
    return l_reduced + r_reduced

def custom_merge_shape(input_shape):
    print "Input shape input is "
    print(input_shape)
    output_shape = (input_shape[0], input_shape[1]*input_shape[3])
    return input_shape

class MotifLayer(Layer):
    def __init__(self, motif_tensor, output_dim, L1, L2, **kwargs):
        _dict = kwargs
        print(_dict)

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
                                 name='{}_W'.format(self.name),
                                 regularizer=l1l2(self.L1, self.L2))
        #super(MotifLayer, self).build()
        super(MotifLayer, self).build(input_shape=input_shape)
        self.built = True

    def call(self, X, mask=None):
        self.MW = self.W.dimshuffle(0, 'x', 'x', 'x') * self.M
        output = T.nnet.conv2d(X, self.MW, border_mode='valid', filter_flip=False)
        # output = conv2d(input=x,filters=self.weightedMotifs,
        #                 border_mode = 'valid', fliter_flip=True)
        output = self.activation(output)
        print(output)
        return output

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim[0], self.output_dim[1], self.output_dim[2])

def get_layer_output(model, layer_index):
    weights = model.layers[layer_index].get_weights()[0]
    weights.shape = weights.shape
    return weights



class DoubleKerasModel(object):
    def __init__(self, DataObj, motif):
        self.DataObj = DataObj
        self.motifs = motif
        self.motif_shape = self.motifs.shape
        self.X_input = self.DataObj.X

    def build_model(self, dropout=0.01, L2=0.0001, L1=0.00001, L2_W2=0.01, L1_W2=0.001, pool_width = 10, extra_conv = False):
        self.X = Input(shape=self.X_input.shape[1:])
        self.X_RC = Input(shape=self.X_input.shape[1:])
        dim = self.DataObj.X.shape[-1] - self.motifs.shape[-1]+1
        shared_motif_output = (self.motif_shape[0], 1, dim)
        # self.shared_motifs = MotifLayer(self.motifs, input_shape=self.X.shape[1:], output_shape=shared_motif_output,activation="relu", W_regularizer=l1l2(L1,L2))
        self.shared_motifs = MotifLayer(motif_tensor=self.motifs,output_dim=shared_motif_output,input_shape=(1,4,150), L1=L1, L2=L2,name='shared_motif_layer',trainable=True)
        self.X_out = self.shared_motifs(self.X)
        self.X_RC_out = self.shared_motifs(self.X_RC)
        self.pool_size = (1,pool_width)
        self.X_max_out = MaxPooling2D(pool_size=self.pool_size)(self.X_out)
        self.X_RC_max_out = MaxPooling2D(pool_size=self.pool_size)(self.X_RC_out)
        #self.X_merged = merge([self.X, self.X_RC], mode=custom_merge,output_shape=custom_merge_shape)
        #Batch size, filters, 0, 133/max pool size
        self.X_merged = merge([self.X_max_out, self.X_RC_max_out], mode="max")

        if extra_conv:
            self.X_merged_out = Convolution2D(nb_filter = 15, nb_row=1, nb_col=5, activation='relu', W_regularizer=l1l2(L1,L2))(self.X_merged)
        else:
            self.X_merged_out = self.X_merged

        self.features_input = Input(shape=(self.DataObj.features_mat.shape[-1],))
        self.X_reduced = Flatten()(self.X_merged_out)
        self.X_feat_concat = merge([self.X_reduced, self.features_input], mode = 'concat', concat_axis = 1)
        self.X_feat_concat_dropout = Dropout(dropout)(self.X_feat_concat)
        self.predictions = Dense(1, activation='sigmoid', W_regularizer=l1l2(L1_W2,L2_W2))(self.X_feat_concat)
        self.model = keras.engine.training.Model(input=[self.X, self.X_RC, self.features_input],output=self.predictions)
        self.optimizer = keras.optimizers.Adam(lr=0.00001)
        self.model.compile(optimizer=self.optimizer, loss='binary_crossentropy')

    def train(self, X, X_RC, F, y, neg_weight, pos_weight, nb_epoch=25, batch_size=100, early_stopping=True, patience = 20, save=False, verbose=False):
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
        self.model.fit(x=[X, X_RC, F], y=y, batch_size = batch_size, nb_epoch = nb_epoch, callbacks=callbacks,validation_split=0.1, shuffle=True, class_weight={0:neg_weight, 1:pos_weight}, verbose=verbose)

    def predict(self, X, X_RC, F, batch_size, verbose = 0):
        return self.model.predict([X, X_RC,F], batch_size=batch_size)


class DoubleModel(object):
    def __init__(self, DataObj, motifs, RC):
        self.DataObj = DataObj
        self.motifs = motifs
        self.X_input = self.DataObj.X
        self.rng = np.random.RandomState(32)
        self.RC=RC

    def scipy_model(self, activation):
        pass
    # L2 = 0.001 Learning rate = 0.00001 works ok
    def model(self, activation=relu, L2_W1 = 0, L2_W2 = 0, gd_algorithm="adam", gd_learning_rate = 0.0001):
        self.gd_algorithm = gd_algorithm
        concat_size = self.motifs.shape[0]+self.DataObj.features_mat.shape[-1]
        self.M = theano.shared(value = self.motifs, name='M', borrow=True)
        #self.W1_values = 2*np.ones((self.motifs.shape[0]), dtype = theano.config.floatX)
        self.W1_values = np.asarray(self.rng.uniform(low=-np.sqrt(6./(self.motifs.shape[0])), high=np.sqrt(6./(self.motifs.shape[0])), size=self.motifs.shape[0]),
                             dtype=theano.config.floatX)
        self.W2_values = np.asarray(self.rng.uniform(low=-np.sqrt(6./float(concat_size)), high=np.sqrt(6./float(concat_size)),
                                                size=(self.motifs.shape[0]+self.DataObj.features_mat.shape[-1],)),
                                    dtype=theano.config.floatX)
        self.W1 = theano.shared(value=self.W1_values, name='W1', borrow=True)
        self.W2 = theano.shared(value=self.W2_values, name='W2', borrow=True)
        self.MW1 = self.W1.dimshuffle(0, 'x', 'x', 'x') * self.M
        self.X = T.ftensor4('X')
        self.conv_out = conv2d(input=self.X, filters=self.MW1, border_mode = 'valid', filter_flip=False)
        self.post_pool = max_pool_2d(input=self.conv_out,
                                     ds=(1,133),
                                     ignore_border=True)
        self.relu_out = activation(self.post_pool)
        self.reduced = self.relu_out[:,:,0,0]
        self.F = T.fmatrix('F') #features vector
        self.concat = T.concatenate([self.reduced, self.F], axis = 1)
        self.b_value = np.asarray(0.000001*np.ones((1,)),dtype=theano.config.floatX)
        self.b = theano.shared(value=self.b_value, name='b', borrow=True)
        self.output = sigmoid(T.dot(self.concat, self.W2.dimshuffle(0,'x')) + self.b)[:,0]
        self.Y = T.dvector('Y')
        self.cost = T.nnet.binary_crossentropy(self.output, self.Y).mean()
        if L2_W1 > 0:
            self.cost = self.cost + L2_W1*(self.W1.norm(2) ** 2)
        if L2_W2 > 0:
            self.cost = self.cost + L2_W2*(self.W2.norm(2) ** 2)

        self.full_params = [self.W1, self.W2, self.b]

        if gd_algorithm == "adam":
            self.updates = adam(self.cost, self.full_params, learning_rate = gd_learning_rate)
        elif gd_algorithm == "RMSprop":
            self.updates = RMSprop(self.cost, self.full_params, lr = gd_learning_rate)
        elif gd_algorithm == "gd":
            self.grads = T.grad(self.cost, self.full_params)
            self.updates = [(param_i, param_i - gd_learning_rate * grad_i) for param_i, grad_i in zip(self.full_params, self.grads)]
        self.train_model = theano.function([self.X,self.F,self.Y], self.cost, updates = self.updates, on_unused_input='warn')
        self.debug_train_model = theano.function([self.X,self.F,self.Y], self.cost, updates = self.updates, on_unused_input='warn', mode = NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
        self.predict_model = theano.function([self.X, self.F], self.output, on_unused_input='warn')

    def model2(self, activation=relu, L2_W1 = 0, L2_W2 = 0, pool_size = 40, gd_algorithm="adam", gd_learning_rate = 0.0001):
        self.gd_algorithm = gd_algorithm
        self.size_multiplier, _ = divmod(133, pool_size)
        concat_size = self.motifs.shape[0]+self.DataObj.features_mat.shape[-1]
        self.M = theano.shared(value = self.motifs, name='M', borrow=True)
        #self.W1_values = 2*np.ones((self.motifs.shape[0]), dtype = theano.config.floatX)
        self.W1_values = np.asarray(self.rng.uniform(low=-np.sqrt(6./(self.motifs.shape[0])), high=np.sqrt(6./(self.motifs.shape[0])), size=self.motifs.shape[0]),
                             dtype=theano.config.floatX)
        self.W2_values = np.asarray(self.rng.uniform(low=-np.sqrt(6./float(concat_size)), high=np.sqrt(6./float(concat_size)),
                                                size=(self.size_multiplier*self.motifs.shape[0]+self.DataObj.features_mat.shape[-1],)),
                                    dtype=theano.config.floatX)
        self.W1 = theano.shared(value=self.W1_values, name='W1', borrow=True)
        self.W2 = theano.shared(value=self.W2_values, name='W2', borrow=True)
        self.MW1 = self.W1.dimshuffle(0, 'x', 'x', 'x') * self.M
        self.X = T.ftensor4('X')
        self.conv_out = conv2d(input=self.X, filters=self.MW1, border_mode = 'valid', filter_flip=False)
        self.post_pool = max_pool_2d(input=self.conv_out,
                                     ds=(1,pool_size),
                                     ignore_border=True)
        self.relu_out = activation(self.post_pool)
        self.reduced = self.relu_out[:,:,0,:].flatten(ndim=2)
        self.F = T.fmatrix('F') #features vector
        self.concat = T.concatenate([self.reduced, self.F], axis = 1)
        self.b_value = np.asarray(0.000001*np.ones((1,)),dtype=theano.config.floatX)
        self.b = theano.shared(value=self.b_value, name='b', borrow=True)
        self.output = sigmoid(T.dot(self.concat, self.W2.dimshuffle(0,'x')) + self.b)[:,0]
        self.Y = T.dvector('Y')
        self.cost = T.nnet.binary_crossentropy(self.output, self.Y).mean()
        if L2_W1 > 0:
            self.cost = self.cost + L2_W1*(self.W1.norm(2) ** 2)
        if L2_W2 > 0:
            self.cost = self.cost + L2_W2*(self.W2.norm(2) ** 2)

        self.full_params = [self.W1, self.W2, self.b]

        if gd_algorithm == "adam":
            self.updates = adam(self.cost, self.full_params, learning_rate = gd_learning_rate)
        elif gd_algorithm == "RMSprop":
            self.updates = RMSprop(self.cost, self.full_params, lr = gd_learning_rate)
        elif gd_algorithm == "gd":
            self.grads = T.grad(self.cost, self.full_params)
            self.updates = [(param_i, param_i - gd_learning_rate * grad_i) for param_i, grad_i in zip(self.full_params, self.grads)]
        self.train_model = theano.function([self.X,self.F,self.Y], self.cost, updates = self.updates, on_unused_input='warn')
        self.debug_train_model = theano.function([self.X,self.F,self.Y], self.cost, updates = self.updates, on_unused_input='warn', mode = NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
        self.predict_model = theano.function([self.X, self.F], self.output, on_unused_input='warn')


    def model_RC(self, activation=relu, L2_W1 = 0, L2_W2 = 0, pool_size = 40, gd_algorithm="adam", gd_learning_rate = 0.0001):
        self.gd_algorithm = gd_algorithm
        self.size_multiplier, _ = divmod(133, pool_size)
        concat_size = self.motifs.shape[0]+self.DataObj.features_mat.shape[-1]
        self.M = theano.shared(value = self.motifs, name='M', borrow=True)
        #self.W1_values = 2*np.ones((self.motifs.shape[0]), dtype = theano.config.floatX)
        self.W1_values = np.asarray(self.rng.uniform(low=-np.sqrt(6./(self.motifs.shape[0])), high=np.sqrt(6./(self.motifs.shape[0])), size=self.motifs.shape[0]),
                             dtype=theano.config.floatX)
        self.W2_values = np.asarray(self.rng.uniform(low=-np.sqrt(6./float(concat_size)), high=np.sqrt(6./float(concat_size)),
                                                size=(self.size_multiplier*self.motifs.shape[0]+self.DataObj.features_mat.shape[-1],)),
                                    dtype=theano.config.floatX)
        self.W1 = theano.shared(value=self.W1_values, name='W1', borrow=True)
        self.W2 = theano.shared(value=self.W2_values, name='W2', borrow=True)
        self.MW1 = self.W1.dimshuffle(0, 'x', 'x', 'x') * self.M
        self.X = T.ftensor4('X')
        self.X_RC = T.ftensor4('X_RC')
        self.conv_out = conv2d(input=self.X, filters=self.MW1, border_mode = 'valid', filter_flip=False)
        self.post_pool = max_pool_2d(input=self.conv_out,
                                     ds=(1,pool_size),
                                     ignore_border=True)
        self.RC_conv_out = conv2d(input=self.X_RC, filters=self.MW1, border_mode='valid',filter_flip=False)
        self.RC_post_pool = max_pool_2d(input=self.RC_conv_out,
                                        ds=(1, pool_size),
                                        ignore_border=True)
        self.relu_out = activation(self.post_pool)
        self.reduced = self.relu_out[:,:,0,:].flatten(ndim=2)
        self.RC_relu_out = activation(self.RC_post_pool)
        self.RC_reduced = self.relu_out[:,:,0,:].flatten(ndim=2)
        self.comb_reduced = T.largest(self.reduced, self.RC_reduced)
        self.F = T.fmatrix('F') #features vector
        self.concat = T.concatenate([self.comb_reduced, self.F], axis = 1)
        self.b_value = np.asarray(0.000001*np.ones((1,)),dtype=theano.config.floatX)
        self.b = theano.shared(value=self.b_value, name='b', borrow=True)
        self.output = sigmoid(T.dot(self.concat, self.W2.dimshuffle(0,'x')) + self.b)[:,0]
        self.Y = T.dvector('Y')
        self.cost = T.nnet.binary_crossentropy(self.output, self.Y).mean()
        if L2_W1 > 0:
            self.cost = self.cost + L2_W1*(self.W1.norm(2) ** 2)
        if L2_W2 > 0:
            self.cost = self.cost + L2_W2*(self.W2.norm(2) ** 2)

        self.full_params = [self.W1, self.W2, self.b]

        if gd_algorithm == "adam":
            self.updates = adam(self.cost, self.full_params, learning_rate = gd_learning_rate)
        elif gd_algorithm == "RMSprop":
            self.updates = RMSprop(self.cost, self.full_params, lr = gd_learning_rate)
        elif gd_algorithm == "gd":
            self.grads = T.grad(self.cost, self.full_params)
            self.updates = [(param_i, param_i - gd_learning_rate * grad_i) for param_i, grad_i in zip(self.full_params, self.grads)]
        self.train_model = theano.function([self.X, self.X_RC, self.F,self.Y], self.cost, updates = self.updates, on_unused_input='warn')
        self.debug_train_model = theano.function([self.X,self.X_RC,self.F,self.Y], self.cost, updates = self.updates, on_unused_input='warn', mode = NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
        self.predict_model = theano.function([self.X,self.X_RC, self.F], self.output, on_unused_input='warn')

    def model_RC2(self, activation=relu, L2_W1 = 0, L2_W2 = 0, pool_size = 40, gd_algorithm="adam", gd_learning_rate = 0.0001):
        self.gd_algorithm = gd_algorithm
        self.size_multiplier, _ = divmod(133, pool_size)
        concat_size = self.motifs.shape[0]+self.DataObj.features_mat.shape[-1]
        self.M = theano.shared(value = 10*self.motifs, name='M', borrow=True)
        #self.W1_values = 2*np.ones((self.motifs.shape[0]), dtype = theano.config.floatX)
        self.W2_values = np.asarray(self.rng.uniform(low=-np.sqrt(6./float(concat_size)), high=np.sqrt(6./float(concat_size)),
                                                size=(self.size_multiplier*self.motifs.shape[0]+self.DataObj.features_mat.shape[-1],)),
                                    dtype=theano.config.floatX)
        self.W2 = theano.shared(value=self.W2_values, name='W2', borrow=True)
        self.X = T.ftensor4('X')
        self.X_RC = T.ftensor4('X_RC')
        self.conv_out = conv2d(input=self.X, filters=self.M, border_mode = 'valid', filter_flip=False)
        self.post_pool = max_pool_2d(input=self.conv_out,
                                     ds=(1,pool_size),
                                     ignore_border=True)
        self.RC_conv_out = conv2d(input=self.X_RC, filters=self.M, border_mode='valid',filter_flip=False)
        self.RC_post_pool = max_pool_2d(input=self.RC_conv_out,
                                        ds=(1, pool_size),
                                        ignore_border=True)
        self.relu_out = activation(self.post_pool)
        self.reduced = self.relu_out[:,:,0,:].flatten(ndim=2)
        self.RC_relu_out = activation(self.RC_post_pool)
        self.RC_reduced = self.relu_out[:,:,0,:].flatten(ndim=2)
        self.comb_reduced = T.largest(self.reduced, self.RC_reduced)
        self.F = T.fmatrix('F') #features vector
        self.concat = T.concatenate([self.comb_reduced, self.F], axis = 1)
        self.b_value = np.asarray(0.0000001*np.ones((1,)),dtype=theano.config.floatX)
        self.b = theano.shared(value=self.b_value, name='b', borrow=True)
        self.output = sigmoid(T.dot(self.concat, self.W2.dimshuffle(0,'x')) + self.b)[:,0]
        self.Y = T.dvector('Y')
        self.ClassWeights = T.fvector('ClassW')
        self.cost = T.nnet.binary_crossentropy(self.output, self.Y).mean()
        if L2_W2 > 0:
            self.cost = self.cost + L2_W2*(self.W2.norm(2) ** 2)

        self.full_params = [self.W2, self.b]

        if gd_algorithm == "adam":
            self.updates = adam(self.cost, self.full_params, learning_rate = gd_learning_rate)
        elif gd_algorithm == "RMSprop":
            self.updates = RMSprop(self.cost, self.full_params, lr = gd_learning_rate)
        elif gd_algorithm == "gd":
            self.grads = T.grad(self.cost, self.full_params)
            self.updates = [(param_i, param_i - gd_learning_rate * grad_i) for param_i, grad_i in zip(self.full_params, self.grads)]
        self.train_model = theano.function([self.X, self.X_RC, self.F,self.Y], self.cost, updates = self.updates, on_unused_input='warn')
        self.debug_train_model = theano.function([self.X,self.X_RC,self.F,self.Y], self.cost, updates = self.updates, on_unused_input='warn', mode = NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
        self.predict_model = theano.function([self.X,self.X_RC, self.F], self.output, on_unused_input='warn')

    def compile_test(self):
        self.multiply_test = theano.function([], self.MW1)
        self.conv_test = theano.function([self.X], self.post_pool)
        self.relu_test = theano.function([self.X], self.relu_out)
        self.reduced_test = theano.function([self.X], self.reduced)
        self.concat_test = theano.function([self.X, self.F], self.concat)
        self.func = theano.function([self.X, self.F], self.output)
        self.cost_test = theano.function([self.X, self.F, self.Y], self.cost)
        if self.gd_algorithm == 'gd':
            self.grad_test = theano.function([self.X, self.F, self.Y], self.grads)

    def fit(self, X, F, Y, n_epochs=40, minibatch_size=150):
        n_batches, last_batch = divmod(X.shape[0], minibatch_size)
        self.costs = []
        start = time.time()
        indices = np.arange(X.shape[0])
        for epoch_idx in range(n_epochs):
            np.random.shuffle(indices)
            for minibatch_idx in range(n_batches):
                start = minibatch_idx*minibatch_size
                end = (minibatch_idx+1)*minibatch_size
                s = indices[start:end]
                cost = self.train_model(X[s], F[s], Y[s])
                self.costs.append(cost)
                # if minibatch_idx > 28:
                #     break
            s = indices[-last_batch:]
            cost = self.train_model(X[s],F[s], Y[s])
            self.costs.append(cost)
            # if epoch_idx == 0:
            #     print cost
            #     return
            if epoch_idx % 5 == 0:
                print "Epoch %i completed in %0.02f, last epoch cost: %0.05f"%(epoch_idx, time.time()-start, np.mean(np.asarray(self.costs[-(n_batches-1):])))

    def predict(self, X, F, minibatch_size = 150):
        n_batches, last_batch = divmod(X.shape[0], minibatch_size)
        Y_pred = np.zeros((X.shape[0],))
        start = time.time()
        indices = np.arange(X.shape[0])
        for minibatch_idx in range(n_batches):
            start = minibatch_idx*minibatch_size
            end = (minibatch_idx+1)*minibatch_size
            s = indices[start:end]
            Y_cur = self.predict_model(X[s], F[s])
            Y_pred[s]=Y_cur.flatten()
            # if minibatch_idx > 28:
            #     break
        s = indices[-last_batch:]
        Y_cur = self.predict_model(X[s],F[s])
        Y_pred[s]=Y_cur.flatten()
        return Y_pred

    def predict2(self, X, X_RC, F, minibatch_size = 150):
        n_batches, last_batch = divmod(X.shape[0], minibatch_size)
        Y_pred = np.zeros((X.shape[0],))
        start = time.time()
        indices = np.arange(X.shape[0])
        for minibatch_idx in range(n_batches):
            start = minibatch_idx*minibatch_size
            end = (minibatch_idx+1)*minibatch_size
            s = indices[start:end]
            Y_cur = self.predict_model(X[s], X_RC[s] ,F[s])
            Y_pred[s]=Y_cur.flatten()
            # if minibatch_idx > 28:
            #     break
        s = indices[-last_batch:]
        Y_cur = self.predict_model(X[s],X_RC[s],F[s])
        Y_pred[s]=Y_cur.flatten()
        return Y_pred

    def fit2(self, X, X_RC, F, Y, n_epochs=40, minibatch_size=150, debug=False):
        n_batches, last_batch = divmod(X.shape[0], minibatch_size)
        self.costs = []
        start = time.time()
        indices = np.arange(X.shape[0])
        for epoch_idx in range(n_epochs):
            np.random.shuffle(indices)
            for minibatch_idx in range(n_batches):
                start = minibatch_idx*minibatch_size
                end = (minibatch_idx+1)*minibatch_size
                s = indices[start:end]
                cost = self.train_model(X[s], X_RC[s], F[s], Y[s])
                self.costs.append(cost)
                if debug and minibatch_idx > 28:
                    break
            s = indices[-last_batch:]
            cost = self.train_model(X[s],X_RC[s], F[s], Y[s])
            self.costs.append(cost)
            if debug and epoch_idx == 0:
                print cost
                return
            if epoch_idx % 5 == 0:
                print "Epoch %i completed in %0.02f, last epoch cost: %0.05f"%(epoch_idx, time.time()-start, np.mean(np.asarray(self.costs[-(n_batches-1):])))

    
class DataConvolver(object):
    def __init__(self, X, X_RC, Y, motif_shape, should_conv = True, ConvLayerObj = None, features_df = None, add_features = True, keep_output = False, RC = True, combine_RC = True, RC_combine_mode = 'max', name = "", debug = False):
        self.X = np.asarray(X, dtype=theano.config.floatX)
        self.X_RC = np.asarray(X_RC, dtype=theano.config.floatX)
        self.Y = Y.astype(bool)
        self.should_conv = should_conv
        self.features_df = features_df
        self.motif_shape = motif_shape
        self.features_mat = np.asarray(self.features_df.values, dtype=theano.config.floatX)
        self.feature_map_shape = [self.motif_shape[0], self.X.shape[-1] - self.motif_shape[-1] + 1]
        self.input_shape = self.X.shape
        self.add_features = add_features
        if self.should_conv:
            self.ConvLayerObj = ConvLayerObj
            self.keep_output = keep_output
            self.RC = RC
            self.combine_RC = combine_RC
            self.RC_combine_mode = RC_combine_mode
            if self.RC_combine_mode == 'max':
                self.RC_max = True
                self.add_RC = False
            elif self.RC_combine_mode == 'add':
                self.RC_max = False
                self.add_RC = True
            self.name = name
            self.debug = debug
            self.Process()

    def Process(self):
        self.X_conv, self.X_full_conv = self.Conv(self.X, self.X.shape, self.ConvLayerObj, keep_output=self.keep_output)
        if self.RC and self.combine_RC:
            self.X_RC_conv, self.X_RC_full_conv = self.Conv(self.X_RC, self.X_RC.shape, self.ConvLayerObj, keep_output=self.keep_output)
            self.X_X_RC_max = np.zeros_like(self.X_conv)
            if self.RC_max:
                for i in range(self.X_conv.shape[0]):
                    for j in range(self.X_conv.shape[1]):
                        if self.add_RC:  #add convolutions from forward and reverse strands
                            self.X_X_RC_max[i,j] = self.X_conv[i,j] + self.X_RC_conv[i,j]
                        else: #else take the max of the two. 
                            self.X_X_RC_max[i,j] = max(self.X_conv[i,j], self.X_RC_conv[i,j])
                if self.add_features:
                    self.X_comb_conv = np.hstack((self.X_X_RC_max, self.features_mat))
                else:
                    self.X_comb_conv = self.X_X_RC_max
            else:
                self.X_X_RC_comb = np.hstack((self.X_conv, self.X_RC_conv)) #merge features into one supervector
                if self.add_features:
                    self.X_comb_conv = np.hstack((self.X_X_RC_comb, self.features_mat))
                    if self.debug:
                        print(self.X_conv.shape)
                        print(self.X_conv)
                        print(self.X_RC_conv.shape)
                        print(self.X_X_RC_comb.shape)
                        print(self.features_mat.shape)
                        print("Final shape is")
                        print(self.X_comb_conv.shape)
                else:
                    self.X_comb_conv = self.X_X_RC_comb
        elif self.RC and not self.combine_RC:
            self.X_RC_conv, self.X_RC_full_conv = self.Conv(self.X_RC, self.X_RC.shape, self.ConvLayerObj, keep_output=self.keep_output)
            if self.add_features:
                if self.debug:
                    print(self.features_mat.shape)
                    print(self.X_RC_conv.shape)
                    print(self.X_conv.shape)
                #self.X_RC_conv_features = np.hstack((self.X_RC_conv, self.features_mat))
                self.X_conv_features = np.hstack((self.X_conv, self.features_mat))
                self.X_comb_conv = np.vstack((self.X_RC_conv, self.X_conv_features))
            else:
                self.X_comb_conv = np.vstack((self.X_RC_conv, self.X_conv)) 
            Y_copy = np.copy(self.Y)
            self.Y = np.hstack((Y_copy, Y_copy)) #Duplicate my Yo output 
        elif not self.RC:
            if self.add_features:
                self.X_comb_conv = np.hstack((self.X_conv, self.features_mat))
            else:
                self.X_comb_conv = self.X_conv
        self.X_comb_conv_shape = self.X_comb_conv.shape
        self.X_comb_conv_width = self.X_comb_conv_shape[1]
        self.classifier_dict = {}


    def Correct_RC_index(self, index):
        index_copy = np.copy(index)
        index_RC = self.X_conv.shape[0] + index_copy
        return np.hstack((index_copy, index_RC))

    def Conv(self, X, input_shape, ConvLayerObj, minibatch_size=2000, max_axis = 1, keep_output=False):
        minibatches, rem = divmod(input_shape[0], minibatch_size)
        # output = np.zeros((X.shape[0], self.motif_shape[1], 0, X.shape[-1] - self.motif_shape[-1] + 1 ))
        # output = np.zeros((X.shape[0], X.shape[-1] - self.motif_shape[-1] + 1 ))
        output = np.zeros((X.shape[0], self.motif_shape[0]))
        if keep_output:
            output_full_conv = np.zeros((input_shape[0], self.feature_map_shape[-1]))
        for idx in range(minibatches):
            cur_conv_output = ConvLayerObj.conv_func(
                X[idx*minibatch_size:(idx+1)*minibatch_size])  #output is index, filter, _, _
            if keep_output:
                output_full_conv[idx*minibatch_size:(idx+1)*minibatch_size] = cur_conv_output
            # print(cur_conv_output.shape)
            for cur_idx in range(minibatch_size):
                #self.transformed_output =self.cur_conv_output[cur_idx, :, 0, :]
                cur_sub = cur_conv_output[cur_idx, :, 0, :]
                # print(cur_sub.shape)
                cur_max = np.amax(cur_conv_output[cur_idx, :, 0, :], axis=max_axis)
                output[idx*minibatch_size + cur_idx, :] = cur_max
                # break
        if rem > 0:
            cur_conv_output = ConvLayerObj.conv_func(
                X[minibatches*minibatch_size:])
            if keep_output:
                output_full_conv[minibatch_size*minibatches:] = cur_conv_output
            # print(cur_conv_output.shape)
            for cur_idx in range(cur_conv_output.shape[0]):
                cur_max = np.amax(cur_conv_output[cur_idx, :, 0, :],
                                  axis=max_axis)
                output[minibatches*minibatch_size + cur_idx, :] = cur_max
        if keep_output:
            return output, output_full_conv
        else:
            return output, None
        

class CvEngine(object):
    def __init__(self,name, motif_tensor, motif_names, merged_tensor, merged_tar,
                 output_dir, features_df, RC = True, keep_output = False, C=0.1, debug=False, penalty='l2', solver='sag'):
        self.name = name
        self.motif_tensor = motif_tensor
        self.merged_tensor = merged_tensor
        self.merged_tar = merged_tar
        self.features_df = features_df
        self.RC = RC
        self.keep_output = keep_output
        self.resultObj = Results(name, output_dir)
        self.debug = debug
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.ConvLayerObj = ConvLayer(motif_tensor)
        self.motif_names = motif_names
        self.convObj = ConvPredictor(merged_tensor, merged_tar, self.motif_tensor.shape,
                                     self.ConvLayerObj, self.features_df, RC = self.RC)

    def start_CV(self, CV_dict, CV_algo, **args):
        #CV_algo = self.convObj.LogisticRegCV(**args)
        idx = 0
        start = time.time()
        for chrom in sorted(CV_dict.iterkeys()):
            input_dict = CV_dict[chrom]
            X_train = input_dict['X_train']
            X_test = input_dict['X_test']
            Y_train = input_dict['Y_train']
            Y_test = input_dict['Y_test']
            pos_indices = input_dict['chrom_indices']
            neg_indices = input_dict['other_indices']
            #self.convObj.LogisticRegCVPredict
            if CV_algo == 'LogisticRegCV':
                Y_pred, Y_true = self.convObj.LogisticRegCVPredict(pos_indices, neg_indices,**args)
                reg_weights = self.convObj.LogisticRegCVObj.coef_.T.flatten()
                reg_bias = self.convObj.LogisticRegCVObj.coef_.T.flatten()
            elif CV_algo == 'ElasticNet':
                Y_pred, Y_true = self.convObj.ElasticNet(pos_indices, neg_indices, **args)
                reg_weights = self.convObj.ElasticNetObj.coef_.T.flatten()
                reg_bias = self.convObj.ElasticNetObj.coef_.T.flatten()
            elif CV_algo == 'LogisticReg':
                Y_pred, Y_true = self.convObj.LogisticRegPredict(pos_indices, neg_indices, **args)
                reg_weights = self.convObj.LogisticRegObj.coef_.T.flatten()
                reg_bias = self.convObj.LogisticRegObj.coef_.T.flatten()
            elif CV_algo == 'RandomForest':
                Y_pred, Y_true = self.convObj.RandomForestObj(pos_indices, neg_indices, **args)
                reg_weights = self.convObj.RandomForestObj.feature_importances_
                reg_biase = self.convObj.randomForestObj.feature_importances_
            if np.amax(Y_true) == 0:
                print "Skipping %s because no positive example"
            reg_weights = pd.Series(data=reg_weights, index=np.arange(len(reg_weights)) ,name=self.name + '_' + chrom)
            self.resultObj.add_cv_result(Y_pred, Y_true, chrom, reg_weights, reg_bias, pos_indices)
            idx += 1
            print("Completed lin reg on chromsome %s in %0.04f seconds"%(chrom, time.time() - start))
            if self.debug:
                if idx > 2:
                    break

    def start_CV_NN(self, CV_dict, **args):
        idx = 0
        start = time.time()
        for chrom in sorted(CV_dict.iterkeys()):
            input_dict = CV_dict[chrom]
            X_train = input_dict['X_train']
            X_test = input_dict['X_test']
            Y_train = input_dict['Y_train']
            Y_test = input_dict['Y_test']
            pos_indices = input_dict['chrom_indices']
            neg_indices = input_dict['other_indices']
            #self.convObj.LogisticRegCVPredict
            Y_pred, Y_true = self.convObj.FC_1layer_model(pos_indices, neg_indices,**args)
            if np.amax(Y_true) == 0:
                print "Skipping %s because no positive example"
            reg_weights = np.random.rand(self.motif_tensor.shape[0])
            reg_bias = [0]
            reg_weights = pd.Series(data=reg_weights, index=self.motif_names,
                                    name=self.name + '_' + chrom)
            self.resultObj.add_cv_result(Y_pred, Y_true, chrom, reg_weights,
                                         reg_bias, pos_indices)
            idx += 1
            print("Completed NN on  chromsome %s in %0.04f seconds"
                  %(chrom, time.time() - start))
            if self.debug:
                if idx > 2:
                    break
                
    def summarize(self, dump_indices=False, dump_preds=False, dump_labels=False, dump_weights_bias=False, prefix=''):
        self.resultObj.summarize(prefix=prefix)
        if dump_indices:
            self.resultObj.dump_indices(prefix=prefix)
        if dump_preds:
            self.resultObj.dump_preds(prefix=prefix)
        if dump_labels:
            self.resultObj.dump_true(prefix=prefix)
        if dump_weights_bias:
            self.resultObj.dump_weights_bias(prefix=prefix)

class CrossPlatformEngine(object):
    def __init__(self, TrainConvObj, TargetDataObj):
        self.TrainConvObj = TrainConvObj
        self.TargetDataObj = TargetDataObj
        return 

class Results(object):
    def __init__(self, data_name, output_dir):
        self.data_name = data_name  # ie
        self.per_chrom_results_dict = {}
        self.per_chrom_preds = {}
        self.per_chrom_true = {}
        self.per_chrom_indices = {}
        self.all_probs = np.empty([1, ])
        self.all_true = np.empty([1, ], dtype=bool)
        self.output_dir = output_dir
        self.motif_weights = []
        self.motif_biases = []
        self.column_names = ''

    def ensure_dir(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def add_cv_result(self, pred, true, chrom, motif_weight, motif_bias,
                      indices):
        self.per_chrom_preds[chrom] = pred
        self.per_chrom_true[chrom] = true
        self.all_probs = np.append(self.all_probs.flatten(), pred.flatten())
        self.all_true = np.append(self.all_true.flatten(), true.flatten())
        cur_resultObj = ClassificationResult(true.flatten(), pred.flatten(),
                                             name=self.data_name + '_' + chrom)
        cur_result = cur_resultObj.binary(true, pred)
        cur_str_result = cur_resultObj.convert_to_str(cur_result)
        print(cur_str_result)
        if len(self.column_names) < 1:
            self.column_names = cur_resultObj.convert_keys(cur_result)
        self.per_chrom_results_dict[chrom] = cur_str_result
        self.per_chrom_indices[chrom] = indices
        self.motif_weights.append(motif_weight)
        self.motif_biases.append(motif_bias)

    def cumulative_result(self):
        self.cum_resultObj = ClassificationResult(self.all_true.flatten(),
                                                  self.all_probs.flatten(),
                                                  name=self.data_name + '_cum')
        self.cum_result = self.cum_resultObj.binary(self.all_true,
                                                    self.all_probs)
        self.str_cum_result = self.cum_resultObj.convert_to_str(self.cum_result)

    def summarize(self, output_file_name='', prefix=''):
        self.cumulative_result()
        print(self.str_cum_result)
        if len(output_file_name) > 1:
            self.output_file = self.output_dir + output_file_name
        else:
            self.output_file = self.output_dir + prefix + "_CV_results.txt"
        fo = open(self.output_file, 'w')
        fo.write("Chrom\t%s\n"%(self.column_names))
        fo.write("Combined\t%s\n"%(self.str_cum_result))
        for chrom in sorted(self.per_chrom_results_dict.iterkeys()):
            result_str = self.per_chrom_results_dict[chrom]
            fo.write("%s\t%s\n"%(chrom, result_str))
        fo.close()

    def dump_indices(self, output_dir='', prefix=''):
        cur_output_dir = ''
        if len(output_dir) > 1:
            cur_output_dir = output_dir + "indices/"
        else:
            cur_output_dir = self.output_dir + "indices/"
        self.ensure_dir(cur_output_dir)
        for chrom in sorted(self.per_chrom_indices.iterkeys()):
            indices = self.per_chrom_indices[chrom]
            cur_output_file = (cur_output_dir + chrom + '_' +
                               prefix + '_pos_indices.txt')
            fo = open(cur_output_file, 'w')
            for idx, val in enumerate(indices):
                fo.write("%i\n" % (val))
            fo.close()

    def process_weights_bias(self):
        self.weights_df = pd.DataFrame(self.motif_weights)
        #self.biases_df = pd.DataFrame(self.biases_df)

    def dump_weights_bias(self, output_file_dir='', prefix=''):
        self.process_weights_bias()
        if len(output_file_dir) > 1:
            cur_output_dir = output_file_dir
        else:
            cur_output_dir = self.output_dir + 'weights/'
        self.ensure_dir(cur_output_dir)

        weights_output_file = cur_output_dir + prefix + '_reg_weights.txt'
        bias_output_file = cur_output_dir + prefix + '_bias.txt'
        self.weights_df.to_csv(weights_output_file, sep = "\t")
        fo = open(bias_output_file, 'w')
        for idx, val in enumerate(self.motif_biases):
            fo.write("%0.04f\n"%(val))
        fo.close()

    def dump_preds(self, output_dir='', prefix=''):
        cur_output_dir = ''
        if len(output_dir)>1:
            cur_output_dir = output_dir
        else:
            cur_output_dir = self.output_dir + 'preds/'
        self.ensure_dir(cur_output_dir)

        for chrom in sorted(self.per_chrom_preds.iterkeys()):
            predicted_prob = self.per_chrom_preds[chrom]
            cur_output_file = (cur_output_dir + chrom + '_' + prefix +
                               '_predicted_prob.txt')
            fo = open(cur_output_file, 'w')
            for idx, val in enumerate(predicted_prob):
                fo.write("%0.04f\n"%(val))
            fo.close()

    def dump_true(self, output_dir='', prefix=''):
        cur_output_dir = ''
        if len(output_dir) > 1:
            cur_output_dir = output_dir
        else:
            cur_output_dir = self.output_dir + 'labels/'
        self.ensure_dir(cur_output_dir)

        for chrom in sorted(self.all_true.iterkeys()):
            true = self.all_true[chrom]
            cur_output_file = (cur_output_dir + chrom + '_' +
                               prefix + '_labels.txt')
            fo = open(cur_output_file, 'w')
            for idx, val in enumerate(true):
                fo.write("%0.04f\n" % (val))
            fo.close()
