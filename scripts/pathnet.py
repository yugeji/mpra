import os
import sys
import numpy as np
import copy
import matplotlib.pyplot as plt
import pandas as pd
import time
import pickle
import theano
from collections import OrderedDict
from sklearn.metrics import auc, log_loss, precision_recall_curve, roc_auc_score, roc_curve
from prg.prg import create_prg_curve, calc_auprg
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
from keras.layers import Input, Dense, Flatten, merge
from keras.engine.topology import Layer, Container
from keras import activations, regularizers, constraints
import keras
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.pooling import MaxPooling1D
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

def calculate_binary_weights(y):
    pos = np.sum(y)
    len_y = len(y.flatten())
    neg = len_y - pos
    return {0: len_y/float(neg), 1: len_y/float(pos)}

class AbstractPathLayer(object):
    def __init__(self, M, name):
        self.M = M 
        self.modules = []
        self.name = name
        self.selected = []

    def select(self, indices):
        self.selected = [self.modules[i] for i in indices]

    def fix(self, indices):
        for i in indices:
            self.modules[i].trainable = False

class ConvolutionPathLayer(AbstractPathLayer):
    def build(self, filters=15, nb_row=4, nb_col=18, L2 = 0.001):
        self.filters = filters
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.L2 = L2
        for i in range(self.M):
            name = "Conv_%i"%(i)
            self.modules.append(Conv2D(filters=filters, nb_row=nb_row,
                                              nb_col=nb_col, activation='relu',
                                              W_regularizer=l2(L2)))
    def rebuild(self):
        for i in range(self.M):
            if self.modules[i].trainable:
                name = "Conv_%i"%(i)
                self.modules[i] = Conv2D(filters=self.filters, nb_row=self.nb_row,
                                                nb_col=self.nb_col, activation='relu',
                                                W_regularizer=l2(self.L2))

    def call(self, Input, pool_width=4):
        #declare auxiliary layers
        pool_size = (1, int(pool_width))
        max_pool_layer = MaxPooling2D(pool_size=pool_size)
        flatten_layer = Flatten()

        intermediate = [module(Input) for module in self.selected]
        max_out = [max_pool_layer(o) for o in intermediate]
        flat_out = [flatten_layer(o) for o in max_out]
        merge_out = merge(flat_out, mode='sum')
        return merge_out

    def call_RC(self, Input, Input_RC, pool_width=4):
        pool_size = (1, int(pool_width))
        max_pool_layer = MaxPooling2D(pool_size=pool_size)
        flatten_layer = Flatten()

        intermediate = [module(Input) for module in self.selected]
        intermediate_RC = [module(Input_RC) for module in self.selected]
        max_out = [max_pool_layer(o) for o in intermediate]
        max_out_RC = [max_pool_layer(o) for o in intermediate_RC]
        flat_out = [flatten_layer(o) for o in max_out]
        flat_out_RC = [flatten_layer(o) for o in max_out_RC]
        merge_RC = [merge([a,b], mode='sum') for a,b in zip(flat_out, flat_out_RC)]
        merge_out = merge(merge_RC, mode='sum')
        return merge_out


class DensePathLayer(AbstractPathLayer):
    def build(self, units=5, L2=0.001):
        self.units = units
        self.L2 = L2
        for i in range(self.M):
            self.modules.append(Dense(units, activation='relu', W_regularizer=l2(L2)))

    #rebuild the layers that are non-trainable 
    def rebuild(self):
        for i in range(self.M):
            if self.modules[i].trainable:
                self.modules[i] = Dense(self.units, activation='relu', W_regularizer=l2(self.L2))

    def call(self, Input):
        intermediate = [module(Input) for module in self.selected]
        merge_out = merge(intermediate, mode='sum')
        return merge_out


class MotifPathLayer(AbstractPathLayer):
    def build(self):
        pass


class PathNet(object):
    def __init__(self, N=4, M=12, tasks=2, input_shape=(1, 4, 150),
                 pool_width=4, lr=0.00001, filters=40, L2=0.0001, units=4,
                 early_stopping=True, val_split=0.1,
                 patience = 5,
                 fix_path = True,
                 task_outputs = None,
                 debug=False):
        self.M = M
        self.N = N  # max number of modules allowed
        self.filters = filters
        self.L2 = L2
        self.pool_width = pool_width
        self.units = units  # FC units
        self.input_shape = input_shape
        self.lr = lr
        self.round = 0
        self.debug = debug
        self.tasks = tasks
        self.current_task = 0
        self.early_stopping = early_stopping
        self.val_split = val_split
        self.patience = patience
        self.fix_path = fix_path
        # Declare Layers
        self.ConvPathObj = ConvolutionPathLayer(M=M, name="ConvPath")
        self.DensePathObj = DensePathLayer(M=M, name="DensePath")
        self.ConvPathObj.build(filters=self.filters, L2=self.L2)
        self.DensePathObj.build(units=self.units, L2=self.L2)
        self.Layers = [self.ConvPathObj, self.DensePathObj]
        self.L = len(self.Layers)
        self.task_final_layers = []
        self.task_outputs = task_outputs
        if self.tasks is not None:
            for i in range(self.tasks):
                self.task_final_layers.append(Dense(1, activation='sigmoid'))
        else:
            for i in range(self.tasks):
                self.tasks_final_layers.append(Dense(task_outputs[i],
                                                     activation='sigmoid'))

        # Declare genotypes
        self.best_genotype = self.generate_genotype(self.L, self.N, self.M)#
        self.challenger_genotype = self.generate_genotype(self.L, self.N, self.M)#

        # Declare models
        self.best_model = None
        self.challenger_model = None

        #Declare overall training history
        self.training_history = {}

    def build_model_from_genotype(self, genotype, task_id=0):
        for i in range(genotype.shape[0]):
            selection_indices = list(genotype[i,:])
            self.Layers[i].select(selection_indices)

        #start building the model
        X = Input(shape=self.input_shape, name="Seq_Input")
        X_RC = Input(shape=self.input_shape, name="RC_Input")
        conv_out = self.Layers[0].call_RC(X, X_RC, pool_width=self.pool_width)
        fc_out = self.Layers[1].call(conv_out)
        p_out = self.task_final_layers[self.current_task](fc_out)
        optimizer = keras.optimizers.Adam(lr=self.lr)
        model = keras.models.Model(inputs=[X, X_RC], outputs=p_out)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_crossentropy'])
        return model

    def generate_genotype(self, L, N, M):
        g = np.zeros((L,N), dtype = int)
        for i in range(L):
            g[i,:] = np.random.choice(M, N, replace=False)
        return g

    def mutate(self, genotype):
        #probability is self.N * self.L
        cutoff = 1/float(self.N * self.L)
        for i in range(genotype.shape[0]):
            for j in range(genotype.shape[1]):
                if np.random.rand() < cutoff:
                    proposed = genotype[i,j] + np.random.randint(-2,2)
                    if proposed > self.M-1: #not sure what to do when we go above M, do we wrap arond for symmetry?
                        proposed = abs(proposed-self.M-1)
                    elif proposed < 0: #same as above
                        proposed = self.M+proposed
                    while proposed in genotype[i,:] and proposed >= 0: #we don't want to have duplicate entires
                        proposed = proposed + np.random.randint(-1,1)
                    genotype[i,j] = proposed
        genotype = np.clip(genotype,0,self.M-1)
        return genotype

    def challenge(self, X, X_RC, y, rounds=25, epochs=50, batch_size=5000, test_dict=None):
        self.X_train, self.X_test, self.X_RC_train, self.X_RC_test, self.y_train, self.y_test = train_test_split(X,X_RC,y, test_size=0.15, random_state=42)
        #protocol is:
        #1: generate new models
        #2: get predicton/fitness 
        #3: compare and keep
        #4: mutate winner and generate new challenger
        self.weights = calculate_binary_weights(self.y_train)
        self.results_dict={}
        if self.early_stopping:
            self.callbacks = []
            self.EarlyStoppingCallBack = keras.callbacks.EarlyStopping('val_loss', patience= self.patience)
            self.callbacks.append(self.EarlyStoppingCallBack)

        start = time.time()
        for i in range(rounds):
            self.best_model = self.build_model_from_genotype(self.best_genotype)
            self.challenger_model = self.build_model_from_genotype(self.challenger_genotype)
            if self.early_stopping:
                th = self.best_model.fit(x=[self.X_train, self.X_RC_train],
                                         y=self.y_train, batch_size=batch_size,
                                         epoch=epochs,
                                         class_weight=self.weights,
                                         callbacks=self.callbacks,
                                         validation_split=self.val_split,
                                         shuffle=True,
                                         verbose=False)
                ch = self.challenger_model.fit(x=[self.X_train, self.X_RC_train],
                                               y=self.y_train,
                                               batch_size=batch_size,
                                               epoch=epochs,
                                               class_weight=self.weights,
                                               callbacks=self.callbacks,
                                               validation_split=self.val_split,
                                               shuffle=True,
                                               verbose=False)
            else:
                th = self.best_model.fit(x=[self.X_train, self.X_RC_train], y=self.y_train,
                                         batch_size=batch_size, epoch=epochs,
                                         class_weight=self.weights, verbose=False)
                ch = self.challenger_model.fit(x=[self.X_train, self.X_RC_train], y=self.y_train,
                                               batch_size=batch_size,
                                               epoch=epochs,
                                               class_weight=self.weights,verbose=False)
            y_pred_best = self.best_model.predict([self.X_test, self.X_RC_test])
            y_pred_challenger = self.challenger_model.predict([self.X_test, self.X_RC_test])
            best_auROC = roc_auc_score(self.y_test.astype(bool), y_pred_best.flatten())
            challenger_auROC = roc_auc_score(self.y_test.astype(bool), y_pred_challenger.flatten())
            self.results_dict[i] = {'best_auroc': best_auROC, 'challenger_auROC': challenger_auROC,
                               'challenger_genotype':self.challenger_genotype,
                               'best_genotype':self.best_genotype}
            if test_dict is not None:
                best_test_auROC = roc_auc_score(test_dict['Y'].astype(bool), self.best_model.predict([test_dict['X'], test_dict['X_RC']]))
                best_challenger_auROC = roc_auc_score(test_dict['Y'].astype(bool), self.challenger_model.predict([test_dict['X'], test_dict['X_RC']]))
                self.results_dict[i]['best_test_auROC'] = best_test_auROC
                self.results_dict[i]['challenger_test_auROC'] = best_challenger_auROC
            #if challenger is better
            if challenger_auROC > best_auROC:
                self.best_genotype = self.challenger_genotype
                self.best_model = self.challenger_model

            #regardless of who wins, mutate the best genotype and generate a new challenger genotype 
            self.best_genotype = self.mutate(self.best_genotype)
            self.best_genotype = np.clip(self.best_genotype, 0, self.M-1)
            assert np.min(self.best_genotype) >= 0, "Negative \
            genotype detected!"
            self.challenger_genotype = self.generate_genotype(self.L, self.N,
                                                              self.M)

            if self.debug:
                print "Iteration %i done in %0.04f"%(i, time.time()-start)
                print self.results_dict[i]

            if i >= 20 and self.debug: 
                return self.results_dict

        return self.results_dict

    def switch_tasks(self, task):
        for i in range(self.L):
            self.Layers[i].fix(self.best_genotype[i,:]) #fix best genotype
            self.Layers[i].rebuild()

        self.current_task = task
        self.training_history[task-1] = self.results_dict

class History(object):
    def __init__(self):
        pass

class SimpleCNNModel(object):
    def __init__(self, X, X_RC, debug=False):
        self.X_input = X
        self.debug = debug
        self.motif_shape = (200,1,4,18)

    def build_model(self, dropout=0.01, L1=0, L2=0.0001, units = 2, pool_width=10, lr=0.0001, merge_mode='sum', conv_filters=160, extra_conv=False, weighted=True):
        self.X = Input(shape=self.X_input.shape[1:], name="Input")
        self.X_RC = Input(shape=self.X_input.shape[1:], name="RC Input")
        self.conv_layer = Conv2D(filters=conv_filters, kernel_size=(4, 18), activation=relu, W_regularizer=l1_l2(L1,L2)) 
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
            self.X_merged_out = Conv2D(filters=15, kernel_size=(1,5), activation='relu', W_regularizer=l1_l2(L1,L2))(self.X_merged)
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

    def train(self, X, X_RC, y, neg_weight, pos_weight, epochs=60, batch_size=2000, early_stopping=True, patience = 10, outside_eval=None, val_split=0.10, save=False, verbose=0):
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
            history = self.model.fit(x=[X, X_RC], y=y, batch_size = batch_size, epochs = epochs, callbacks=callbacks,validation_data=outside_eval, shuffle=True, class_weight={0:neg_weight, 1:pos_weight}, verbose=verbose)
        else:
            history = self.model.fit(x=[X, X_RC], y=y, batch_size = batch_size, epochs = epochs, callbacks=callbacks,validation_split=val_split, shuffle=True, class_weight={0:neg_weight, 1:pos_weight}, verbose=verbose)
        return history

    def predict(self, X, X_RC, batch_size=5000, verbose = 0):
        return self.model.predict([X, X_RC], batch_size=batch_size)
