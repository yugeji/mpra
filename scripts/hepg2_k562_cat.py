from __future__ import absolute_import, division, print_function
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers.core import (
    Activation, Dense, Dropout, Flatten,
    Permute, Reshape, TimeDistributedDense
)
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import GRU
from keras.regularizers import l1, l2, l1l2
from keras.utils import np_utils
from keras import backend as K
import numpy as np
import copy
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import time
import re
import numpy as np
from collections import OrderedDict
from sklearn.metrics import auc, log_loss, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from prg.prg import create_prg_curve, calc_auprg
from sklearn.grid_search import ParameterGrid
from keras.utils.np_utils import to_categorical

class ClassificationResult(object):
    def __init__(self, labels, predictions, name=None):
        self.predictions = predictions
        self.labels = labels
        self.flat_predictions = predictions.flatten()
        self.flat_labels = labels.flatten()
        self.results = []

    def self_binary(self, labels, predictions):
        self.results = self.binary(self.labels, self.predictions)

    def cat_binary(self, labels, predictions):
        class1_results = self.binary(self.labels[:,0].astype(bool), self.predictions[:,0])
        class2_results = self.binary(self.labels[:,1].astype(bool), self.predictions[:,1])
        class3_results = self.binary(self.labels[:,2].astype(bool), self.predictions[:,2])
        print("Class 1 Performance\n" + self.convert_to_str(class1_results))
        print("Class 2 Performance\n" + self.convert_to_str(class2_results))
        print("Class 3 Performance\n" + self.convert_to_str(class3_results))


    def binary(self, labels, predictions):
        def loss(labels, predictions):
            return log_loss(labels, predictions)


        def positive_accuracy(labels, predictions, threshold=0.5):
            return 100 * (predictions[labels] > threshold).mean()


        def negative_accuracy(labels, predictions, threshold=0.5):
            return 100 * (predictions[~labels] < threshold).mean()


        def balanced_accuracy(labels, predictions, threshold=0.5):
            return (positive_accuracy(labels, predictions, threshold) +
                    negative_accuracy(labels, predictions, threshold)) / 2


        def auROC(labels, predictions):
            return roc_auc_score(labels, predictions)


        def auPRC(labels, predictions):
            precision, recall = precision_recall_curve(labels, predictions)[:2]
            return auc(recall, precision)


        def auPRG(labels, predictions):
            return calc_auprg(create_prg_curve(labels, predictions))


        def recall_at_precision_threshold(labels, predictions, precision_threshold):
            precision, recall = precision_recall_curve(labels, predictions)[:2]
            return 100 * recall[np.searchsorted(precision - precision_threshold, 0)]

        results = [
            ('Loss', loss(labels, predictions)),
            ('Balanced_accuracy', balanced_accuracy(
                labels, predictions)),
            ('auROC', auROC(labels, predictions)),
            ('auPRC', auPRC(labels, predictions)),
            ('auPRG', auPRG(labels, predictions)),
            ('Recall_at_5%_FDR', recall_at_precision_threshold(
                labels, predictions, 0.95)),
            ('Recall_at_10%_FDR', recall_at_precision_threshold(
                labels, predictions, 0.9)),
            ('Recall_at_20%_FDR', recall_at_precision_threshold(
                labels, predictions, 0.8)),
            ('Num_Positives', labels.sum()),
            ('Num_Negatives', (1 - labels).sum())]
        return results

    def continuous(self):
        mse = self.MSE()
        ase = self.ASE() 
        self.results = [('MSE', mse), ('ASE', ase)]

    def MSE(self):
        return np.mean((self.flat_predictions - self.flat_labels)**2)

    def ASE(self):
        return np.mean(np.abs(self.flat_predictions - self.flat_labels))

    def __str__(self):
        strs = []
        for idx, (key, val) in enumerate(self.results):
            _str = "%0.04f"%(val)
            strs.append(_str)
        return "\t".join(strs)

    def convert_to_str(self, results):
        strs = []
        for idx, (key, val) in enumerate(results):
            _str = "%0.04f"%(val)
            strs.append(_str)
        return "\t".join(strs)

    def __getitem__(self, item):
        return np.array([task_results[item] for task_results in self.results])

class Initalizer(object):
    def __init__(self, dna_file, target_file, name):
        self.dna_file = dna_file
        self.target_file = target_file
        self.N = sum(1 for line in open(target_file))
        f = open(self.dna_file, 'r')
        line = f.readline()
        split_line = line.rstrip().split()
        self.W = 4
        self.H = len(split_line)
        self.name = name
        f.close()

    def prepare(self):
        self.prep_dna()
        self.prep_tar()
        print("Preparation of input matrices complete")

    def prep_dna(self):
        idx = 0 
        self.dna_tensor = np.zeros((self.N, 1, self.W, self.H))
        f = open(self.dna_file, 'r')
        for line in f:
            split_line = line.rstrip().split()
            _N, _W = divmod(idx, 4)
            if len(split_line) < 145:
                print("Illegal line at %i"%(idx))
                continue
            self.dna_tensor[_N, 0, _W, :] = np.asarray(split_line, dtype = int)
            idx += 1

    def prep_tar(self):
        idx = 0
        self.tar_vec = np.zeros(self.N)
        f = open(self.target_file, 'r')
        for line in f:
            split_line = line.rstrip().split()
            self.tar_vec[idx] = int(split_line[0])
            idx += 1

    def return_test_train_3class(self, test_size = 0.2):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
        self.converted_cat = to_categorical(self.tar_vec)
        for train_index, test_index in sss.split(self.dna_tensor, self.tar_vec):
            X_train, X_test = self.dna_tensor[train_index], self.dna_tensor[test_index]
            Y_train, Y_test = self.converted_cat[train_index], self.converted_cat[test_index]
            tar_train, tar_test = self.tar_vec[train_index], self.tar_vec[test_index]
        return X_train, X_test, Y_train, Y_test, tar_train, train_test_split

    def return_test_train_2class(self, test_size = 0.2):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
        for train_index, test_index in sss.split(self.dna_tensor, self.tar_vec):
            X_train, X_test = self.dna_tensor[train_index], self.dna_tensor[test_index]
            Y_train, Y_test = self.tar_vec[train_index], self.tar_vec[test_index]
        return X_train, X_test, Y_train, Y_test


    def get_weight_3class(self, tar_vec):
        counts = np.zeros(3)
        for idx, val in enumerate(self.tar_vec):
            if val == 0:
                counts[0] += 1
            elif val == 1: 
                counts[1] += 1
            elif val == 2:
                counts[2] += 1
        _sum = np.sum(counts)
        return _sum / float(counts[0]), _sum / float(counts[1]), _sum / float(counts[2])

    def get_weight_2class(self, tar_vec):
        counts = np.zeros(2)
        for idx, val in enumerate(self.tar_vec):
            if val == 0:
                counts[0] += 1
            elif val == 1: 
                counts[1] += 1
        _sum = np.sum(counts)
        return _sum / float(counts[0]), _sum / float(counts[1])


class CatModel(object):
    #some copy and paste from dragonn @ https://github.com/kundajelab/dragonn/blob/master/dragonn/models.py
    def __init__(self, verbose = 0, seq_length = 150, name = ""):
        self.verbose = verbose 
        self.seq_length = seq_length
        self.input_shape = (1,4,self.seq_length)
        self.name = ""

    def model(self, num_filters=(30, 15), conv_width=(15, 10),
                pool_width=4, L1=0.01, L2 = 0.01, nb_classes = 3, dropout=0.01, GRU_size=35, TDD_size=15, use_RNN = False):
        self.seq_length = self.seq_length
        self.input_shape = (1,4, self.seq_length)
        self.num_tasks = 1
        self.model = Sequential()
        self.num_filters = num_filters
        self.conv_width = conv_width
        self.nb_classes = nb_classes
        assert len(num_filters) == len(conv_width)
        for i, (nb_filter, nb_col) in enumerate(zip(num_filters, conv_width)):
            conv_height = 4 if i == 0 else 1
            self.model.add(Convolution2D(
                nb_filter=nb_filter, nb_row=conv_height,
                nb_col=nb_col, activation='linear',
                init='he_normal', input_shape=self.input_shape,
                W_regularizer=l1l2(L1,L2), b_regularizer=l1l2(L1,L2)))
            # self.model.add(Convolution2D(
            #     nb_filter=nb_filter, nb_row=conv_height,
            #     nb_col=nb_col, activation='linear',
            #     init='he_normal', input_shape=self.input_shape,
            #     W_regularizer=l1(L1), b_regularizer=l1(L1)))
            self.model.add(Activation('relu'))
            self.model.add(Dropout(dropout))
            #self.model.add(MaxPooling2D(pool_size=(1, pool_width)))
        self.model.add(MaxPooling2D(pool_size=(1, pool_width)))
        self.model.add(Flatten())
        self.model.add(Dense(36), W_regularizer=l1l2(L1, L2), b_regularizer=l1l2(L1, L2))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(output_dim=3))
        self.model.add(Activation('softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')


    def train(self, X, y, neg_weight, neutral_weight, pos_weight, batch_size = 10000, num_epochs = 100):
        #minimum do enough epochs to span entire set 
        self.model.fit(X, y, batch_size=batch_size, nb_epoch = num_epochs,verbose = self.verbose, class_weight= {0:neg_weight, 1:neutral_weight, 2:pos_weight})

    def get_sequence_filters(self):
        """
        Returns 3D array of 2D sequence filters.
        """
        return self.model.layers[0].get_weights()[0].squeeze(axis=1)

    def predict(self, X):
        return self.model.predict(X, batch_size=128, verbose=self.verbose)

    def test(self, X, y):
        return ClassificationResult(y, self.predict(X))

    def score(self, X, y, metric):
        return self.test(X, y)



class BinaryModel(object):
    #some copy and paste from dragonn @ https://github.com/kundajelab/dragonn/blob/master/dragonn/models.py
    def __init__(self, verbose = 0, seq_length = 150, name = ""):
        self.verbose = verbose 
        self.seq_length = seq_length
        self.input_shape = (1,4,self.seq_length)
        self.name = ""

    def model(self, num_filters=(16, 12), conv_width=(14, 10),
                pool_width=12, L1=0.0001, L2 = 0.01, nb_classes = 1, dropout=0, GRU_size=35, TDD_size=15, use_RNN = False):
        self.seq_length = self.seq_length
        self.input_shape = (1,4, self.seq_length)
        self.num_tasks = 1
        self.model = Sequential()
        self.num_filters = num_filters
        self.conv_width = conv_width
        self.nb_classes = nb_classes
        assert len(num_filters) == len(conv_width)
        for i, (nb_filter, nb_col) in enumerate(zip(num_filters, conv_width)):
            conv_height = 4 if i == 0 else 1
            self.model.add(Convolution2D(
                nb_filter=nb_filter, nb_row=conv_height,
                nb_col=nb_col, activation='linear',
                init='he_normal', input_shape=self.input_shape,
                W_regularizer=l1l2(L1,L2), b_regularizer=l1l2(L1,L2)))
            # self.model.add(Convolution2D(
            #     nb_filter=nb_filter, nb_row=conv_height,
            #     nb_col=nb_col, activation='linear',
            #     init='he_normal', input_shape=self.input_shape,
            #     W_regularizer=l1(L1), b_regularizer=l1(L1)))
            self.model.add(Activation('relu'))
            self.model.add(Dropout(dropout))
            #self.model.add(MaxPooling2D(pool_size=(1, pool_width)))
        self.model.add(MaxPooling2D(pool_size=(1, pool_width)))
        self.model.add(Flatten())
        #self.model.add(Dense(12 , W_regularizer=l1l2(L1, L2), b_regularizer=l1l2(L1, L2)))
        #self.model.add(Activation('relu'))
        #self.model.add(Dropout(dropout))
        self.model.add(Dense(output_dim=1))
        self.model.add(Activation('sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy')

    def model2(self, num_filters= 300, conv_width = 12, pool_width = 9, L1 = 0.0001, L2 = 0.01, nb_classes = 1, dropout = 0):
        self.seq_length = self.seq_length
        self.input_shape = (1,4, self.seq_length)
        self.num_tasks = 1
        self.model = Sequential()
        self.num_filters = num_filters
        self.conv_width = conv_width
        self.nb_classes = nb_classes
        self.model.add(Convolution2D(
                nb_filter=num_filters, nb_row=4,
                nb_col=conv_width, activation='linear',
                init='he_normal', input_shape=self.input_shape,
                W_regularizer=l1l2(L1,L2), b_regularizer=l1l2(L1,L2)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(dropout)) 
            #self.model.add(MaxPooling2D(pool_size=(1, pool_width)))
        self.model.add(MaxPooling2D(pool_size=(1, pool_width)))
        self.model.add(Flatten())
        self.model.add(Dense(18, W_regularizer=l1l2(L1, L2), b_regularizer=l1l2(L1, L2)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(output_dim=1))
        self.model.add(Activation('sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy')        


    def train(self, X, y, neg_weight, pos_weight, batch_size = 10000, num_epochs = 100, validation_data = None):
        #minimum do enough epochs to span entire set 
        self.model.fit(X, y, batch_size=batch_size, nb_epoch = num_epochs,verbose = self.verbose, class_weight= {0:neg_weight, 1:pos_weight}, validation_data = validation_data)

    def get_sequence_filters(self):
        """
        Returns 3D array of 2D sequence filters.
        """
        return self.model.layers[0].get_weights()[0].squeeze(axis=1)

    def predict(self, X):
        return self.model.predict(X, batch_size=128, verbose=self.verbose)

    def test(self, X, y):
        return ClassificationResult(y, self.predict(X))

    def score(self, X, y, metric):
        return self.test(X, y)
