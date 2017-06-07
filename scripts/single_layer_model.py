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
from sklearn.metrics import auc, log_loss, precision_recall_curve, roc_auc_score
from prg.prg import create_prg_curve, calc_auprg
from theano import pp
from sklearn.grid_search import ParameterGrid
from sklearn.preprocessing import scale, StandardScaler 
np.random.seed(42)
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers.core import (
    Activation, Dense, Dropout, Flatten,
    Permute, Reshape, TimeDistributedDense
)
import keras
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import GRU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2, l1l2
from keras.utils import np_utils
from keras import backend as K
from sklearn.linear_model import SGDClassifier

class CrossValProcessor(object):
    def __init__(self, data_sets = 'all'):
        self.dna_files = ["/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/HEPG2_act_mpra_dna.txt", "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/HEPG2_rep_mpra_dna.txt", "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/K562_act_mpra_dna.txt", "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/K562_rep_mpra_dna.txt", "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/LCL/LCL_act_mpra_dna.txt"]
        self.target_files = ["/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/HEPG2_act_mpra_tar.txt", "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/HEPG2_rep_mpra_tar.txt", "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/K562_act_mpra_tar.txt", "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/K562_rep_mpra_tar.txt", "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/LCL/LCL_act_mpra_tar.txt"]
        self.anno_files = ["/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/HEPG2_act_mpra_det.txt", "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/HEPG2_rep_mpra_det.txt", "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/K562_act_mpra_det.txt", "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/K562_rep_mpra_det.txt", "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/LCL/LCL_act_mpra_det.txt"]
        self.output_dirs = ["./HEPG2_act/", "./HEPG2_rep/", "./K562_act/", "./K562_rep/", "./LCL_act/"]
        self.names = ["HEPG2_act", "HEPG2_rep", "K562_act", "K562_rep", "LCL_act"]
        self.status = ["act", "rep", "act", "rep", "act"]
        if data_sets == 'all':
            self.HepG2_act_index = 0
            self.K562_act_index = 2
            self.LCL_act_index = 4
            self.HepG2_dict, self.HepG2_binObj = self.process_single_dataset(self.HepG2_act_index)
            self.K562_dict, self.K562_binObj = self.process_single_dataset(self.K562_act_index)
            self.LCL_dict, self.LCL_binObj = self.process_single_dataset(self.LCL_act_index, LCL = True)

        elif data_sets == 'HepG2':
            self.HepG2_dict, self.HepG2_binObj = self.process_single_dataset(self.HepG2_act_index)
        elif data_sets == 'K562':
            self.K562_dict, self.K562_binObj = self.process_single_dataset(self.K562_act_index)
        elif data_sets == 'LCL':
            self.LCL_dict, self.LCL_binObj = self.process_single_dataset(self.LCL_act_index)

    def process_single_dataset(self, cur_index, LCL = False):
        cur_DNA = [self.dna_files[cur_index]]
        cur_target = [self.target_files[cur_index]]
        cur_anno_files = [self.anno_files[cur_index]]
        cur_output_dirs = [self.output_dirs[cur_index]]
        cur_name = [self.names[cur_index]]
        cur_status = [self.status[cur_index]]
        cur_BinObj = BinaryClassProcessor(cur_DNA, cur_target, cur_anno_files, cur_status, LCL = LCL)
        cur_dict = cur_BinObj.generate_CV_dict(cur_BinObj.anno_dfs[0])
        return cur_dict, cur_BinObj

class BinaryClassProcessor(object):
    def __init__(self, dna_files, target_files, anno_files, status,
                 LCL = False, type = "bin"):
        self.dna_files = dna_files
        self.target_files = target_files
        self.anno_files = anno_files
        self.status = status
        self.LCL = LCL
        self.type = "bin"
        self.process_all_files()

    def change_tensor(self, tensor):
        mat = np.zeros((tensor.shape[0], tensor.shape[3]*tensor.shape[2]))
        for idx in range(tensor.shape[0]):
            current_dna = tensor[idx,0,:,:]
            flattened_dna = current_dna.flatten()
            mat[idx,:]=flattened_dna
        return mat

    def transform_array_theano(self, array):
        return np.asarray(array, dtype = theano.config.floatX)

    def return_train_test_chrom(self, chrom):
        chrom_indices, other_indices = self.find_chr_by_color(self.chroms,
                                                              chrom)
        X_train = self.transform_array_theano(self.merged_tensor[other_indices])
        Y_train = np.asarray(self.merged_tar[other_indices], dtype = 'int32')
        X_test = self.transform_array_theano(self.merged_tensor[chrom_indices])
        Y_test = np.asarray(self.merged_tar[chrom_indices], dtype = 'int32')
        self.shuffle_array = np.arange(X_train.shape[0])
        np.random.shuffle(self.shuffle_array)
        return X_train[self.shuffle_array], X_test, Y_train[self.shuffle_array], Y_test, chrom_indices, other_indices
    #The optimal running order is
    #process_all_files
    #color_chroms
    #return_train_test_chrom on desired chromosome
    #collapse tensors to Nx600 vectors and feed into deep neural net.

    #given the set of all chromosomes, and a target chromosome, select which indices to pick as training and test set
    def find_chr_by_color(self, chromlist, chrom):
        chrom_indices = []#chrom_indiceds 
        other_indices = []
        for idx, val in enumerate(chromlist):
            if val == chrom:
                chrom_indices.append(idx)
            else:
                #this is a test comment
                other_indices.append(idx)
        return chrom_indices, other_indices

    #process annotation files to grab the necessary chromosomal values
    def color_chroms(self, anno_df):
        chrom_array = []
        for idx, val in enumerate(list(anno_df.iloc[:, 0])):
            chrom_array.append(val)
        return chrom_array

    def generate_CV_dict(self, anno_df):
        chrom_names = []
        for idx in range(1,23):
            chrom_names.append('chr' + str(idx))
        if self.LCL == False:
            chrom_names.append('chrX')
        chrom_dict = {}
        for chrom_index, chrom_str in enumerate(chrom_names):
            X_train, X_test, Y_train, Y_test, chrom_indices, other_indices = self.return_train_test_chrom(chrom_str)
            chrom_dict[chrom_str] = {}
            chrom_dict[chrom_str]['X_train'] = X_train
            chrom_dict[chrom_str]['X_test'] = X_test
            chrom_dict[chrom_str]['Y_train'] = Y_train
            chrom_dict[chrom_str]['Y_test'] = Y_test
            chrom_dict[chrom_str]['chrom_indices'] = chrom_indices
            chrom_dict[chrom_str]['other_indices'] = other_indices
        return chrom_dict

    def process_all_files(self):
        self.tar_vecs = []
        self.dna_tensors = []
        self.anno_dfs = []
        self.balance = [0, 0]
        self.chroms = []
        for idx in range(len(self.dna_files)):
            anno_df, dna_tensor, tar_vec, cur_balance = self.process_one_file(
                idx)
            self.tar_vecs.append(tar_vec)
            self.anno_dfs.append(anno_df)
            self.dna_tensors.append(dna_tensor)
            self.chroms += self.color_chroms(anno_df)
            self.balance[0] += cur_balance[0]
            self.balance[1] += cur_balance[1]
        self.merged_tensor = np.concatenate(self.dna_tensors)
        self.merged_tar = np.concatenate(self.tar_vecs)
        self.bin_merged_tar = self.merged_tar

    def process_one_file(self, idx):
        dna_file = self.dna_files[idx]
        target_file = self.target_files[idx]
        anno_file = self.anno_files[idx]
        anno_df, N = self.prep_anno(anno_file)
        status = self.status[idx]
        dna_tensor = self.prep_dna(dna_file, N)
        tar_vec, balance = self.prep_tar(target_file, N, status, self.type)
        return anno_df, dna_tensor, tar_vec, balance

    def prep_anno(self, anno_file):
        anno_df = pd.read_csv(anno_file, sep="\t", header=0)
        N = len(anno_df.index)
        return anno_df, N

    def prep_dna(self, dna_file, N):
        idx = 0
        dna_tensor = np.zeros((N, 1, 4, 150))
        f = open(dna_file, 'r')
        for line in f:
            split_line = line.rstrip().split()
            _N, _W = divmod(idx, 4)
            dna_tensor[_N, 0, _W, :] = np.asarray(split_line, dtype=int)
            idx += 1
        return dna_tensor

    def prep_tar(self, tar_file, N, status, type="tanh"):
        idx = 0
        tar_vec = np.zeros(N)
        f = open(tar_file, 'r')
        balance = [0, 0]
        for line in f:
            split_line = line.rstrip().split()
            if type == "tanh":
                if status == "act":
                    val = int(split_line[0])
                    tar_vec[idx] = val
                    if int(val) == 1:
                        balance[1] += 1
                    elif int(val) == 0:
                        balance[0] += 1

                elif status == "rep":
                    val = int(split_line[0])
                    if val == 1:
                        tar_vec[idx] = 1
                        balance[1] += 1
                    elif val == 0:
                        tar_vec[idx] = 0
                        balance[0] += 1

            elif type == "bin":
                if status == "act":
                    val = int(split_line[0])
                    tar_vec[idx] = val
                    if int(val) == 1:
                        balance[1] += 1
                    elif int(val) == 0:
                        balance[0] += 1

                elif status == "rep":
                    val = int(split_line[0])
                    if val == 1:
                        tar_vec[idx] = 1
                        balance[1] += 1
                    elif val == 0:
                        tar_vec[idx] = 0
                        balance[0] += 1
            idx += 1
        return tar_vec, balance

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
        class1_results = self.binary(self.labels[:, 0].astype(bool),
                                     self.predictions[:, 0])
        class2_results = self.binary(self.labels[:, 1].astype(bool),
                                     self.predictions[:, 1])
        class3_results = self.binary(self.labels[:, 2].astype(bool),
                                     self.predictions[:, 2])
        print("Class 1 Performance\n" + self.convert_to_str(class1_results))
        print("Class 2 Performance\n" + self.convert_to_str(class2_results))
        print("Class 3 Performance\n" + self.convert_to_str(class3_results))

    def binary(self, labels, predictions):
        def loss(labels, predictions):
            try:
                loss = log_loss(labels, predictions)
                return loss
            except ValueError:
                return 0

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

        def recall_at_precision_threshold(labels, predictions,
                                          precision_threshold):
            precision, recall = precision_recall_curve(labels, predictions)[:2]
            return 100 * recall[np.searchsorted(precision -
                                                precision_threshold, 0)]

        results = [('Loss', loss(labels, predictions)), (
            'Balanced_accuracy', balanced_accuracy(
                labels, predictions)), ('auROC', auROC(labels, predictions)),
                   ('auPRC', auPRC(labels, predictions)),
                   ('auPRG', auPRG(labels, predictions)), (
                       'Recall_at_5%_FDR', recall_at_precision_threshold(
                           labels, predictions, 0.95)),
                   ('Recall_at_10%_FDR',
                    recall_at_precision_threshold(labels, predictions, 0.9)), (
                        'Recall_at_20%_FDR', recall_at_precision_threshold(
                            labels, predictions,
                            0.8)), ('Num_Positives', labels.sum()), (
                                'Num_Negatives', (1 - labels).sum())]
        return results

    def continuous(self):
        mse = self.MSE()
        ase = self.ASE()
        self.results = [('MSE', mse), ('ASE', ase)]

    def MSE(self):
        return np.mean((self.flat_predictions - self.flat_labels)**2)

    def ASE(self):
        return np.mean(np.abs(self.flat_predictions - self.flat_labels))

    def convert_keys(self, results):
        strs = []
        for idx, (key, val) in enumerate(results):
            strs.append(key)
        return "\t".join(strs)
        
    def __str__(self):
        strs = []
        for idx, (key, val) in enumerate(self.results):
            _str = "%0.04f" % (val)
            strs.append(_str)
        return "\t".join(strs)

    def convert_to_str(self, results):
        strs = []
        for idx, (key, val) in enumerate(results):
            _str = "%0.04f" % (val)
            strs.append(_str)
        return "\t".join(strs)

    def __getitem__(self, item):
        return np.array([task_results[item] for task_results in self.results])


class ConvLayer(object):
    def __init__(self, motifs):
        self.motifs = motifs
        self.motif_shape = self.motifs.shape
        self.W = theano.shared(value = self.motifs, name = 'W', borrow = True)
        self.X = T.ftensor4('X')
        self.conv_out = conv2d(input = self.X, filters=self.W, border_mode = "valid", filter_flip = False)
        self.conv_func = theano.function([self.X], self.conv_out)

class ConvPredictor(object):
    def __init__(self, X, Y, motif_shape, ConvLayerObj, features_df, add_features = True):
        self.X = np.asarray(X, dtype = theano.config.floatX)
        #self.motifs = np.asarray(motfs, dtype = theano.config.floatX)
        self.Y = Y
        self.motif_shape = motif_shape
        self.ConvLayerObj = ConvLayerObj
        self.input_shape = self.X.shape
        self.features_df = features_df
        self.X_conv = self.Conv(self.X, self.X.shape, self.ConvLayerObj)
        if add_features:
            self.X_conv = np.hstack(self.X_conv, self.features_df.values)
    def Conv(self, X, input_shape, ConvLayerObj, minibatch_size = 2000):
        minibatches, rem = divmod(input_shape[0], minibatch_size)
        output = np.zeros((input_shape[0], self.motif_shape[0]))
        for idx in range(minibatches):
            cur_conv_output = ConvLayerObj.conv_func(X[idx*minibatch_size:(idx+1)*minibatch_size])
        #output is index, filter, _, _
            for cur_idx in range(minibatch_size):
                #self.transformed_output =self.cur_conv_output[cur_idx, :, 0, :]
                cur_max = np.amax(cur_conv_output[cur_idx, :, 0, :], axis = 1)
                output[idx*minibatch_size + cur_idx,:]= cur_max
        if rem > 0:
            cur_conv_output = ConvLayerObj.conv_func(X[minibatches*minibatch_size:])
            for cur_idx in range(cur_conv_output.shape[0]):
                cur_max = np.amax(cur_conv_output[cur_idx, :, 0, :], axis = 1)
                output[minibatches*minibatch_size + cur_idx, :] = cur_max
        return output

    def LogisticRegPredict(self, chrom_indices, other_indices, penalty = 'l2', solver = 'sag', tol = 0.001, C = 0.1, max_iter = 100, should_scale = False):
        self.LogisticRegObj = LogisticRegression(class_weight = 'balanced', penalty = penalty, solver = solver, tol = tol, C = C, max_iter = max_iter, random_state = 42)
        if should_scale:
            self.LogisticRegObj.fit(scale(self.X_conv[other_indices]), self.Y[other_indices])
        else:
            self.LogisticRegObj.fit(self.X_conv[other_indices], self.Y[other_indices])
        return self.LogisticRegObj.predict_proba(self.X_conv[chrom_indices])[:,1], self.Y[chrom_indices].astype(bool)

    def LogisticRegCVPredict(self, chrom_indices, other_indices, penalty = 'l2', solver = 'lbfgs', tol = 0.001, n_jobs = 3, max_iter = 100, should_scale = False):
        self.LogisticRegCVObj = LogisticRegressionCV(class_weight = 'balanced', penalty = penalty, solver = solver, n_jobs = n_jobs,  tol = tol, max_iter = max_iter, random_state = 42)
        if should_scale:
            self.LogisticRegCVObj.fit(scale(self.X_conv[other_indices]), self.Y[other_indices])
        else:
            self.LogisticRegCVObj.fit(self.X_conv[other_indices], self.Y[other_indices])
        return self.LogisticRegCVObj.predict_proba(self.X_conv[chrom_indices])[:,1], self.Y[chrom_indices].astype(bool)

    def ElasticNet(self, chrom_indices, other_indices, loss = 'log', penalty = "elasticnet", should_scale = True):
        self.ElasticNetObj = SGDClassifier(loss = loss, penalty = penalty, class_weight = 'balanced', learning_rate = 'optimal', random_state = 42)
        if should_scale:
            scaler = StandardScaler()
            self.scaler.fit(X_conv[other_indices])
            X_train= self.scaler.transform(self.X_conv[other_indices])
            X_test = self.scaler.transofrm(self.X_conv[chrom_indices])
        else:
            X_train = self.X_conv[other_indices]
            X_test = self.X_conv[chrom_indices]
        self.ElasticNetObj.fit(X_train, self.Y[other_indice])
        return self.ElasticNetObj.predict_proba(X_test)[:,1], self.Y[chrom_indices.astype(bool)]

    def get_class_weights(self, tar_vec):
        counts = np.zeros(2)
        for idx, val in enumerate(tar_vec):
            if val == 0:
                counts[0] += 1
            elif val == 1: 
                counts[1] += 1
        _sum = np.sum(counts)
        return _sum / float(counts[0]), _sum / float(counts[1])

    def FC(self, chrom_indices, other_indices, patience = 20, L1 = 1, L2 = 1, dropout_input = 0.2, dropout_hidden = 0.2, n_hidden = 5):
        self.model = Sequential()
        self.model.add(Dropout(dropout_input, input_shape =(self.X_conv.shape[1],)))
        self.model.add(Dense(n_hidden, init = 'glorot_uniform', W_regularizer= l1l2(L1, L2), b_regularizer = l1l2(L1,L2)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(dropout_hidden))
        self.model.add(Dense(output_dim=1, init = 'glorot_uniform', W_regularizer = l1l2(L1, L2), b_regularizer = l1l2(L1, L2)))
        self.model.add(Activation('sigmoid'))
        num_epochs = 130
        learning_rate = 0.0001
        self.adam = keras.optimizers.Adam(lr = learning_rate)
        self.model.compile(optimizer = self.adam, loss = 'binary_crossentropy', metrics=['binary_crossentropy'])
        self.earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience = patience, verbose = True,  mode = 'auto')
        neg_weight, pos_weight = self.get_class_weights(self.Y[other_indices])
        self.model.fit(x = self.X_conv[other_indices], y = self.Y[other_indices], batch_size = 128, nb_epoch = num_epochs, validation_split = 0.1, callbacks = [self.earlyStopping], class_weight= {0: neg_weight, 1: pos_weight}, verbose = False)
        #self.model.fit(x = self.X_conv[other_indices], y = self.Y[other_indices], batch_size = 2500, nb_epoch = 150, validation_split = 0.10, callbacks = [self.earlyStopping], shuffle = True, class_weight= 'auto', verbose = False)
        #self.model.fit(x = self.X_conv[other_indices], y = self.Y[other_indices], batch_size = 2500, nb_epoch = 125, shuffle = True, class_weight= {0: neg_weight, 1: pos_weight}, verbose = False)
        pred = self.model.predict(self.X_conv[chrom_indices], batch_size = 1000)
        return pred.flatten(), self.Y[chrom_indices].astype(bool)

class CvEngine(object):
    def __init__(self,name, motif_tensor, motif_names, merged_tensor, merged_tar, output_dir, C = 0.1, debug = False, penalty = 'l2', solver = 'sag'):
        self.name = name
        self.motif_tensor = motif_tensor
        self.merged_tensor = merged_tensor
        self.merged_tar = merged_tar
        self.resultObj = Results(name, output_dir)
        self.debug = debug
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.ConvLayerObj = ConvLayer(motif_tensor)
        self.motif_names = motif_names
        self.convObj = ConvPredictor(merged_tensor, merged_tar, self.motif_tensor.shape, self.ConvLayerObj)

    def start_CV(self, CV_dict):
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
            Y_pred, Y_true = self.convObj.LogisticRegPredict(pos_indices, neg_indices, C = self.C, penalty = self.penalty, solver = self.solver)
            if np.amax(Y_true) == 0:
                print "Skipping %s because no positive example"
            reg_weights = self.convObj.LogisticRegObj.coef_.T.flatten()
            reg_bias = self.convObj.LogisticRegObj.intercept_[0]
            reg_weights = pd.Series(data = reg_weights, index = self.motif_names, name = self.name + '_' + chrom)
            self.resultObj.add_cv_result(Y_pred, Y_true, chrom, reg_weights, reg_bias, pos_indices)
            idx += 1
            print("Completed lin reg on chromsome %s in %0.04f seconds"%(chrom, time.time() - start))
            if self.debug:
                if idx > 2:
                    break

    def start_CV_NN(self, CV_dict, dropout_input = 0.2, dropout_hidden =0.05, L1 = 0, L2 = 10, patience = 20, n_hidden = 15):
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
            Y_pred, Y_true = self.convObj.FC(pos_indices, neg_indices, patience = patience, dropout_input = dropout_input, dropout_hidden = dropout_hidden, L1 = L1, L2 = L2, n_hidden = n_hidden)

            if np.amax(Y_true) == 0:
                print "Skipping %s because no positive example"
            reg_weights = np.random.rand(self.motif_tensor.shape[0])
            reg_bias = [0]
            reg_weights = pd.Series(data = reg_weights, index = self.motif_names, name = self.name + '_' + chrom)
            self.resultObj.add_cv_result(Y_pred, Y_true, chrom, reg_weights, reg_bias, pos_indices)
            idx += 1
            print("Completed NN on  chromsome %s in %0.04f seconds"%(chrom, time.time() - start))
            if self.debug:
                if idx > 2:
                    break
                
    def summarize(self, dump_indices = False, dump_preds = False, dump_labels = False, dump_weights_bias = False, prefix = ''):
        self.resultObj.summarize(prefix = prefix)
        if dump_indices:
            self.resultObj.dump_indices(prefix = prefix)
        if dump_preds:
            self.resultObj.dump_preds(prefix=prefix)
        if dump_labels:
            self.resultObj.dump_true(prefix=prefix)
        if dump_weights_bias:
            self.resultObj.dump_weights_bias(prefix=prefix)

class Results(object):
    def __init__(self, data_name, output_dir):
        self.data_name = data_name #ie
        self.per_chrom_results_dict = {}
        self.per_chrom_preds = {}
        self.per_chrom_true = {}
        self.per_chrom_indices = {}
        self.all_probs = np.empty([1,])
        self.all_true = np.empty([1,], dtype = bool)
        self.output_dir = output_dir
        self.motif_weights = []
        self.motif_biases = []
        self.column_names = ''

    def add_cv_result(self, pred, true, chrom, motif_weight, motif_bias, indices):
        self.per_chrom_preds[chrom] = pred
        self.per_chrom_true[chrom] = true
        self.all_probs = np.append(self.all_probs.flatten(), pred.flatten())
        self.all_true = np.append(self.all_true.flatten(), true.flatten())
        cur_resultObj = ClassificationResult(true.flatten(), pred.flatten(), name = self.data_name + '_' + chrom)
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
        self.cum_resultObj = ClassificationResult(self.all_true.flatten(), self.all_probs.flatten(), name =self.data_name + '_cum')
        self.cum_result = self.cum_resultObj.binary(self.all_true, self.all_probs)
        self.str_cum_result = self.cum_resultObj.convert_to_str(self.cum_result)

    def summarize(self, output_file_name = '', prefix = ''):
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

    def dump_indices(self, output_dir = '', prefix = ''):
        cur_output_dir = ''
        if len(output_dir)>1:
            cur_output_dir = output_dir + "indices/"
        else:
            cur_output_dir = self.output_dir + "indices/"

        for chrom in sorted(self.per_chrom_indices.iterkeys()):
            indices = self.per_chrom_indices[chrom]
            cur_output_file = cur_output_dir + chrom + '_' + prefix + '_pos_indices.txt'
            fo = open(cur_output_file, 'w')
            for idx, val in enumerate(indices):
                fo.write("%i\n"%(val))
            fo.close()

    def process_weights_bias(self):
        self.weights_df = pd.DataFrame(self.motif_weights)
        #self.biases_df = pd.DataFrame(self.biases_df)

    def dump_weights_bias(self, output_file_dir = '', prefix = ''):
        self.process_weights_bias()
        if len(output_file_dir) > 1:
            cur_output_dir = output_file_name
        else:
            cur_output_dir = self.output_dir + 'weights/'

        weights_output_file = cur_output_dir + prefix + '_reg_weights.txt'
        bias_output_file = cur_output_dir + prefix + '_bias.txt'
        self.weights_df.to_csv(weights_output_file, sep = "\t")
        fo = open(bias_output_file, 'w')
        for idx, val in enumerate(self.motif_biases):
            fo.write("%0.04f\n"%(val))
        fo.close()

    def dump_preds(self, output_dir = '', prefix = ''):
        cur_output_dir = ''
        if len(output_dir)>1:
            cur_output_dir = output_dir
        else:
            cur_output_dir = self.output_dir + 'preds/'

        for chrom in sorted(self.per_chrom_preds.iterkeys()):
            predicted_prob = self.per_chrom_preds[chrom]
            cur_output_file = cur_output_dir + chrom + '_' + prefix + '_predicted_prob.txt'
            fo = open(cur_output_file, 'w')
            for idx, val in enumerate(predicted_prob):
                fo.write("%0.04f\n"%(val))
            fo.close()

    def dump_true(self,output_dir = '', prefix = ''):
        cur_output_dir = ''
        if len(output_dir)>1:
            cur_output_dir = output_dir
        else:
            cur_output_dir = self.output_dir + 'labels/'

        for chrom in sorted(self.all_true.iterkeys()):
            true = self.all_true[chrom]
            cur_output_file = cur_output_dir + chrom + '_' + prefix +  '_labels.txt'
            fo = open(cur_output_file, 'w')
            for idx, val in enumerate(true):
                fo.write("%0.04f\n"%(val))
            fo.close()

