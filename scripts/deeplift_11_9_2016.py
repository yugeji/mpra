from __future__ import absolute_import, division, print_function
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers.core import (Activation, Dense, Dropout, Flatten, Permute,
                               Reshape, TimeDistributedDense)
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
from prg.prg import create_prg_curve, calc_auprg
from sklearn.grid_search import ParameterGrid
from sklearn.model_selection import StratifiedShuffleSplit
from keras.utils.np_utils import to_categorical
from keras.layers.normalization import BatchNormalization

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


#end copy + paste
class Mutagen(object):
    def __init__(self, ref_file, alt_file, prob_file, sig_file):
        self.ref_file = ref_file
        self.alt_file = alt_file
        self.prob_file = prob_file
        self.sig_file = sig_file

    def prepare(self):
        self.prep_sig()
        self.process_dna()

    def process_dna(self):
        self.ref_tensor = self.prep_dna(self.ref_file, self.N)
        self.alt_tensor = self.prep_dna(self.alt_file, self.N)

    def prep_sig(self):
        self.sig_df = pd.read_csv(self.sig_file, sep="\t", header=None)
        self.Z = self.sig_df[0]
        self.N = len(self.sig_df[0])

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

    def mutate_loc(self, indices, mutate_to):
        #indices - all spots to be mutated
        self.mutated_tensor = self.inital_obj.dna_tensor[indices, :, :, :]
        self.labels = self.inital_obj.tar_vec[indices]
        self.pre_mutation_tensor = copy.deepcopy(self.mutated_tensor)
        for idx, (val, new) in enumerate(mutate_to):
            self.mutated_tensor[idx, :, :, val] = new

    def predict(self):
        pass


class Initalizer(object):
    def __init__(self, dna_file, target_file, anno_file, output_dir, name):
        self.dna_file = dna_file
        self.target_file = target_file
        self.anno_file = anno_file
        self.output_dir = output_dir
        f = open(self.dna_file, 'r')
        line = f.readline()
        split_line = line.rstrip().split()
        self.W = 4
        self.H = len(split_line)
        self.name = name
        f.close()

    def prepare(self):
        self.prep_anno()
        self.prep_dna()
        self.prep_tar()
        print("Preparation of input matrices complete")

    def prep_anno(self):
        self.anno_df = pd.read_csv(self.anno_file, sep="\t", header=0)
        self.N = len(self.anno_df.index)

    def prep_dna(self):
        idx = 0
        self.dna_tensor = np.zeros((self.N, 1, self.W, self.H))
        f = open(self.dna_file, 'r')
        for line in f:
            split_line = line.rstrip().split()
            _N, _W = divmod(idx, 4)
            self.dna_tensor[_N, 0, _W, :] = np.asarray(split_line, dtype=int)
            idx += 1

    def prep_tar(self):
        idx = 0
        self.tar_vec = np.zeros(self.N)
        f = open(self.target_file, 'r')
        for line in f:
            split_line = line.rstrip().split()
            self.tar_vec[idx] = int(split_line[0])
            idx += 1

    def find_chr_index(self, chrom):
        indices = []
        for idx, val in enumerate(list(self.anno_df.iloc[:, 0])):
            if val == chrom:
                indices.append(idx)
        return indices

    def mask(self, indices):
        mask = np.ones(self.N, dtype=bool)
        mask[indices, ] = False
        test_tensor = self.dna_tensor[indices, :, :]
        train_tensor = self.dna_tensor[mask, :, :]
        test_tar = self.tar_vec[indices]
        train_tar = self.tar_vec[mask]
        return train_tensor, test_tensor, train_tar, test_tar

    def split_chrom(self, chrom):
        indices = self.find_chr_index(chrom)
        train_tensor, test_tensor, train_tar, test_tar = self.mask(indices)
        return Data(train_tensor, test_tensor, train_tar, test_tar, chrom,
                    indices, self)


class Data(object):
    def __init__(self, train_tensor, test_tensor, train_tar, test_tar, chrom,
                 indices, initalObj):
        self.train_tensor = train_tensor
        self.test_tensor = test_tensor
        self.train_tar = train_tar
        self.test_tar = test_tar
        self.chrom = chrom
        self.indices = indices
        self.initalObj = initalObj
        self.N_train = self.train_tensor.shape[0]
        self.N_test = self.test_tensor.shape[0]
        self.seq_length = self.initalObj.H
        self.name = self.initalObj.name


def process_result_objs(obj_list, output_file_name):
    concat_labels = []
    concat_predictions = []
    for idx, val in enumerate(obj_list):
        concat_labels.append(val.labels)
        concat_predictions.append(val.predictions)
    concated_labels = np.concatenate(concat_labels)
    concated_predictions = np.concatenate(concat_predictions)
    o = open(output_file_name, 'w')
    for idx, (val1,
              val2) in enumerate(zip(concated_labels, concated_predictions)):
        o.write("%i\t%0.05f\n" % (val1, val2))
    o.close()
    overall_class_obj = ClassificationResult(
        concated_labels.astype(bool), concated_predictions)
    return overall_class_obj


class DenseNeuralNet(object):
    def __init__(self, verbose=0, seq_length=600, name=""):
        self.verbose = verbose
        self.seq_length = seq_length
        #self.input_shape = (1, 4, self.seq_length)
        self.input_shape = 600 
        self.name = ""
        
    def init_model(self, l1 = 0.01, l2 = 0.01, dropout=0.01):
        self.seq_length = self.seq_length
        self.input_length = self.seq_length*4 
        self.model = Sequential()
        self.model.add(Dense(400, input_dim=600, init='uniform',
                             W_regularizer=l1l2(l1,l2),
                             b_regularizer=l1l2(l1,l2)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(dropout))
        self.model.add(Dense(100, init='uniform'))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(dropout))
        self.model.add(Dense(10, init='uniform'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(dropout))
        self.model.add(Dense(output_dim=1))
        self.model.add(Activation('sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy')

    def train(self, x_train, y_train, x_test, y_test, pos_weight, neg_weight):
        self.model.fit(x_train, y_train, batch_size=1000, nb_epoch=30, verbose=1, class_weight={0:neg_weight, 1:pos_weight},validation_data=(x_test, y_test))
class BinaryClassProcessor(object):
    def __init__(self, dna_files, target_files, anno_files, status,
                 type="cat"):
        self.dna_files = dna_files
        self.target_files = target_files
        self.anno_files = anno_files
        self.status = status
        self.type = "bin"
        self.process_all_files()
        
    def return_test_train(self, test_size=0.2):
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=0)
        if self.type == "tanh":
            for train_index, test_index in sss.split(self.merged_tensor,
                                                     self.merged_tensor):
                X_train, X_test = self.merged_tensor[
                    train_index], self.merged_tensor[test_index]
                Y_train, Y_test = self.merged_tar[
                    train_index], self.merged_tar[test_index]
            return X_train, X_test, Y_train, Y_test
        elif self.type == "bin":
            self.converted_cat = self.merged_tar
            for train_index, test_index in sss.split(self.merged_tensor,
                                                     self.merged_tensor):
                X_train, X_test = self.merged_tensor[
                    train_index], self.merged_tensor[test_index]
                Y_train, Y_test = self.converted_cat[
                    train_index], self.converted_cat[test_index]
            return X_train, X_test, Y_train, Y_test

    def change_tensor(self, tensor):
        mat = np.zeros((tensor.shape[0], tensor.shape[3]*tensor.shape[2]))
        for idx in range(tensor.shape[0]):
            current_dna = tensor[idx,0,:,:]
            flattened_dna = current_dna.flatten()
            mat[idx,:]=flattened_dna
        return mat
        
    def return_train_test_chrom(self, chrom):
        chrom_indices, other_indices = self.find_chr_by_color(self.chroms,
                                                              chrom)
        X_train = self.merged_tensor[other_indices]
        Y_train = self.merged_tar[other_indices]
        X_test = self.merged_tensor[chrom_indices]
        Y_test = self.merged_tar[chrom_indices]
        return X_train, X_test, Y_train, Y_test
    
    #The optimal running order is
    #process_all_files
    #color_chroms
    #return_train_test_chrom on desired chromosome
    #collapse tensors to Nx600 vectors and feed into deep neural net.

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

    def color_chroms(self, anno_df):
        chrom_array = []
        for idx, val in enumerate(list(anno_df.iloc[:, 0])):
            chrom_array.append(val)
        return chrom_array


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

    def bin_weights(self):
        bal_sum = sum(self.balance)
        neg_weight = bal_sum / float(self.balance[0])
        pos_weight = bal_sum / float(self.balance[1])
        return neg_weight, pos_weight

if __name__ == "__main__":
    # dna_file = sys.argv[1]
    # target_file = sys.argv[2]
    # anno_file = sys.argv[3]
    # output_dir = sys.argv[4]
    dna_files = [
        "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/HEPG2_act_mpra_dna.txt",
        "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/HEPG2_rep_mpra_dna.txt",
        "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/K562_act_mpra_dna.txt",
        "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/K562_rep_mpra_dna.txt",
        "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/LCL/LCL_act_mpra_dna.txt"
    ]
    target_files = [
        "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/HEPG2_act_mpra_tar.txt",
        "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/HEPG2_rep_mpra_tar.txt",
        "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/K562_act_mpra_tar.txt",
        "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/K562_rep_mpra_tar.txt",
        "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/LCL/LCL_act_mpra_tar.txt"
    ]
    anno_files = [
        "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/HEPG2_act_mpra_det.txt",
        "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/HEPG2_rep_mpra_det.txt",
        "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/K562_act_mpra_det.txt",
        "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/K562_rep_mpra_det.txt",
        "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/LCL/LCL_act_mpra_det.txt"
    ]
    output_dirs = [
        "./HEPG2_act/", "./HEPG2_rep/", "./K562_act/", "./K562_rep/",
        "./LCL_act/"
    ]
    names = ["HEPG2_act", "HEPG2_rep", "K562_act", "K562_rep", "LCL_act"]
    status = ["act", "rep", "act", "rep", "act"]
    multiClassObj = MultiClassProcessor(
        dna_files, target_files, anno_files, status, type="cat")
    X_train, X_test, Y_train, Y_test = multiClassObj.return_test_train()
    modelObj = Model(verbose=0, seq_length=150)
    modelObj.train(X_train, Y_train)
    self_result_obj = modelObj.test(X_test, Y_test)
    modelObj.train()
    ref_file = "LCL_act_mpra_emVar_ref.txt"
    alt_file = "LCL_act_mpra_emVar_alt.txt"
    z_file = "LCL_act_mpra_emVar_tar.txt"
    prob_file = "LCL_act_mpra_emVar_eft.txt"
    det_file = "./processed_cnn/LCL_act_mpra_emVar_det.txt"
    det_df = pd.read_csv(det_file, sep="\t", header=0)
    log_det = det_df["LogSkew.Comb"]

