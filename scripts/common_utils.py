import os
import sys
import gzip
import numpy as np
import copy
import matplotlib.pyplot as plt
import pandas as pd
import time
import pickle
from collections import OrderedDict
from sklearn.metrics import auc, log_loss, precision_recall_curve, roc_auc_score, roc_curve
from prg.prg import create_prg_curve, calc_auprg
from sklearn.grid_search import ParameterGrid
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from Bio.Alphabet import IUPAC 
from sklearn.preprocessing import scale 
np.random.seed(42)
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression

class CrossValProcessor(object):
    def __init__(self, data_sets='all',
                 output_dirs=["./HEPG2_act/", "./HEPG2_rep/", "./K562_act/",
                              "./K562_rep/", "./LCL_act/"]):
        self.dna_files = ["/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/HEPG2_act_mpra_dna.txt",
                          "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/HEPG2_rep_mpra_dna.txt",
                          "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/K562_act_mpra_dna.txt",
                          "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/K562_rep_mpra_dna.txt",
                          "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/LCL_act_mpra_dna.txt"]
        self.target_files = ["/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/HEPG2_act_mpra_tar.txt",
                             "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/HEPG2_rep_mpra_tar.txt",
                             "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/K562_act_mpra_tar.txt",
                             "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/K562_rep_mpra_tar.txt",
                             "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/LCL_act_mpra_tar.txt"]
        self.anno_files = ["/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/HEPG2_act_mpra_det.txt",
                           "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/HEPG2_rep_mpra_det.txt",
                           "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/K562_act_mpra_det.txt",
                           "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/K562_rep_mpra_det.txt",
                           "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/LCL_act_mpra_det.txt"]
        self.feature_files = ["/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/HEPG2_act_feat.txt",
                              "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/HEPG2_rep_feat.txt",
                              "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/K562_act_feat.txt",
                              "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/K562_rep_feat.txt",
                              "/home/alvin/Dropbox/Lab/CNN/data/processed_cnn/LCL_act_feat.txt"]
        self.output_dirs = output_dirs
        for directory in self.output_dirs:
            if not os.path.exists(directory):
                os.makedirs(directory)
        #self.output_dirs = ["./HEPG2_act/", "./HEPG2_rep/", "./K562_act/", "./K562_rep/", "./LCL_act/"]
        self.names = ["HEPG2_act", "HEPG2_rep", "K562_act", "K562_rep", "LCL_act"]
        self.status = ["act", "rep", "act", "rep", "act"]
        if data_sets == 'all':
            self.HepG2_act_index = 0
            self.K562_act_index = 2
            self.LCL_act_index = 4
            self.HepG2_dict, self.HepG2_binObj = self.process_single_dataset(self.HepG2_act_index, LCL = False)
            self.K562_dict, self.K562_binObj = self.process_single_dataset(self.K562_act_index, LCL = False)
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
        cur_feature_file = [self.feature_files[cur_index]]
        cur_BinObj = BinaryClassProcessor(cur_DNA, cur_target, cur_anno_files, cur_status, cur_feature_file, LCL = LCL)
        cur_dict = cur_BinObj.generate_CV_dict(cur_BinObj.anno_dfs[0])
        return cur_dict, cur_BinObj

class BinaryClassProcessor(object):
    def __init__(self, dna_files, target_files, anno_files, status,
                 feature_files, LCL, RC = True, type = "bin"):
        self.dna_files = dna_files
        self.target_files = target_files
        self.anno_files = anno_files
        self.status = status
        self.LCL = LCL
        self.RC = RC
        self.type = "bin"
        self.feature_files = feature_files
        self.process_all_files()
        self.anno_df = self.anno_dfs[0]
        self.feature_df = self.process_features(self.anno_df, self.feature_dfs[0])
        self.features_mat = np.asarray(self.feature_df.values,dtype=np.float32)
    def change_tensor(self, tensor):
        mat = np.zeros((tensor.shape[0], tensor.shape[3]*tensor.shape[2]))
        for idx in range(tensor.shape[0]):
            current_dna = tensor[idx,0,:,:]
            flattened_dna = current_dna.flatten()
            mat[idx,:]=flattened_dna
        return mat

    def transform_array_theano(self, array):
        return np.asarray(array, dtype =np.float32)

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

    def generate_RC_tensor(self, input_tensor, mode='th'): #input is regular tensors
        output_tensor = np.zeros_like(input_tensor)
        for input_idx in range(input_tensor.shape[0]):
            if mode =='th':
                cur_mat = input_tensor[input_idx,0,:,:]
            elif mode == 'tf':
                cur_mat = input_tensor[input_idx,:,:,0]
            #the rows are A, C, G, T 
            copy_mat = np.zeros_like(cur_mat)
            for nuc_pos in list(reversed(range(cur_mat.shape[1]))): #Reverse
                copy_pos = cur_mat.shape[1] - nuc_pos - 1
                max_idx = np.argmax(cur_mat[:,nuc_pos])
                if max_idx == 0:
                    copy_mat[3,copy_pos] = 1 
                elif max_idx == 1:
                    copy_mat[2, copy_pos] = 1
                elif max_idx == 2:
                    copy_mat[1, copy_pos] = 1
                elif max_idx == 3:
                    copy_mat[0, copy_pos] = 1
            if mode == 'th':
                output_tensor[input_idx,0,:,:] = copy_mat
            elif mode == 'tf':
                output_tensor[input_idx,:,:,0] = copy_mat
        return output_tensor

    def process_features(self, anno_df, feature_df):
        ID_set = set(anno_df["ID"])
        to_drop_indices = []
        for index, row in feature_df.iterrows():
            if row.name not in ID_set:
                to_drop_indices.append(index)
        feature_df.drop(to_drop_indices, axis = 0, inplace = True)
        assert feature_df.shape[0] == anno_df.shape[0]
        assert np.all(feature_df.index.values == anno_df["ID"].values)
        return feature_df

    def generate_subtensors(self, input_tensor, num_copies, debug = False):
        input_shape = input_tensor.shape
        output_tensor = np.zeros((input_shape[0]*num_copies, input_shape[1], input_shape[2], input_shape[3]-num_copies+1), dtype=np.float32)
        if debug:
            print("Shape of output tensor is ")
            print(output_tensor.shape)
        for input_idx in range(input_tensor.shape[0]):
            cur_mat = input_tensor[input_idx,0,:,:]
            if debug:
                print("Shape of current matrix is ")
                print(cur_mat.shape)
            for offset in range(num_copies):
                end_index = cur_mat.shape[1] - num_copies + 1
                if debug:
                    print("On offset %i, my start and end indices are: %i, %i"%(offset, offset, offset+end_index))
                output_mat = cur_mat[:,offset:(offset+end_index)]
                if debug:
                    print("I am inserting an output matrix with size")
                    print(output_mat.shape)
                    print("and my index into output array is %i"%(input_idx*num_copies + offset))
                output_tensor[input_idx*num_copies + offset, 0, :, :] = output_mat
        return output_tensor

    def generate_random_subtensors(self, input_tensor, num_copies, debug = False):
        input_shape = input_tensor.shape
        output_tensor = np.zeros((input_shape[0], input_shape[1], input_shape[2], input_shape[3]-num_copies+1), dtype=np.float32)
        for input_idx in range(input_tensor.shape[0]):
            offset = np.random.randint(low=0, high=num_copies)
            cur_mat = input_tensor[input_idx,0,:,:]
            end_index = cur_mat.shape[1] - num_copies + 1
            output_mat = cur_mat[:,offset:(offset+end_index)]
            output_tensor[input_idx, 0, :, :] = output_mat
        return output_tensor

    def generate_centered_subtensors(self, input_tensor, num_copies):
        right_offset = num_copies-1
        input_shape = input_tensor.shape
        output_tensor = np.zeros((input_shape[0],input_shape[1], input_shape[2], input_shape[3]-num_copies+1), dtype=np.float32)
        for input_idx in range(input_tensor.shape[0]):
            offset = np.floor(right_offset / float(2))
            cur_mat = input_tensor[input_idx,0,:,:]
            end_index = cur_mat.shape[1] - num_copies + 1
            output_mat = cur_mat[:,offset:(offset+end_index)]
            output_tensor[input_idx, 0, :, :] = output_mat
        return output_tensor

    def transform_indices(self, indices, num_copies):
        output_indices = np.zeros((len(indices)*num_copies,),dtype=int)
        for idx, val in enumerate(indices):
            for offset in range(num_copies):
                output_indices[idx*num_copies+offset] = val
        return output_indices

    def transform_labels(self, labels, num_copies):
        return np.repeat(labels, num_copies)

    def convert_mat_to_DNA(self, mat):  #input is 4x150 1/0 binarized motif
        DNA_result = ""
        for idx in range(mat.shape[1]):
            max_idx = np.argmax(mat[:,idx])
            if max_idx == 0:
                DNA_result += "A"
            elif max_idx == 1:
                DNA_result += "C"
            elif max_idx == 2:
                DNA_result += "G"
            elif max_idx == 3:
                DNA_result += "T"
        return DNA_result

    def convert_to_fasta(self, tensor, names, file_name):
        record_array = []
        for idx in range(tensor.shape[0]):
            dna = Seq(self.convert_mat_to_DNA(tensor[idx,0,:,:]), IUPAC.unambiguous_dna)
            record = SeqRecord(dna, id = names[idx])
            record_array.append(record)
        SeqIO.write(record_array, file_name, "fasta")

    def positive_negative_indices(self, merged_tar):
        pos_array = []
        neg_array = []
        for idx, val in enumerate(merged_tar):
            if val == 0:
                neg_array.append(idx)
            elif val == 1:
                pos_array.append(idx)
        assert len(pos_array) + len(neg_array) == len(merged_tar)
        return pos_array, neg_array 

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
        self.feature_dfs = []
        self.balance = [0, 0]
        self.chroms = []
        for idx in range(len(self.dna_files)):
            anno_df, dna_tensor, tar_vec, cur_balance, feature_df = self.process_one_file(idx)
            self.tar_vecs.append(tar_vec)
            self.anno_dfs.append(anno_df)
            self.feature_dfs.append(feature_df)
            self.dna_tensors.append(dna_tensor)
            self.chroms += self.color_chroms(anno_df)
            self.balance[0] += cur_balance[0]
            self.balance[1] += cur_balance[1]
        self.merged_tensor = np.concatenate(self.dna_tensors)
        if self.RC:
            self.merged_RC_tensor = self.generate_RC_tensor(self.merged_tensor)
        self.merged_tar = np.concatenate(self.tar_vecs)
        self.bin_merged_tar = self.merged_tar

    def process_one_file(self, idx):
        dna_file = self.dna_files[idx]
        target_file = self.target_files[idx]
        anno_file = self.anno_files[idx]
        anno_df, N = self.prep_anno(anno_file)
        status = self.status[idx]
        dna_tensor = self.prep_dna(dna_file, N)
        if os.path.exists(self.feature_files[idx]):
            feature_df = pd.read_csv(self.feature_files[idx], sep="\t")
        elif os.path.exists(self.feature_files[idx] + '.tar.gz'):
            feature_df = pd.read_csv(self.feature_files[idx] + 'tar.gz', sep="\t")
        elif os.path.exists(self.feature_files[idx] + '.gz'):
            feature_df = pd.read_csv(self.feature_files[idx] + '.gz', sep="\t")
        tar_vec, balance = self.prep_tar(target_file, N, status, self.type)
        return anno_df, dna_tensor, tar_vec, balance, feature_df

    def prep_anno(self, anno_file):
        anno_df = pd.read_csv(anno_file, sep="\t", header=0)
        N = len(anno_df.index)
        return anno_df, N

    def prep_dna(self, dna_file, N):
        idx = 0
        dna_tensor = np.zeros((N, 1, 4, 150))
        print dna_file
        if os.path.exists(dna_file):
            f = open(dna_file, 'r')
            print "original"
        elif os.path.exists(dna_file + '.tar.gz'):
            f = gzip.open(dna_file + '.tar.gz', 'r')
            print "tar.gz"
        elif os.path.exists(dna_file + '.gz'):
            f = gzip.open(dna_file + '.gz', 'r')
            print ".gz"
        for line in f:
            split_line = line.rstrip().split()
            _N, _W = divmod(idx, 4)
            dna_tensor[_N, 0, _W, :] = np.asarray(split_line, dtype=int)
            idx += 1
        return dna_tensor

    def prep_tar(self, tar_file, N, status, type="tanh"):
        idx = 0
        tar_vec = np.zeros(N)
        if os.path.exists(tar_file):
            f = open(tar_file, 'r')
        elif os.path.exists(tar_file + '.tar.gz'):
            f = gzip.open(tar_file + '.tar.gz', 'r')
        elif os.path.exists(tar_file + '.gz'):
            f = gzip.open(tar_file + '.gz', 'r')
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


def th_to_tf(tensor):
    return np.transpose(tensor, (0, 2, 3, 1))

def th_to_1d(tensor):
    return np.squeeze(tensor)

def expand_tensor(tensor):
    return np.expand_dims(tensor, axis=3)

class ClassificationResult(object):
    def __init__(self, labels, predictions, name=None):
        self.predictions = predictions
        self.labels = labels
        self.flat_predictions = predictions.flatten()
        self.flat_labels = labels.flatten()
        self.results = []
        self.self_binary()

    def self_binary(self):
        self.results = self.binary(self.labels, self.predictions)

    def other_binary(self, labels, predictions):
        self.results = self.binary(labels, predictions)

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

        def sense_at_FPR(labels, predictions):
            fpr, tpr, threshold = roc_curve(labels,predictions)
            def nearest(array, value):
                idx = (array-value).argmin()
                return idx
            idx = nearest(fpr, 0.05)
            return tpr[idx]
        
        def recall_at_precision_threshold(labels, predictions,
                                          precision_threshold):
            precision, recall = precision_recall_curve(labels, predictions)[:2]
            return 100 * recall[np.searchsorted(precision -
                                                precision_threshold, 0)]

        results = [('Loss', loss(labels, predictions)), (
            'Balanced_accuracy', balanced_accuracy(
                labels, predictions)), ('auROC', auROC(labels, predictions)),
                   ('auPRC', auPRC(labels, predictions)),
                   ('auPRG', auPRG(labels, predictions)),
                   ('Senstivity_at_5%_FPR', sense_at_FPR(labels, predictions)),
                   ('Recall_at_5%_FDR', recall_at_precision_threshold(
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

class JsonWriter(object):
    def __init__(self, main_file):
        pass
