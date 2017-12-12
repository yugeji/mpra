#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from scipy.stats import entropy
import numpy as np
import math
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
from Bio import motifs
from scipy.stats import pearsonr
from multiprocessing import Pool, Process
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import AffinityPropagation


def msd(self, pwm1, pwm2):
    return np.sqrt(np.sum(np.square(pwm1 - pwm2)))

def pearson(self, pwm1, pwm2):
    r, _ = pearsonr(pwm1.flatten(), pwm2.flatten())
    return r

class MotifDistanceProcessor(object):
    def __init__(self, pwms, motif_names, lengths):
        self.pwms = pwms
        self.motif_names = motif_names
        self.lengths = lengths
        self.v_msd = np.vectorize(msd)

    def load_Pouya_dist_matrix(self, pouya_dist_loc='~/Dropbox/Lab/CNN/data/pouya_motifs/motifs-sim.txt'):
        df = pd.read_csv(pouya_dist_loc, sep="\t", index_col=0)
        return df

    def cluster_Pouya(self, mat, **kwargs):
        AP_obj = AffinityPropagation(affinity="precomputed",**kwargs)
        cluster_labels = AP_obj.fit_predict(mat)
        return cluster_labels

    def compute_distance_matrix(self, multiprocess=False, manual=False):
        if multiprocess:
            pass
        else:
            pass

    def match_names_to_indices(self, motif_names, df):
        motif_name_set = set(motif_names)
        indices = []
        for idx, val in enumerate(list(df.index.values)):
            if val in motif_name_set:
                indices.append(idx)
        return indices

    def process_Pouya(self, motif_names, **kwargs):
        self.Pouya_df = self.load_Pouya_dist_matrix()
        self.indices = self.match_names_to_indices(motif_names, self.Pouya_df)
        self.reduced_df = self.Pouya_df.iloc[self.indices, self.indices]
        self.cluster_indices = self.cluster_Pouya(self.reduced_df.values, **kwargs)
        self.max_cluster = np.max(self.cluster_indices)+1
        self.binary_mat = np.zeros((self.reduced_df.shape[0], self.max_cluster), dtype=np.float32)
        for idx, val in enumerate(self.cluster_indices):
            self.binary_mat[idx,val]=1
        return self.binary_mat

    def compute_match(self, pwm1, pwm2, len1, len2, mode="squared"):
        pwm1 = np.asarray(pwm1)
        pwm2 = np.asarray(pwm2)
        if len1 == len2:
            if mode == "squared":
                return self.v_msd(pwm1, pwm2)
            elif mode == "pearson":
                return pearson(pwm1, pwm2)
        elif len1 < len2:
            return self.compute_sub_match(pwm2, pwm1, len2, len1, mode=mode)
        elif len1 > len2:
            return self.compute_sub_match(pwm1, pwm2, len1, len2, mode=mode)

    def compute_sub_match(self, pwm1, pwm2, len1, len2, mode="squared"):
        assert len1 > len2
        assert pwm1.shape[0] > pwm2.shape[0]
        sub_indices = len1-len2
        distances = np.zeros((sub_indices,))
        for idx in range(sub_indices):
            start = idx
            end = idx + len2
            truncated_pwm = pwm1[start:end,:]
            assert truncated_pwm.shape==pwm2.shape
            if mode=="squared":
                d = self.v_msd(pwm1,pwm2)
            elif mode=="pearson":
                d = pearson(pwm1, pwm2)
            distances[idx] = d
        return np.max(distances)

    def distance(pwm1, pwm2):
        assert pwm1.shape == pwm2.shape, "Shapes are unequal"
        np.apply_over_axes(pwm)

class MotifProcessor(object):
    """Class to process ENCODE motifs"""
    def __init__(self, encodeMotifFileName="/home/alvin/Dropbox/Lab/CNN/data/pouya_motifs/all_encode_motifs.txt", ENCODE_only=True, cell_line = None, meme_file = None):
        """Initalize the processor with the ENCODE motif file name
        All motifs are encoded in [A,C,G,T] format in the encode motif file
        Args:
            enocodeMotifFileName: File location for the encode motifs"""
        self.encodeMotifFileName = encodeMotifFileName
        if ENCODE_only:
            self.all_motifs, self.motif_names = self.process()
        else:
            encode_motifs, encode_names = self.process()
            meme_motifs, meme_names, self.evalues = self.process_STEME_output(meme_file, cell_line)
            self.all_motifs = encode_motifs + meme_motifs
            self.motif_names = encode_names + meme_names

    def process_MEME_output(self, meme_output_file, cell_line, mode = "STEME"):
        all_motifs = []
        motif_names = []
        with open(meme_output_file) as handle:
            for m in motifs.parse(handle, "MEME"):
                all_motifs.append(m.pwm)
                motif_names.append("%s_%s_%s"%(mode, cell_line, m.consensus))
        return all_motifs, motif_names

    def process_STEME_output(self, steme_output_file, cell_line):
        motif_names = []
        motif_lengths = []
        motif_evalues = []
        all_motifs = []
        current_motif = []
        idx = 0
        p = re.compile("MOTIF (STEME-.*?)$")
        r = re.compile("^.*w=(.*?)nsites.*E=(.*)?$")
        f = open(steme_output_file, 'r')
        for line in f:
            if line.startswith('MOTIF'):
                re_match = p.search(line)
                if re_match:
                    #print(re_match.group(1))
                    motif_name = re_match.group(1).rstrip()
                    motif_names.append(motif_name)
                    if len(current_motif)>0:
                        all_motifs.append(current_motif)
                        current_motif = []
                    else:
                        continue
                else:
                    print("Unable to match, current line is %s"%(line))
            elif line.startswith('letter'):
                re_match = r.search(line)
                if re_match:
                    #print(re_match.group(1))
                    length = int(re_match.group(1).rstrip())
                    e_value = float(re_match.group(2).rstrip())
                    motif_lengths.append(length)
                    motif_evalues.append(e_value)
            elif line.startswith('0'):
                split_line = line.rstrip().split()
                current_motif.append(split_line)
            idx+=1
        return all_motifs, motif_names, motif_evalues


    def process(self, filename):
        """Processes the motif file"""
        f = open(filename, 'r')
        p = re.compile("^>(.*?)\s+(.*?)$")
        all_motifs = []
        motif_names = []
        current_motif = []
        idx = 0
        for line in f:
            if line.startswith('>'):
                re_match = p.search(line)
                if re_match:
                    motif_name = re_match.group(1)
                    motif_names.append(motif_name)
                    if len(current_motif)>0:
                        all_motifs.append(current_motif)
                        current_motif = []
                    else:
                        continue
                else:
                    print("Unable to match, current line is %s"%(line))
            else:
                split_line = line.rstrip().split()
                current_motif.append(split_line[1:])
            idx+=1
        all_motifs.append(current_motif)
        return all_motifs, motif_names


    def find_length_distributions(self):
        """Diagnostic function to determine the maximum and distribution of motifs in the
        ENCODE motif data set"""
        current_max = 0
        dist = []
        for idx, val in enumerate(self.all_motifs):
            if len(val)>current_max:
               current_max = len(val)
            dist.append(len(val))
        return current_max,dist 

    def generate_filters(self, max_length = 18, padding = 0.25, truncate=False):
        """Generates filter weights"""
        padding = float(padding)
        self.max_length = 18
        def determine_num(all_motifs,length):
            """Determine number of motifs below a certain max_length"""
            num=0
            for idx, val in enumerate(all_motifs):
                if length>=len(val):
                    num+=1
            return num

        def pack_less(motif_size, max_size):
            """Determine optimal starting point for a motif size given max motif size """
            diff_size = max_size - motif_size
            return int(math.floor(diff_size/float(2)))

        def pack_more(motif_size, max_size):
            pass
        num_motifs = determine_num(self.all_motifs,max_length)
        init_matrix = np.reshape(np.asarray([padding]*(num_motifs*max_length*4), dtype = 'float32') , (num_motifs, 4, max_length))
        cur_idx = 0
        arrays = []
        selected_motif_names=[]
        self.length = []
        self.shannon_index = []
        for idx, val in enumerate(self.all_motifs):
            motif_np = np.asarray(val).astype(float)
            motif_transpose = motif_np.T
            motif_length = motif_transpose.shape[1]
            if len(val)==max_length:
                init_matrix[cur_idx]=motif_transpose
                arrays.append(init_matrix[cur_idx])
                cur_idx+=1
                selected_motif_names.append(self.motif_names[idx])
                self.length.append(len(val))
            elif len(val)<max_length:
                start_idx = pack_less(motif_length, max_length)
                end_idx = start_idx+motif_length
                #print(start_idx)
                #print(end_idx)
                init_matrix[cur_idx, :, start_idx:end_idx]=motif_transpose
                arrays.append(init_matrix[cur_idx])
                selected_motif_names.append(self.motif_names[idx])
                cur_idx+=1
                self.length.append(len(val))
        self.matrix = init_matrix
        return init_matrix, selected_motif_names

    def reshape_tensor(self, init_matrix, selected_motif_names):
        self.num_motifs = init_matrix.shape[0]
        self.motif_width = self.max_length
        self.reshaped_motif_tensor = np.zeros((self.num_motifs, 1, 4, self.motif_width), dtype = 'float32')
        for idx, val in enumerate(init_matrix):
            self.reshaped_motif_tensor[idx, 0, :, :] = val 
        return self.reshaped_motif_tensor, selected_motif_names

    def generate_custom_CNN_filters(self, max_length = 18, padding = 0.25, truncate = False):
        init_matrix, motif_names = self.generate_filters(max_length, padding, truncate)
        reshaped_motif_tensor, motif_names = self.reshape_tensor(init_matrix, motif_names)
        reshaped_motif_tensor = np.asarray(reshaped_motif_tensor, dtype = 'float32')
        return reshaped_motif_tensor, motif_names

if __name__ == '__main__':
    processorObj = MotifProcessor()
    processorObj.process()
