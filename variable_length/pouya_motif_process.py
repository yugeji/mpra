from __future__ import absolute_import, division, print_function
import numpy as np
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


#processing pouya motifs, from Alvin
def process(filename):
    """Processes the motif file"""
    f = open(filename, 'r')
    p = re.compile("^>(.*?)\s+(.*?)$")
    all_motifs = {}
    motif_names = []
    current_motif = []
    idx = 0
    for line in f:
        if line.startswith('>'):
            re_match = p.search(line)
            if re_match:
                motif_name = re_match.group(1)
#                motif_names.append(motif_name)
                if len(current_motif)>0:
                    all_motifs[motif_name] = current_motif
                    current_motif = []
                else:
                    continue
            else:
                print("Unable to match, current line is %s"%(line))
        else:
            split_line = line.rstrip().split()
            current_motif.append(split_line[1:])
        idx+=1
    all_motifs[motif_name] = current_motif
    return all_motifs

#filter motifs by p < .05
def filtered_pouya(motif_file, filter_file):
    f = open(filter_file, 'r')
    motif_dict = process(motif_file)
    res_motif_dict = {}

    for _ in xrange(11): #skip the first 11 lines, not motifs
        next(f)

    for line in f: #going through all motifs relevant to the data
        vals = line.split()
        try:
            if float(vals[-2]) < .05: #only take motifs with p < .05
                motif_name = vals[0].split(")")[1]
                
                #process motifs first: known, then disc, or 1
                best_motif = "motif_non1"
                for k in motif_dict.keys():
                    if motif_name in k: #this could a motif we're looking for
                        ranking = k.split("_")[1]
                        cur_ranking = best_motif.split("_")[1]
                        if cur_ranking.isdigit():
                            if ranking.isdigit() and int(ranking) < int(cur_ranking):
                                best_motif = k
                        elif 'known' in best_motif:
                            if ranking.isdigit():
                                best_motif = k
                            elif 'known' in ranking and int(ranking[5:]) < int(cur_ranking[5:]):
                                best_motif = k
                        elif 'disc' in best_motif:
                            if ranking.isdigit() or 'known' in ranking:
                                best_motif = k
                            elif 'disc' in ranking and int(ranking[4:]) < int(cur_ranking[4:]):
                                best_motif = k
                        else:
                            best_motif = k    
                try:
                    res_motif_dict[motif_name] = motif_dict[best_motif]
                except KeyError:
                    print (best_motif, "not found")
                
        except ValueError:
            break
            
    return res_motif_dict


#zero-padding because motifs are various sizes
def zero_padded_motifs(motif_dict): #returns the same thing, but zero-padded
    for k in motif_dict.keys():
        result = np.zeros((29, 4))
        v = np.asarray(motif_dict[k])
        result[:v.shape[0], :v.shape[1]] = v
        motif_dict[k] = result
    return motif_dict

print (len(zero_padded_motifs(filtered_pouya('all_encode_motifs.txt', 'lm2_082417.txt'))))
#print (len(filtered_pouya('all_encode_motifs.txt', 'lm2_082417.txt').keys()))
        
            
        
    



