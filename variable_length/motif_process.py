import numpy as np

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

def processing_jasper_motifs(self):
    #processing jasper motifs
    longest_motif = 64

    motifs = []
    f = open("jasper_motifs.txt")
    lines = f.readlines()

    for i in range(len(lines)):
        if lines[i][0] == ">": #first line of a motif
            m = []
            for j in range(1, 5): #ACGT
                l = lines[i + j]
                s = l[5:len(l)-2].split()
                s = map(int, s)
                s = s + [0]*(longest_motif-len(s))
                m.append(s)
            #normalize
            for j in range(len(m[0])):
                sum = m[0][j] + m[1][j] + m[2][j] + m[3][j]
                if sum == 0:
                    break
                for b in range(4):
                    val = m[b][j]/float(sum)
                    m[b][j] = val

            motifs.append(m)

    f.close()

    fl = open("jasper_processed.txt", "wb")
    for motif in motifs:
        for base in motif:
            l = [str(float(x)) for x in base]
            fl.write(" ".join(l))
            fl.write(" ")
        fl.write("\n")

#processing pouya motifs, from Alvin
def process(filename):
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


print process('all_encode_motifs.txt')
