import os 
import numpy as np
from scipy.stats import pearsonr
from sklearn.cluster import DBSCAN

def compute_match(pwm1, pwm2, len1, len2):
	pwm1 = np.asarray(pwm1)
	pwm2 = np.asarray(pwm2)

	if len1 == len2:
		r, _ = pearsonr(pwm1.flatten(), pwm2.flatten())
		return r
	elif len2 > len1:
		return compute_sub_match(pwm2,pwm1,len2,len1)
	elif len1 > len2:
		return compute_sub_match(pwm1,pwm2,len1,len2)
 
def compute_sub_match(pwm1, pwm2, len1, len2):
	#len1 > len2 
	sub_indices = len1-len2
	pearson_rs = []
	for idx in range(sub_indices):
		offset_start = idx
		offset_end = idx + len(len2)
		truncated_pwm = pwm1[offset_start:offset_end,:]
		r, _ = pearsonr(truncated_pwm, pwm2)
		pearson_rs.append(r)

	return np.max(np.asarray(pearson_rs))

def compute_similarity_matrix(pwms,lengths):
	length = pwms.shape[0]
	distance_matrix = np.zeros((length, length))
	for idx1, val1 in enumerate(pwms):
		for idx2, val2 in enumerate(pwms):
			if idx1 == idx2:
				distance_metric[idx1,idx2]=1
			else:
				len1=lengths[idx1]
				len2=lengths[idx2]
				distance_metric[idx1,idx2]=compute_match(val1,val2,len1,len2)
	return distance_matrix

def compute_num_clusters(distance_matrix):
	cluster_obj = DBSCAN(metric='precomputed')
	cluster_obj.fit(distance_matrix)
	return cluster_obj
