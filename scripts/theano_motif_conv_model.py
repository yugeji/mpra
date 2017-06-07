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
np.random.seed(42)
theano.config.exception_verbosity = 'low'
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

#define the RMS prop optimizer
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
        return X_train[self.shuffle_array], X_test, Y_train[self.shuffle_array], Y_test
    
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

    def bin_weights(self, y, pos_multiplier = 1.3):
        return_array = np.zeros((len(y),))
        bal_sum = sum(self.balance)
        neg_weight = bal_sum / float(self.balance[0])
        pos_weight = bal_sum / float(self.balance[1])
        for idx, val in enumerate(y):
            if val == 0:
                return_array[idx] = neg_weight
            elif val == 1:
                return_array[idx] = pos_multiplier *pos_weight
        return_array = np.asarray(return_array, dtype = theano.config.floatX)
        return neg_weight, pos_weight, return_array

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

    def calculate_actual_convs(self, w = 1, b = 0):
        output_mat = np.zeros((self.dna.shape[0], self.motifs.shape[0], 1, self.dna.shape[-1]-self.motifs.shape[-1]+1))
        for n in range(self.dna.shape[0]):
            for k in range(self.motifs.shape[0]):
                for i in range(self.dna.shape[-1] - self.motifs.shape[-1] + 1):
                    output_mat[n,k,0,i] = Rectifier(w*np.sum((self.dna[n,0,:,i:i+self.motifs.shape[3]]*self.motifs[k,0,:,:]).flatten())+ b)
        return output_mat

# def slice_two_update(X, w):
#     return X * w, X+w

def motif_slice_updatei(X,w):
    out,_ = theano.scan(fn = ScaleFilter, outputs_info = None, sequences = [X,w])
    return out

def ScaleFilter(conv_output, weight):
    return weight*conv_output

def Iterator(idx, X, w, Z):
    val = X[:,idx,:,:]*w[idx]
    Z = T.set_subtensor(Z[:,idx,:,:], val)

def Rectifier(x):
    return np.maximum(x,0)

def ScaleMotifs(F, W):
    return F * W


class FilterZeroPad(object):
    def __init__(self, x, input_shape, motif_tensor, motif_shape, max_pool_size = 3, activation = T.nnet.relu, max_pool=True):
        #self.F = T.ftensor4('F')
        #self.X = T.ftensor4('X')
        #input_shape is just the input_tensor.shape
        #self.X = theano.shared(value = np.asarray(input_tensor, dtype = theano.config.floatX),  name = 'X', borrow = True)
        #self.F = theano.shared(value = np.asarray(motif_tensor, dtype = theano.config.floatX), name = 'F', borrow = True)
        #self.W = theano.shared(value = 2*np.ones(input_shape[0], dtype = theano.config.floatX), name = 'W', borrow = True)
        #self.b = theano.shared(value = 2*np.ones(motif_tensor.shape[0], dtype = theano.config.floatX), name = 'b', borrow = True)
        self.input_shape = input_shape
        rng = np.random.RandomState()
        self.motif_shape = motif_shape
        self.pooling_size = (1,max_pool_size)
        self.post_max_pool_size,_ = divmod(self.input_shape[3] - self.motif_shape[3] + 1, max_pool_size)
        self.X = x
        #self.F = theano.shared(motif_tensor, borrow = True)
        self.F = motif_tensor
        #self.W_values = np.asarray(np.ones((motif_shape[0]), dtype= theano.config.floatX))
        self.W_values = np.asarray(rng.uniform(low = -np.sqrt(6./(self.motif_shape[0])),
                                            high = np.sqrt(6./(self.motif_shape[0])),
                                            size = (self.motif_shape[0],)),
                                dtype=theano.config.floatX)
        #initalize the W and b vectors to the total number of motifs inputted
        self.W = theano.shared(value = self.W_values, name ='W1', borrow = True)
        #self.W = theano.shared(value = np.ones((motif_shape[0]), dtype = theano.config.floatX), name = 'W1', borrow = True)
        #self.W = theano.shared(value = np.arange((motif_shape[0]), dtype = theano.config.floatX), name = 'W1', borrow = True)
        self.b = theano.shared(value = 0.001*np.ones((motif_shape[0]), dtype = theano.config.floatX), name = 'b1', borrow = True)
        self.input_shape = [1,4,input_shape[3]]
        self.filter_shape = [1,4,motif_shape[3]]
        self.FW, updates = theano.scan(fn= ScaleMotifs, sequences = [self.F, self.W], outputs_info = None)
        self.conv_out = conv2d(input= self.X, filters = self.FW,  border_mode = 'valid', filter_flip=False)
        if max_pool == True:
            self.post_pool = max_pool_2d(input = self.conv_out, ds = self.pooling_size, ignore_border = True)
            output = activation(self.post_pool + self.b.dimshuffle('x',0,'x','x'))
        else:
            output = activation(self.conv_out + self.b.dimshuffle('x',0,'x','x'))
        #add norms
        #self.b_L1_norm = self.b.norm(1)
        self.b_L2_norm = self.b.norm(2)
        self.matrix = T.ftensor4('x')
        self.W_L1_norm = self.W.norm(1)
        self.W_L2_norm = self.W.norm(2)
        self.regularize_params = [ 15*self.W_L1_norm, 2*self.W_L2_norm, self.b_L2_norm]
        #self.model(pooling_size = (1,max_pool_size))

        #self.conv_out = conv2d(input= self.X, filters = self.FW,  border_mode = 'valid', filter_flip=False)
        #self.conv_out = conv2d(input= self.X, filters = self.F, filter_shape = self.filter_shape, input_shape = self.input_shape, border_mode = 'valid', filter_flip = False)
        #conv2d gives output of (batch size, output channels, output row, output columns)

        # self.conv_out.dimshuffle(3,0,1,2) #recast tensor so we can scan along the axis of each depth slice (motif)
        # self.conv_weighted, updates = theano.scan(fn = ScaleFilter, outputs_info = None, sequences = [self.conv_out, self.W])
        # self.conv_weighted.dimshuffle(1,2,3,0) #recast tensor back to original form by undoing the dimshuffle from earlier
        #if max_pool == True:
            #self.post_pool = max_pool_2d(input = self.conv_out, ds = self.pooling_size, ignore_border = True)#should ignore_border be false? Double check with Yue later.
            #self.post_pool_weighted, updates = theano.scan(fn=motif_slice_update, sequences = self.post_pool, non_sequences = self.W)
            #self.post_pool_dimshuffled = self.post_pool.dimshuffle(1,0,2,3)
            #self.weighted_post_pool, _ = theano.scan(fn=ScaleFilter, sequences= [self.post_pool_dimshuffled, self.W])
            #self.weighted_post_pool = self.weighted_post_pool.dimshuffle(1,0,2,3)
            #output = activation(self.weighted_post_pool + self.b.dimshuffle('x', 0, 'x', 'x'))
        #else:
            #self.conv_out.dimshuffle(1,0,2,3)
            #self.weighted_conv_out, _ = theano.scan(fn=ScaleFilter, sequences=[self.conv_out, self.W])
            #self.weighted_conv_out.dimshuffle(1,0,2,3)
            #self.conv_weighted, updates = theano.scan(fn=motif_slice_update, sequences = self.conv_out, non_sequences = self.W)
            #output = activation(self.weighted_conv_out + self.b.dimshuffle('x', 0, 'x', 'x')) #non-linearities
        self.output = output 
        self.params = [self.W, self.b]

    def first_layer_test(self):
        self.first_layer_func = theano.function([], self.output)
        self.conv_test_func = theano.function([], self.conv_out)
        # self.weighted_func = theano.function([], self.weighted_post_pool)
        # self.post_pool_func = theano.function([], self.post_pool)
        # self.dimshuffle = theano.function([], self.post_pool_dimshuffled)
        self.FW_result = theano.function([], self.FW)
def FC_output(X, W, b):
    #return activation(T.dot(X,W).flatten(1) + b)
    return T.dot(X, W) + b

class PoolingLayer(object):
    def __init__(self, input, filter_shape, input_shape, poolsize=(1, 3)):
        self.input = input
        rng = np.random.RandomState()
        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize))
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype=theano.config.floatX), name = 'W2', borrow=True)
        b_values =0.01* np.ones((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name = 'b2',borrow=True)
        conv_out = conv2d(input=input, filters=self.W, filter_shape=filter_shape, input_shape=input_shape)
        pooled_out = max_pool_2d(input=conv_out, ds=poolsize, ignore_border=True)
        # Using tanh activation layer
        self.output = T.nnet.relu(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]
        #self.b_L1_norm = self.b.norm(1)
        self.b_L2_norm = self.b.norm(2)
        self.W_L1_norm = self.W.norm(1)
        self.W_L2_norm = self.W.norm(2)
        self.regularize_params = [ self.W_L1_norm, self.W_L2_norm, self.b_L2_norm]


class FullyConnected(object):
    def __init__(self, input, n_in, n_out, minibatch_size, W=None, b=None, activation = T.nnet.relu, dropout = False):
        self.X = input
        rng = np.random.RandomState()
        if W is None:
            W_values = np.asarray(rng.uniform(low = -np.sqrt(6./(n_in + n_out)),
                                              high = np.sqrt(6./(n_in + n_out)),
                                              size = (n_in, n_out)),
                                  dtype=theano.config.floatX)
            W = theano.shared(value = W_values, name ='W3', borrow = True)
        if b is None:
            b_values = np.asarray(0.01*np.ones((n_out,)), dtype=theano.config.floatX)
            b = theano.shared(value = b_values, name = 'b3', borrow = True)
        self.W = W
        self.b = b
        self.lin_output, update = theano.scan(fn=FC_output, sequences = self.X, non_sequences = [self.W, self.b])
        self.output = T.nnet.relu(self.lin_output)
        self.params = [self.W, self.b]
        #self.b_L1_norm = self.b.norm(1)
        self.b_L2_norm = self.b.norm(2)
        self.W_L1_norm = self.W.norm(1)
        self.W_L2_norm = self.W.norm(2)
        self.regularize_params = [ self.W_L1_norm, self.W_L2_norm, self.b_L2_norm]
#generic conv + pool layer
class LogisticRegression(object):
    def __init__(self, input, n_in, n_out, W = None, b = None):
        rng = np.random.RandomState()
        self.input = input
        self.W = theano.shared(value=np.ones((n_in,n_out), dtype=theano.config.floatX), name='W4', borrow=True)
        if W is None:
            W_values = np.asarray(rng.uniform(low = -np.sqrt(6./(n_in + n_out)),
                                              high = np.sqrt(6./(n_in + n_out)),
                                              size = (n_in, n_out)),
                                  dtype=theano.config.floatX)
            W = theano.shared(value = W_values, name ='W4', borrow = True)
        self.b = theano.shared(value=0.01*np.ones((n_out,), dtype=theano.config.floatX), name='b4', borrow=True)
        #multi-class classifier 
        #self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        #self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        #self.lin_output, update = theano.scan(fn=FC_output, sequences = self.input, non_sequences = [self.W, self.b])
        #self.p_y_given_x = T.nnet.sigmoid(self.lin_output)
        #self.p_y_given_x.flatten(1)
        #self.p_y_given_x = self.p_y_given_x[:,0]
        self.p_y_given_x = T.nnet.sigmoid(T.dot(self.input, self.W) + self.b)
        self.p_y_given_x = self.p_y_given_x[:,0]
        self.y_pred = self.p_y_given_x > 0.5
        self.params = [self.W, self.b]
        #self.b_L1_norm = self.b.norm(1)
        self.b_L2_norm = self.b.norm(2)
        self.W_L1_norm = self.W.norm(1)
        self.W_L2_norm = self.W.norm(2)
        self.regularize_params = [ self.W_L1_norm, self.W_L2_norm, self.b_L2_norm]

    def negative_log_likelihood(self, y):
        return -T.sum(y*T.log(self.p_y_given_x))

    def cross_entropy(self, y):
        return T.mean(T.nnet.categorical_crossentropy(self.p_y_given_x, y))
    
    def binary_crossentropy(self,y, weights):
        return T.mean(weights*T.nnet.binary_crossentropy(self.p_y_given_x[0], y))

    def test_binary_crossentropy(self,y):
        return T.mean(T.nnet.binary_crossentropy(self.p_y_given_x[0],y))

    def errors(self, y):
        return T.mean(T.neq(self.y_pred,y))
        # check if y has same dimension of y_pred
        # if y.ndim != self.y_pred.ndim:
        #     raise TypeError('y should have the same shape as self.y_pred',('y', y.type, 'y_pred', self.y_pred.type))
        # # check if y is of the correct datatype
        # if y.dtype.startswith('int'):
        #     # the T.neq operator returns a vector of 0s and 1s, where 1
        #     # represents a mistake in prediction
        #     return T.mean(T.neq(self.y_pred, y))
        # else:
        #     raise NotImplementedError()

class ConvModel(object):
    def __init__(self, debug = False, load = False, save = False):
        self.debug = debug
        self.load = load
        self.save = save
        self.x = T.ftensor4('x')
        self.y = T.ivector('y')
        self.weights = T.fvector('weights')
        self.index = T.lscalar()
        self.start_time = time.time()

    def fit(self,  dna_tensor, labels, test_dna_tensor, test_labels, motif_tensor, weight_array, lambd=0.0001, batch_size = 400, epochs = 25,custom_layer_max_pool_size = 4, conv_layer_conv_width = 15, conv_layer_n_filters = 150, conv_layer_pool_size = 3, FC_output = 50, gd_algorithm = "gd", gd_learning_rate = 0.0005):
        """Fit the custom convolutional neural network

        Keyword arguments:
        dna_tensor -- training tensor (batch_size,1,4,150)
        labels -- binary labels for the training set
        test_dna_tensor -- test tensor
        test_labels -- test labels
        motif_tensor -- motif_tensor (currently 0 or 0.25 padded)
        lambd -- shared L1/L2 regularization term for W and b for all layers
        batch_size -- batch size
        epochs -- number of epochs to loop over the training data
        custom_layer_max_pool_size -- pooling size for the first layer
        FC_output -- number of outputs from the fully connected layer
        """
        #declare some shape parameters and batch size
        self.layers = []
        self.motif_tensor = theano.shared(motif_tensor,borrow=True)
        self.motif_tensor_shape = motif_tensor.shape
        self.dna_tensor_shape = dna_tensor.shape
        self.test_dna_tensor_shape = test_dna_tensor.shape
        self.batch_size = batch_size
        self.weight_array = theano.shared(weight_array, borrow = True)
        self.rng = np.random.RandomState()
        #make everything a shared variable...or else it won't be able to slice during minibatches (theano bug maybe?)
        self.dna_tensor = theano.shared(dna_tensor, borrow = True)
        self.labels = theano.shared(labels, borrow = True)
        self.test_dna_tensor = theano.shared(test_dna_tensor, borrow =True)
        self.test_labels = theano.shared(test_labels, borrow = True)
        self.true_labels = copy.deepcopy(test_labels)
        #calculate the number of features per motif to the 2nd hidden layer (FC in this current architecture) based on first layer outputs and first layer pooling size
        num_first_layer_features,_ = divmod((self.dna_tensor_shape[-1] - self.motif_tensor_shape[-1] + 1), custom_layer_max_pool_size)
        #now we multiply per motif features by the total number of motifs
        num_second_layer_features,_ = divmod((num_first_layer_features - conv_layer_conv_width+ 1), conv_layer_pool_size) 
        num_inputs_to_FC = num_second_layer_features * conv_layer_n_filters
        print "Number of inputs to FC layer is %i"%(num_inputs_to_FC)
        self.layers.append(FilterZeroPad(x=self.x,
                                         input_shape=(self.batch_size, self.dna_tensor_shape[1], self.dna_tensor_shape[2], self.dna_tensor_shape[3]),
                                         motif_tensor=self.motif_tensor,
                                         motif_shape=self.motif_tensor_shape,
                                         max_pool_size=custom_layer_max_pool_size))
        self.layers.append(PoolingLayer(input = self.layers[-1].output,
                                        input_shape = (self.batch_size, self.motif_tensor_shape[0], 1, num_first_layer_features),
                                        filter_shape = (conv_layer_n_filters, self.motif_tensor_shape[0],1,conv_layer_conv_width),
                                        poolsize = (1,conv_layer_pool_size),
        ))
        self.layers.append(FullyConnected(input = self.layers[-1].output.flatten(2), n_in = num_inputs_to_FC, n_out=FC_output, minibatch_size = self.batch_size))
        self.layers.append(LogisticRegression(input = self.layers[-1].output,n_in = FC_output, n_out = 1))

        #specify the parameters and regularization parameters
        self.full_params = self.layers[0].params + self.layers[1].params + self.layers[2].params + self.layers[3].params
        self.regularize_params = self.layers[0].regularize_params + self.layers[1].regularize_params + self.layers[2].regularize_params + self.layers[3].regularize_params

        # self.train_givens = {self.x: dna_tensor[self.index*self.batch_size:(self.index + 1)*self.batch_size],
        #specify the training and test givens, accounting for mini-batching
        #                self.f: motif_tensor,
        #                self.y: labels[self.index*batch_size:(self.index + 1)*batch_size]
        # }
        # self.test_givens = {self.x: test_dna_tensor[self.index*batch_size:(self.index + 1)*batch_size],
        #                self.f: motif_tensor,
        #                self.y: test_labels[self.index*batch_size:(self.index + 1)*batch_size]
        # }
        X = Y*alpha-Z/(N)
        #create cost functions - regularization first, and then the full cost function 
        self.regularization_cost = T.sum(self.regularize_params)
        #self.cost = self.layers[-1].binary_crossentropy(self.y) + lambd * self.regularization_cost
        self.cost = self.layers[-1].binary_crossentropy(self.y, self.weights)+lambd*self.regularization_cost
        self.test_cost = self.layers[-1].test_binary_crossentropy(self.y)
        #self.cost = self.layers[-1].negative_log_likelihood(self.y) + lambd * self.regularization_cost
        #create the updates using specified algorithm
        if gd_algorithm == "adam":
            self.updates = adam(self.cost, self.full_params, learning_rate = gd_learning_rate)
        elif gd_algorithm == "RMSprop":
            self.updates = RMSprop(self.cost, self.full_params, lr = gd_learning_rate)
        elif gd_algorithm == "gd":
            self.grads = T.grad(self.cost, self.full_params)
            self.hessian = T.grad(self.grads, self.full_params)
            self.updates = [(param_i, param_i - gd_learning_rate * grad_i) for param_i, grad_i in zip(self.full_params, self.grads)]
        #specify the theano training and testing functions
        self.train_givens = {self.x: self.dna_tensor[self.index*self.batch_size:(self.index + 1)*self.batch_size],
                             self.y: self.labels[self.index*self.batch_size:(self.index + 1)*self.batch_size],
                             self.weights: self.weight_array[self.index*self.batch_size:(self.index + 1)*self.batch_size]
        }
        self.test_givens = {self.x: self.test_dna_tensor[self.index*self.batch_size:(self.index + 1)*self.batch_size],
                    self.y: self.test_labels[self.index*self.batch_size:(self.index + 1)*self.batch_size]
        }
        #self.test_givens = {self.x: self.test_dna_tensor[self.index*self.batch_size:(self.index + 1)*self.batch_size],
        #}
        self.debug_givens = {self.x: self.dna_tensor[self.index * self.batch_size:(self.index + 1)*self.batch_size],
                             self.y: self.labels[self.index * self.batch_size:(self.index + 1)*self.batch_size],
                             self.weights: self.weight_array[self.index*self.batch_size:(self.index + 1)*self.batch_size]
        }
        #self.train_model = theano.function([self.index], self.cost, updates = self.updates, givens = self.train_givens, on_unused_input = 'warn')
        #self.train_model = theano.function([self.index], self.cost, updates = self.updates, givens = self.train_givens, on_unused_input = 'warn', mode = DebugMode(check_py_code=False))
        self.train_model = theano.function([self.index], self.cost, updates = self.updates, givens = self.train_givens, on_unused_input = 'warn', mode = NanGuardMode(nan_is_error=True, inf_is_error = True, big_is_error = True))
        self.debug_model = theano.function([self.index], self.layers[-1].binary_crossentropy(self.y, self.weights), givens = self.debug_givens, on_unused_input = 'warn')
        #self.test_model = theano.function([self.index], self.layers[-1].errors(self.y), givens = self.test_givens, on_unused_input = 'warn')
        self.score_model = theano.function([self.index], self.test_cost, givens = self.test_givens, on_unused_input = 'warn')
        #specify more givens for training
        self.y_given_x = theano.function([self.index], self.layers[-1].p_y_given_x, givens = self.debug_givens, on_unused_input = 'warn')
        self.test_y_given_x = theano.function([self.index], self.layers[-1].p_y_given_x, givens = self.test_givens, on_unused_input = 'warn')
        self.n_train_batches,_ = divmod(self.dna_tensor_shape[0],self.batch_size)
        self.n_test_batches,_ = divmod(self.test_dna_tensor_shape[0],self.batch_size)
        self.cur_epoch = 0
        self.epochs = epochs
        self.n_iters = self.epochs * self.n_train_batches
        self.train_cost = 9999999999
        self.train_cost_records = []
        self.test_cost_records = []
        self.test_accuracy_records = []
        #self.start_train()

    def debug_train(self):
        for minibatch_index in xrange(self.n_train_batches):
            self.debug_output = self.debug_model(minibatch_index)
            print(self.debug_output.shape)
            print(self.debug_output)
            print(self.y.shape)
            print(self.y)
            break

    def start_train(self):
        print("Starting training! Initalization of the model took %0.03f seconds"%(time.time() - self.start_time))
        print("Starting training with batch size %i"%(self.batch_size))
        self.start_time = time.time()
        self.training_cost = np.zeros((self.epochs, self.n_train_batches))
        while self.cur_epoch < self.epochs:
            self.cur_epoch += 1
            for minibatch_index in xrange(self.n_train_batches):
                self.train_cost = self.train_model(minibatch_index)
                self.train_cost_records.append(self.train_cost)
                self.training_cost[self.cur_epoch, minibatch_index] = self.train_cost
                # if minibatch_index % 100 == 0:
                #     print("Minibatch %i on epoch %i complete in %0.04f seconds with %0.04f training cost"%(minibatch_index, self.cur_epoch, time.time()-self.start_time, self.train_cost.flatten()[0]))
            print("Training epoch %i took %0.03f seconds and has current training cost %0.03f"%(self.cur_epoch, time.time()-self.start_time, self.train_cost))
            if self.cur_epoch > 0:
                self.result = []
                print("Evaluating test loss!")
                test_losses = [self.score_model(i) for i in xrange(self.n_test_batches)]
                self.avg_test_loss = np.mean(test_losses)
                self.test_cost_records.append(self.avg_test_loss)
                print("Training cost at epoch %i is %0.03f"%(self.cur_epoch, self.avg_test_loss))
                print("Evaluating test accuracy!")
                #test_accuracy = [self.test_model(i) for i in xrange(self.n_test_batches)]
                #self.avg_test_accuracy = np.mean(test_accuracy)
                #self.test_accuracy_records.append(self.avg_test_accuracy)
                for batch_index in range(self.n_test_batches):
                    self.batch_result = self.test_y_given_x(batch_index)
                    self.result = np.hstack((self.result, self.batch_result))
                classObj = ClassificationResult(self.true_labels[:self.n_test_batches * self.batch_size].flatten(), self.result.flatten()) 
                classObj.self_binary(self.true_labels[:self.n_test_batches * self.batch_size].flatten().astype(bool), self.result.flatten())
                print(classObj.results)
                self.classObj = classObj
        print("Training complete\n")
        self.final_test_loss = self.test_cost_records[-1]
        self.final_training_loss = self.train_cost_records[-1]
        self.training_time = time.time()-self.start_time
        print("Final time taken is %0.03f, training loss is %0.03f, and test loss is %0.03f"%(self.training_time, self.final_training_loss, self.final_test_loss))

    def predict(self, dna_tensor, batch_size = 250):
        predict_start_time = time.time()
        self.predict_givens = {self.x: self.test_dna_tensor[self.index * self.batch_size: (self.index + 1)* self.batch_size]}
        self.pred_model = theano.function([self.index], self.layers[-1].p_y_given_x,
                                          givens = self.predict_givens, on_unused_input='warn', allow_input_downcast = True)
        self.result = []
        for batch_index in range(self.n_test_batches):
            batch_result = self.pred_model(batch_index)
            self.result = np.hstack((self.result, batch_result))
        print("Finished prediction in %0.03f seconds"%(time.time() - predict_start_time))
        return self.result

    def save(self, path = "TheanoMotifModel.pck"):
        path = os.path.join(os.path.split(__file__)[0], path)
        with open(path, 'wb') as output:
            pickle.dump(self.layers, output, pickle.HIGHEST_PROTOCOL)
        print("Saving completed")

    def load(self, path = "TheanoMotifModel.pck"):
        path = os.path.join(os.path.split(__file__)[0], path)
        with open(path, "rb") as input_file:
            self.layers = pickle.load(input_file)
