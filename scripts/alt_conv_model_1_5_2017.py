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
from sklearn.metrics import auc, log_loss, precision_recall_curve, roc_auc_score, roc_curve
from prg.prg import create_prg_curve, calc_auprg
from theano import pp
from sklearn.neural_network import MLPClassifier
from sklearn.grid_search import ParameterGrid
from sklearn.preprocessing import scale, StandardScaler 
np.random.seed(42)
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from common_utils import ClassificationResult
from common_utils import BinaryClassProcessor

class ConvLayer(object):
    def __init__(self, motifs):
        self.motifs = motifs
        self.motif_shape = self.motifs.shape
        self.W = theano.shared(value=self.motifs, name='W', borrow=True)
        self.X = T.ftensor4('X')
        self.conv_out = conv2d(input=self.X, filters=self.W,
                               border_mode="valid", filter_flip=False)
        self.conv_func = theano.function([self.X], self.conv_out, on_unused_input='warn')

# class ConvLayer(object):
#     def __init__(self, motifs):
#         self.motifs = motifs
#         self.motif_shape = self.motifs.shape
#         self.W = theano.shared(value=self.motifs, name='W', borrow=True)
#         self.X = T.ftensor4('X')
#         self.conv_out = conv2d(input=self.X, filters=self.W,
#                                border_mode="valid", filter_flip=False)
#         self.conv_func = theano.function([self.X, self.conv_out])

class ConvPredictor(object):
    def __init__(self, X, X_RC, Y, motif_shape, ConvLayerObj, features_df, add_features = True, keep_output = False, RC = True, combine_RC = False, RC_max = True, add_RC = False, debug = False):
        self.X = np.asarray(X, dtype=theano.config.floatX)
        #self.motifs = np.asarray(motfs, dtype = theano.config.floatX)
        self.Y = Y
        self.RC = RC
        self.combine_RC = combine_RC 
        self.RC_max = RC_max
        self.debug = debug
        self.X_RC = np.asarray(X_RC, dtype=theano.config.floatX)
        self.motif_shape = motif_shape
        self.features_df = features_df
        self.features_mat = self.features_df.values
        self.ConvLayerObj = ConvLayerObj
        self.feature_map_shape = [self.motif_shape[0], self.X.shape[-1] - self.motif_shape[-1] + 1]
        self.input_shape = self.X.shape
        self.X_conv, self.X_full_conv = self.Conv(self.X, self.X.shape, self.ConvLayerObj, keep_output=keep_output)
        if RC and combine_RC:
            self.X_RC_conv, self.X_RC_full_conv = self.Conv(self.X_RC, self.X_RC.shape, self.ConvLayerObj, keep_output=keep_output)
            self.X_X_RC_max = np.zeros_like(self.X_conv)
            if self.RC_max:
                for i in range(self.X_conv.shape[0]):
                    for j in range(self.X_conv.shape[1]):
                        if add_RC:  #add convolutions from forward and reverse strands
                            self.X_X_RC_max[i,j] = self.X_conv[i,j] + self.X_RC_conv[i,j]
                        else: #else take the max of the two. 
                            self.X_X_RC_max[i,j] = max(self.X_conv[i,j], self.X_RC_conv[i,j])
                if add_features:
                    self.X_comb_conv = np.hstack((self.X_X_RC_max, self.features_mat))
                else:
                    self.X_comb_conv = self.X_X_RC_max
            else:
                self.X_X_RC_comb = np.hstack((self.X_conv, self.X_RC_conv)) #merge features into one supervector
                if add_features:
                    self.X_comb_conv = np.hstack((self.X_X_RC_comb, self.features_mat))
                    if self.debug:
                        print(self.X_conv.shape)
                        print(self.X_conv)
                        print(self.X_RC_conv.shape)
                        print(self.X_X_RC_comb.shape)
                        print(self.features_mat.shape)
                        print("Final shape is")
                        print(self.X_comb_conv.shape)
                else:
                    self.X_comb_conv = self.X_X_RC_comb
        elif RC and not combine_RC:
            self.X_RC_conv, self.X_RC_full_conv = self.Conv(self.X_RC, self.X_RC.shape, self.ConvLayerObj, keep_output=keep_output)
            if add_features:
                if debug:
                    print(self.features_mat.shape)
                    print(self.X_RC_conv.shape)
                    print(self.X_conv.shape)
                self.X_RC_conv_features = np.hstack((self.X_RC_conv, self.features_mat))
                self.X_conv_features = np.hstack((self.X_conv, self.features_mat))
                self.X_comb_conv = np.vstack((self.X_RC_conv_features, self.X_conv_features))
            else:
                self.X_comb_conv = np.vstack((self.X_RC_conv, self.X_conv))
            Y_copy = np.copy(self.Y)
            self.Y = np.hstack((Y_copy, Y_copy))
        elif not RC:
            if add_features:
                self.X_comb_conv = np.hstack((self.X_conv, self.features_mat))
            else:
                self.X_comb_conv = self.X_conv
        self.X_comb_conv_shape = self.X_comb_conv.shape
        self.X_comb_conv_width = self.X_comb_conv_shape[1]
        self.classifier_dict = {}

    def Correct_RC_index(self,index):
        index_copy = np.copy(index)
        index_RC = self.X_conv.shape[0] + index_copy
        return np.hstack((index_copy, index_RC))

    def Conv(self, X, input_shape, ConvLayerObj, minibatch_size=2000, max_axis = 1, RC= False, keep = False):
        minibatches, rem = divmod(input_shape[0], minibatch_size)
        # output = np.zeros((X.shape[0], self.motif_shape[1], 0, X.shape[-1] - self.motif_shape[-1] + 1 ))
        # output = np.zeros((X.shape[0], X.shape[-1] - self.motif_shape[-1] + 1 ))
        output = np.zeros((X.shape[0], self.motif_shape[0]))
        if keep:
            output_full_conv = np.zeros((input_shape[0], self.feature_map_shape[-1]))
        for idx in range(minibatches):
            cur_conv_output = ConvLayerObj.conv_func(
                X[idx*minibatch_size:(idx+1)*minibatch_size])  #output is index, filter, _, _
            if keep:
                output_full_conv[idx*minibatch_size:(idx+1)*minibatch_size] = cur_conv_output
            # print(cur_conv_output.shape)
            for cur_idx in range(minibatch_size):
                #self.transformed_output =self.cur_conv_output[cur_idx, :, 0, :]
                cur_sub = cur_conv_output[cur_idx, :, 0, :]
                # print(cur_sub.shape)
                cur_max = np.amax(cur_conv_output[cur_idx, :, 0, :], axis=max_axis)
                output[idx*minibatch_size + cur_idx, :] = cur_max
                # break
        if rem > 0:
            cur_conv_output = ConvLayerObj.conv_func(
                X[minibatches*minibatch_size:])
            if keep:
                output_full_conv[minibatch_size*minibatches:] = cur_conv_output
            # print(cur_conv_output.shape)
            for cur_idx in range(cur_conv_output.shape[0]):
                cur_max = np.amax(cur_conv_output[cur_idx, :, 0, :],
                                  axis=max_axis)
                output[minibatches*minibatch_size + cur_idx, :] = cur_max
        if keep:
            return output, output_full_conv
        else:
            return output, None

    def LogisticRegPredict(self, chrom_indices, other_indices, penalty='l2', solver='sag', tol=0.001, C=0.1, max_iter=100, should_scale=True, train_only=False):
        if self.RC and not self.combine_RC:
            chrom_indices = self.Correct_RC_index(chrom_indices)
            other_indices = self.Correct_RC_index(other_indices)
        self.LogisticRegObj = LogisticRegression(class_weight='balanced', penalty=penalty, solver=solver, tol=tol, C=C, max_iter=max_iter, random_state=42)
        if should_scale:
            scaler = StandardScaler()
            scaler.fit(self.X_comb_conv[other_indices])
            X_train= scaler.transform(self.X_comb_conv[other_indices])
            if not train_only:
                X_test = scaler.transform(self.X_comb_conv[chrom_indices])
        else:
            X_train = self.X_comb_conv[other_indices]
            if not train_only:
                X_test = self.X_comb_conv[chrom_indices]
        self.LogisticRegObj.fit(X_train, self.Y[other_indices])
        self.classifier_dict['LogisticReg'] = self.LogisticRegObj
        if train_only:
            return self.LogisticRegObj
        else:
            return self.LogisticRegObj.predict_proba(self.X_test)[:, 1],\
            self.Y[chrom_indices].astype(bool)

    def LogisticRegCVPredict(self, chrom_indices, other_indices, penalty = 'l2', solver = 'lbfgs', tol = 0.0001, n_jobs = 4, max_iter = 150, should_scale = True, train_only = False):
        if self.RC and not self.combine_RC:
            chrom_indices = self.Correct_RC_index(chrom_indices)
            other_indices = self.Correct_RC_index(other_indices)
        self.LogisticRegCVObj = LogisticRegressionCV(class_weight = 'balanced', penalty = penalty, solver = solver, n_jobs = n_jobs,  tol = tol, max_iter = max_iter, random_state = 42)
        if should_scale:
            scaler = StandardScaler()
            scaler.fit(self.X_comb_conv[other_indices])
            X_train= scaler.transform(self.X_comb_conv[other_indices])
            if not train_only:
                X_test = scaler.transform(self.X_comb_conv[chrom_indices])
        else:
            X_train = self.X_comb_conv[other_indices]
            if not train_only:
                X_test = self.X_comb_conv[chrom_indices]
        self.LogisticRegCVObj.fit(X_train, self.Y[other_indices])
        self.classifier_dict['LogisticRegCV'] = self.LogisticRegCVObj
        if train_only:
            return self.LogisticRegCVObj
        else:
            return self.LogisticRegCVObj.predict_proba(X_test)[:,1], self.Y[chrom_indices].astype(bool)

    def ElasticNet(self, chrom_indices, other_indices, alpha = 0.0001, l1_ratio = 0.15, loss = 'log', learning_rate = "optimal", penalty = "elasticnet", should_scale = True, train_only = False):
        if self.RC and not self.combine_RC:
            chrom_indices = self.Correct_RC_index(chrom_indices)
            other_indices = self.Correct_RC_index(other_indices)
        self.ElasticNetObj = SGDClassifier(loss = loss, alpha = alpha, l1_ratio = l1_ratio, penalty = penalty, class_weight = 'balanced', learning_rate = learning_rate, random_state = 42)
        if should_scale:
            scaler = StandardScaler()
            scaler.fit(self.X_comb_conv[other_indices])
            X_train= scaler.transform(self.X_comb_conv[other_indices])
            if not train_only:
                X_test = scaler.transform(self.X_comb_conv[chrom_indices])
        else:
            X_train = self.X_comb_conv[other_indices]
            if not train_only:
                X_test = self.X_comb_conv[chrom_indices]
        self.ElasticNetObj.fit(X_train, self.Y[other_indices])
        self.classifier_dict['ElasticNet']=self.ElasticNetObj
        if train_only:
            return self.ElasticNetObj
        else:
            return self.ElasticNetObj.predict_proba(X_test)[:,1], self.Y[chrom_indices].astype(bool)
    
    def RandomForest(self, chrom_indices, other_indices, n_estimators=500, n_jobs = 4, should_scale = True, train_only = False):
        if self.RC and not self.combine_RC:
            chrom_indices = self.Correct_RC_index(chrom_indices)
            other_indices = self.Correct_RC_index(other_indices)
        self.RandomForestObj = RandomForestClassifier(n_estimators=n_estimators, n_jobs = n_jobs, class_weight = 'balanced')
        if should_scale:
            scaler = StandardScaler()
            scaler.fit(self.X_comb_conv[other_indices])
            X_train= scaler.transform(self.X_comb_conv[other_indices])
            if not train_only:
                X_test = scaler.transform(self.X_comb_conv[chrom_indices])
        else:
            X_train = self.X_comb_conv[other_indices]
            if not train_only:
                X_test = self.X_comb_conv[chrom_indices]
        self.RandomForestObj.fit(X_train, self.Y[other_indices])
        self.classifier_dict['RandomForest'] = self.RandomForestObj
        if train_only:
            return self.RandomForestObj
        else:
            return self.RandomForestObj.predict_proba(X_test)[:,1], self.Y[chrom_indices].astype(bool)

    def GradientBoosting(self, chrom_indices, other_indices, n_estimators=500, should_scale = True, train_only = False):
        if self.RC and not self.combine_RC:
            chrom_indices = self.Correct_RC_index(chrom_indices)
            other_indices = self.Correct_RC_index(other_indices)
        self.GradientBoostingObj = GradientBoostingClassifier(n_estimators=n_estimators)
        if should_scale:
            scaler = StandardScaler()
            scaler.fit(self.X_comb_conv[other_indices])
            X_train= scaler.transform(self.X_comb_conv[other_indices])
            if not train_only:
                X_test = scaler.transform(self.X_comb_conv[chrom_indices])
        else:
            X_train = self.X_comb_conv[other_indices]
            if not train_only:
                X_test = self.X_comb_conv[chrom_indices]
        self.GradientBoostingObj.fit(X_train, self.Y[other_indices])
        self.classifier_dict['GradientBoostingObj'] = self.GradientBoostingObj
        if train_only:
            return self.GradientBoostingObj
        else:
            return self.GradientBoostingObj.predict_proba(X_test)[:,1], self.Y[chrom_indices].astype(bool)

    def MLP(self, chrom_indices, other_indices, hidden_layer_sizes = (100,), should_scale = True, train_only = False):
        if self.RC and not self.combine_RC:
            chrom_indices = self.Correct_RC_index(chrom_indices)
            other_indices = self.Correct_RC_index(other_indices)
        self.MLPObj = MLPClassifier(hidden_layer_sizes = hidden_layer_sizes)
        if should_scale:
            scaler = StandardScaler()
            scaler.fit(self.X_comb_conv[other_indices])
            X_train= scaler.transform(self.X_comb_conv[other_indices])
            if not train_only:
                X_test = scaler.transform(self.X_comb_conv[chrom_indices])
        else:
            X_train = self.X_comb_conv[other_indices]
            if not train_only:
                X_test = self.X_comb_conv[chrom_indices]
        self.MLPObj.fit(X_train, self.Y[other_indices])
        self.classifier_dict['MLP'] = self.MLPObj
        if train_only:
            return self.MLPObj
        else:
            return self.MLPObj.predict_proba(X_test)[:,1], self.Y[chrom_indices].astype(bool)

    def Predict(self, X, Y, predictor = "ElasticNet"):
        classObj = self.classifier_dict[predictor]
        return classObj.predict_proba(X)[:, 1], Y.astype(bool)

    def get_class_weights(self, tar_vec):
        counts = np.zeros(2)
        for idx, val in enumerate(tar_vec):
            if val == 0:
                counts[0] += 1
            elif val == 1: 
                counts[1] += 1
        _sum = np.sum(counts)
        return _sum / float(counts[0]), _sum / float(counts[1])


    def FC_1layer_model(self, chrom_indices, other_indices, patience=40, L1=1, L2=1, dropout_input =0, dropout_hidden=0, n_hidden=5):
        input_shape = self.X_comb_conv.shape[-1]
        self.model = Sequential()
        self.model.add(Dropout(dropout_input, input_shape=(input_shape,)))
        self.model.add(Dense(n_hidden, init='glorot_uniform',
                             W_regularizer=l1l2(L1, L2)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(dropout_hidden))
        self.model.add(Dense(n_hidden, init = 'glorot_uniform', W_regularizer=l1l2(L1,L2)))
        self.model.add(Dropout(dropout_hidden))
        self.model.add(Dense(output_dim=1, init='glorot_uniform',
                             W_regularizer=l1l2(L1, L2)))
        self.model.add(Activation('sigmoid'))
        num_epochs = 130
        learning_rate = 0.0001
        self.adam = keras.optimizers.Adam(lr = learning_rate)
        self.model.compile(optimizer=self.adam, loss = 'binary_crossentropy', metrics=['binary_crossentropy'])
        self.earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=True,  mode='auto')
        neg_weight, pos_weight = self.get_class_weights(self.Y[other_indices])
        self.model.fit(x=self.X_comb_conv[other_indices], y=self.Y[other_indices],
                       batch_size=128, nb_epoch=num_epochs, validation_split=0.1,
                       callbacks=[self.earlyStopping],
                       class_weight={0: neg_weight, 1: pos_weight}, verbose=False)
        #self.model.fit(x = self.X_comb_conv[other_indices], y = self.Y[other_indices], batch_size = 2500, nb_epoch = 150, validation_split = 0.10, callbacks = [self.earlyStopping], shuffle = True, class_weight= 'auto', verbose = False)
        #self.model.fit(x = self.X_comb_conv[other_indices], y = self.Y[other_indices], batch_size = 2500, nb_epoch = 125, shuffle = True, class_weight= {0: neg_weight, 1: pos_weight}, verbose = False)
        pred = self.model.predict(self.X_comb_conv[chrom_indices], batch_size=1000)
        return pred.flatten(), self.Y[chrom_indices].astype(bool)

    def Conv_model(self, chrom_indices, other_indices, input_shape = 133, patience = 20, L1 = 1, L2 = 1, dropout_input = 0.2, dropout_hidden = 0.2, n_hidden = 5):
        self.model = Sequential()
        self.model.add(Dropout(dropout_input, input_shape(input_shape, )))
        #self.model.add(Convolution())


class CvEngine(object):
    def __init__(self,name, motif_tensor, motif_names, merged_tensor, merged_tar,
                 output_dir, features_df, RC = True, keep_output = False, C=0.1, debug=False, penalty='l2', solver='sag'):
        self.name = name
        self.motif_tensor = motif_tensor
        self.merged_tensor = merged_tensor
        self.merged_tar = merged_tar
        self.features_df = features_df
        self.RC = RC
        self.keep_output = keep_output
        self.resultObj = Results(name, output_dir)
        self.debug = debug
        self.C = C
        self.penalty = penalty
        self.solver = solve
        self.ConvLayerObj = ConvLayer(motif_tensor)
        self.motif_names = motif_names
        self.convObj = ConvPredictor(merged_tensor, merged_tar, self.motif_tensor.shape,
                                     self.ConvLayerObj, self.features_df, RC = self.RC)

    def start_CV(self, CV_dict, CV_algo = 'elasticnet', **args):
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
            if CV_algo == 'LogisticRegCV':
                Y_pred, Y_true = self.convObj.LogisticRegCVPredict(pos_indices, neg_indices,**args)
                reg_weights = self.convObj.LogisticRegCVObj.coef_.T.flatten()
                reg_bias = self.convObj.LogisticRegCVObj.coef_.T.flatten()
            elif CV_algo == 'ElasticNet':
                Y_pred, Y_true = self.convObj.ElasticNet(pos_indices, neg_indices, **args)
                reg_weights = self.convObj.ElasticNetObj.coef_.T.flatten()
                reg_bias = self.convObj.ElasticNetObj.coef_.T.flatten()
            elif CV_algo == 'LogisticReg':
                Y_pred, Y_true = self.convObj.LogisticRegPredict(pos_indices, neg_indices, **args)
                reg_weights = self.convObj.LogisticRegObj.coef_.T.flatten()
                reg_bias = self.convObj.LogisticRegObj.coef_.T.flatten()
            elif CV_algo == 'RandomForest':
                Y_pred, Y_true = self.convObj.RandomForestObj(pos_indices, neg_indices, **args)
                reg_weights = self.convObj.RandomForestObj.feature_importances_
                reg_biase = self.convObj.randomForestObj.feature_importances_
            if np.amax(Y_true) == 0:
                print "Skipping %s because no positive example"
            reg_weights = pd.Series(data=reg_weights, index=np.arange(len(reg_weights)) ,name=self.name + '_' + chrom)
            self.resultObj.add_cv_result(Y_pred, Y_true, chrom, reg_weights, reg_bias, pos_indices)
            idx += 1
            print("Completed lin reg on chromsome %s in %0.04f seconds"%(chrom, time.time() - start))
            if self.debug:
                if idx > 2:
                    break

    def start_CV_NN(self, CV_dict, **args):
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
            Y_pred, Y_true = self.convObj.FC_1layer_model(pos_indices, neg_indices,**args)
            if np.amax(Y_true) == 0:
                print "Skipping %s because no positive example"
            reg_weights = np.random.rand(self.motif_tensor.shape[0])
            reg_bias = [0]
            reg_weights = pd.Series(data=reg_weights, index=self.motif_names,
                                    name=self.name + '_' + chrom)
            self.resultObj.add_cv_result(Y_pred, Y_true, chrom, reg_weights,
                                         reg_bias, pos_indices)
            idx += 1
            print("Completed NN on  chromsome %s in %0.04f seconds"
                  %(chrom, time.time() - start))
            if self.debug:
                if idx > 2:
                    break
                
    def summarize(self, dump_indices=False, dump_preds=False, dump_labels=False, dump_weights_bias=False, prefix=''):
        self.resultObj.summarize(prefix=prefix)
        if dump_indices:
            self.resultObj.dump_indices(prefix=prefix)
        if dump_preds:
            self.resultObj.dump_preds(prefix=prefix)
        if dump_labels:
            self.resultObj.dump_true(prefix=prefix)
        if dump_weights_bias:
            self.resultObj.dump_weights_bias(prefix=prefix)

    
class Results(object):
    def __init__(self, data_name, output_dir):
        self.data_name = data_name  # ie
        self.per_chrom_results_dict = {}
        self.per_chrom_preds = {}
        self.per_chrom_true = {}
        self.per_chrom_indices = {}
        self.all_probs = np.empty([1, ])
        self.all_true = np.empty([1, ], dtype=bool)
        self.output_dir = output_dir
        self.motif_weights = []
        self.motif_biases = []
        self.column_names = ''

    def ensure_dir(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def add_cv_result(self, pred, true, chrom, motif_weight, motif_bias,
                      indices):
        self.per_chrom_preds[chrom] = pred
        self.per_chrom_true[chrom] = true
        self.all_probs = np.append(self.all_probs.flatten(), pred.flatten())
        self.all_true = np.append(self.all_true.flatten(), true.flatten())
        cur_resultObj = ClassificationResult(true.flatten(), pred.flatten(),
                                             name=self.data_name + '_' + chrom)
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
        self.cum_resultObj = ClassificationResult(self.all_true.flatten(),
                                                  self.all_probs.flatten(),
                                                  name=self.data_name + '_cum')
        self.cum_result = self.cum_resultObj.binary(self.all_true,
                                                    self.all_probs)
        self.str_cum_result = self.cum_resultObj.convert_to_str(self.cum_result)

    def summarize(self, output_file_name='', prefix=''):
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

    def dump_indices(self, output_dir='', prefix=''):
        cur_output_dir = ''
        if len(output_dir) > 1:
            cur_output_dir = output_dir + "indices/"
        else:
            cur_output_dir = self.output_dir + "indices/"
        self.ensure_dir(cur_output_dir)
        for chrom in sorted(self.per_chrom_indices.iterkeys()):
            indices = self.per_chrom_indices[chrom]
            cur_output_file = (cur_output_dir + chrom + '_' +
                               prefix + '_pos_indices.txt')
            fo = open(cur_output_file, 'w')
            for idx, val in enumerate(indices):
                fo.write("%i\n" % (val))
            fo.close()

    def process_weights_bias(self):
        self.weights_df = pd.DataFrame(self.motif_weights)
        #self.biases_df = pd.DataFrame(self.biases_df)

    def dump_weights_bias(self, output_file_dir='', prefix=''):
        self.process_weights_bias()
        if len(output_file_dir) > 1:
            cur_output_dir = output_file_dir
        else:
            cur_output_dir = self.output_dir + 'weights/'
        self.ensure_dir(cur_output_dir)

        weights_output_file = cur_output_dir + prefix + '_reg_weights.txt'
        bias_output_file = cur_output_dir + prefix + '_bias.txt'
        self.weights_df.to_csv(weights_output_file, sep = "\t")
        fo = open(bias_output_file, 'w')
        for idx, val in enumerate(self.motif_biases):
            fo.write("%0.04f\n"%(val))
        fo.close()

    def dump_preds(self, output_dir='', prefix=''):
        cur_output_dir = ''
        if len(output_dir)>1:
            cur_output_dir = output_dir
        else:
            cur_output_dir = self.output_dir + 'preds/'
        self.ensure_dir(cur_output_dir)

        for chrom in sorted(self.per_chrom_preds.iterkeys()):
            predicted_prob = self.per_chrom_preds[chrom]
            cur_output_file = (cur_output_dir + chrom + '_' + prefix +
                               '_predicted_prob.txt')
            fo = open(cur_output_file, 'w')
            for idx, val in enumerate(predicted_prob):
                fo.write("%0.04f\n"%(val))
            fo.close()

    def dump_true(self, output_dir='', prefix=''):
        cur_output_dir = ''
        if len(output_dir) > 1:
            cur_output_dir = output_dir
        else:
            cur_output_dir = self.output_dir + 'labels/'
        self.ensure_dir(cur_output_dir)

        for chrom in sorted(self.all_true.iterkeys()):
            true = self.all_true[chrom]
            cur_output_file = (cur_output_dir + chrom + '_' +
                               prefix + '_labels.txt')
            fo = open(cur_output_file, 'w')
            for idx, val in enumerate(true):
                fo.write("%0.04f\n" % (val))
            fo.close()
