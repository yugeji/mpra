import numpy as np
import os
import theano
import pandas as pd
import matplotlib.pyplot as plt
import time
from common_utils import ClassificationResult
from common_utils import BinaryClassProcessor
from keras_motif import DoubleKerasModel 
from sklearn.preprocessing import StandardScaler

class DataConvolver(object):
    def __init__(self, X, X_RC, Y, motif_shape, should_conv = True, ConvLayerObj = None, features_mat = None, add_features = True, keep_output = False, RC = True, combine_RC = True, RC_combine_mode = 'max', name = "", debug = False):
        self.X = np.asarray(X, dtype=theano.config.floatX)
        self.X_RC = np.asarray(X_RC, dtype=theano.config.floatX)
        self.Y = Y.astype(bool)
        self.should_conv = should_conv
        self.motif_shape = motif_shape
        self.features_mat = np.asarray(features_mat, dtype=theano.config.floatX)
        self.feature_map_shape = [self.motif_shape[0], self.X.shape[-1] - self.motif_shape[-1] + 1]
        self.input_shape = self.X.shape
        self.add_features = add_features
        if self.should_conv:
            self.ConvLayerObj = ConvLayerObj
            self.keep_output = keep_output
            self.RC = RC
            self.combine_RC = combine_RC
            self.RC_combine_mode = RC_combine_mode
            if self.RC_combine_mode == 'max':
                self.RC_max = True
                self.add_RC = False
            elif self.RC_combine_mode == 'add':
                self.RC_max = False
                self.add_RC = True
            self.name = name
            self.debug = debug
            self.Process()

    def Process(self):
        self.X_conv, self.X_full_conv = self.Conv(self.X, self.X.shape, self.ConvLayerObj, keep_output=self.keep_output)
        if self.RC and self.combine_RC:
            self.X_RC_conv, self.X_RC_full_conv = self.Conv(self.X_RC, self.X_RC.shape, self.ConvLayerObj, keep_output=self.keep_output)
            self.X_X_RC_max = np.zeros_like(self.X_conv)
            if self.RC_max:
                for i in range(self.X_conv.shape[0]):
                    for j in range(self.X_conv.shape[1]):
                        if self.add_RC:  #add convolutions from forward and reverse strands
                            self.X_X_RC_max[i,j] = self.X_conv[i,j] + self.X_RC_conv[i,j]
                        else: #else take the max of the two. 
                            self.X_X_RC_max[i,j] = max(self.X_conv[i,j], self.X_RC_conv[i,j])
                if self.add_features:
                    self.X_comb_conv = np.hstack((self.X_X_RC_max, self.features_mat))
                else:
                    self.X_comb_conv = self.X_X_RC_max
            else:
                self.X_X_RC_comb = np.hstack((self.X_conv, self.X_RC_conv)) #merge features into one supervector
                if self.add_features:
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
        elif self.RC and not self.combine_RC:
            self.X_RC_conv, self.X_RC_full_conv = self.Conv(self.X_RC, self.X_RC.shape, self.ConvLayerObj, keep_output=self.keep_output)
            if self.add_features:
                if self.debug:
                    print(self.features_mat.shape)
                    print(self.X_RC_conv.shape)
                    print(self.X_conv.shape)
                #self.X_RC_conv_features = np.hstack((self.X_RC_conv, self.features_mat))
                self.X_conv_features = np.hstack((self.X_conv, self.features_mat))
                self.X_comb_conv = np.vstack((self.X_RC_conv, self.X_conv_features))
            else:
                self.X_comb_conv = np.vstack((self.X_RC_conv, self.X_conv)) 
            Y_copy = np.copy(self.Y)
            self.Y = np.hstack((Y_copy, Y_copy)) #Duplicate my Yo output 
        elif not self.RC:
            if self.add_features:
                self.X_comb_conv = np.hstack((self.X_conv, self.features_mat))
            else:
                self.X_comb_conv = self.X_conv
        self.X_comb_conv_shape = self.X_comb_conv.shape
        self.X_comb_conv_width = self.X_comb_conv_shape[1]
        self.classifier_dict = {}


    def Correct_RC_index(self, index):
        index_copy = np.copy(index)
        index_RC = self.X_conv.shape[0] + index_copy
        return np.hstack((index_copy, index_RC))

    def Conv(self, X, input_shape, ConvLayerObj, minibatch_size=2000, max_axis = 1, keep_output=False):
        minibatches, rem = divmod(input_shape[0], minibatch_size)
        # output = np.zeros((X.shape[0], self.motif_shape[1], 0, X.shape[-1] - self.motif_shape[-1] + 1 ))
        # output = np.zeros((X.shape[0], X.shape[-1] - self.motif_shape[-1] + 1 ))
        output = np.zeros((X.shape[0], self.motif_shape[0]))
        if keep_output:
            output_full_conv = np.zeros((input_shape[0], self.feature_map_shape[-1]))
        for idx in range(minibatches):
            cur_conv_output = ConvLayerObj.conv_func(
                X[idx*minibatch_size:(idx+1)*minibatch_size])  #output is index, filter, _, _
            if keep_output:
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
            if keep_output:
                output_full_conv[minibatch_size*minibatches:] = cur_conv_output
            # print(cur_conv_output.shape)
            for cur_idx in range(cur_conv_output.shape[0]):
                cur_max = np.amax(cur_conv_output[cur_idx, :, 0, :],
                                  axis=max_axis)
                output[minibatches*minibatch_size + cur_idx, :] = cur_max
        if keep_output:
            return output, output_full_conv
        else:
            return output, None


def scaler(X):
    std = StandardScaler()
    std.fit(X)
    return std.transform(X)

def get_weights(Y):
    pos=np.sum(np.asarray(Y))
    neg=len(Y)-pos
    return float(len(Y))/pos, float(len(Y))/neg

class CvEngine(object):
    def __init__(self, binObj, CV_dict, output_dir, name, motif_tensor, motif_names, RC = True, keep_output = False, debug=False):
        self.name = name
        self.CV_dict = CV_dict
        self.motif_tensor = motif_tensor
        self.motif_names = motif_names
        self.dataObj = binObj
        self.output_dir = output_dir
        self.keep_output = keep_output
        self.resultObj = Results(name, output_dir)
        self.RC = RC
        self.debug = debug
        self.kwargs = None

    def start_CV_NN(self, kwargs):
        self.kwargs = kwargs
        if self.debug:
            print(kwargs)
            print('Starting CV NN')
        idx = 0
        start = time.time()
        for chrom in sorted(self.CV_dict.iterkeys()):
            input_dict = self.CV_dict[chrom]
            pos_indices = input_dict['chrom_indices']
            neg_indices = input_dict['other_indices']
            X_train= self.dataObj.merged_tensor[neg_indices]
            X_test = self.dataObj.merged_tensor[pos_indices]
            X_RC_train = self.dataObj.merged_RC_tensor[neg_indices]
            X_RC_test = self.dataObj.merged_RC_tensor[pos_indices]
            F_train = self.dataObj.features_mat[neg_indices]
            F_test = self.dataObj.features_mat[pos_indices]
            Y_train = self.dataObj.merged_tar[neg_indices]
            Y_test = self.dataObj.merged_tar[pos_indices]
            pos_weight, neg_weight = get_weights(Y_train)
            #self.convObj.LogisticRegCVPredict
            DataConvObj = DataConvolver(X_train, X_RC_train, Y_train, self.motif_tensor.shape, features_mat=F_train,should_conv=False)
            KerasModelObj = DoubleKerasModel(DataConvObj, self.motif_tensor)
            KerasModelObj.build_model(kwargs)
            KerasModelObj.train(X_train, X_RC_train, F_train, Y_train, neg_weight, pos_weight, nb_epoch=50, batch_size=250)
            Y_pred = KerasModelObj.predict(X_test, X_RC_test, F_test, batch_size = 150).flatten()
            if np.amax(Y_test) == 0:
                print "Skipping %s because no positive example"
                continue
            # reg_weights = KerasModelObj.model.layers[2].get_weights()[0]
            reg_bias = [0]
            reg_weights = np.random.rand(self.motif_tensor.shape[0])
            reg_weights = pd.Series(data=reg_weights, index=self.motif_names,
                                    name=self.name + '_' + chrom)
            reg_bias = 0
            
            self.resultObj.add_cv_result(Y_pred, Y_test.astype(bool), chrom, reg_weights,
                                         reg_bias, pos_indices)
            idx += 1
            if self.debug:
                print("Completed NN on chromsome %s in %0.04f seconds"
                    %(chrom, time.time() - start))
            if self.debug:
                if idx > 0:
                    break
        self.resultObj.cumulative_result()

    def predOther(self, otherObj, name):
        start = time.time()
        chrom = name
        DataConvObj = DataConvolver(self.dataObj.merged_tensor, self.dataObj.merged_RC_tensor, self.dataObj.merged_tar, self.motif_tensor.shape, should_conv=False, features_mat=self.dataObj.features_mat)
        KerasModelObj = DoubleKerasModel(DataConvObj, self.motif_tensor)
        pos_weight, neg_weight = get_weights(DataConvObj.Y)
        KerasModelObj.build_model(self.kwargs)
        KerasModelObj.train(DataConvObj.X, DataConvObj.X_RC, DataConvObj.features_mat,
                            DataConvObj.Y, neg_weight, pos_weight, nb_epoch=50, batch_size=250)
        reg_bias = [0]
        reg_weights = np.random.rand(self.motif_tensor.shape[0])
        # reg_weights = KerasModelObj.model.layers[2].get_weights()[0]
        reg_weights = pd.Series(data=reg_weights, index=self.motif_names,
                                name=self.name + '_' + chrom)
        Y_pred = KerasModelObj.predict(otherObj.merged_tensor, otherObj.merged_RC_tensor, otherObj.features_mat, batch_size = 150).flatten()
        self.resultObj.add_cv_result(Y_pred, otherObj.merged_tar.astype(bool), chrom, reg_weights, reg_bias, indices=np.ones((self.dataObj.merged_tensor.shape[0],)))
        if self.debug:
            print "Completed predicting on %s in %0.02f seconds"%(name, time.time()-start)
        cum_resultObj = ClassificationResult(otherObj.merged_tar.astype(bool).flatten(),
                                                  Y_pred.flatten(),
                                                  name=self.name+ '_' + name)
        if self.debug:
            print "Printing result obj"
            print str(cum_resultObj)
            print cum_resultObj.results
        return cum_resultObj.results[2][1]

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

    def return_combined_auroc(self):
        return self.resultObj.cum_result[2][1]

    def wipe_results(self):
        self.resultObj = Results(self.name, self.output_dir)

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
        self.ensure_dir(output_dir)
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
        print(self.str_cum_result)

    def summarize(self, output_file_name='', prefix=''):
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
