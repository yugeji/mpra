import os
import re
import sys
import numpy as np
import time
import keras
from keras.models import model_from_json, load_model
import gzip
import theano
theano.gof.compilelock.set_lock_status(False)
#LCL uses K562 as early stop
#HepG2 uses LCL/K562
#
def load_network(name):
    json_file = open("%s_model.json"%(name),'r')
    weights_file = open("%s_weights.h5"%(name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights_file)
    return loaded_model

def load_h5(name):
    h5_file = "%s.h5" % (name)
    model = load_model(h5_file, custom_objects={'MotifLayer':MotifLayer,
                                            'MotifConnectionsLayer':MotifConnectionsLayer})

    return model

# def load(name):
#     return load_model(name, custom_objects={'MotifLayer':MotifLayer,
#                                             'MotifConnectionsLayer':MotifConnectionsLayer})

def load(name):
    return load_model(name)

class BatchObj(object):
    def __init__(self, directory = "/data1/mpra/seqs/"):
        self.batch_num = 0
        self.directory = directory

    def process_batch(self, batch_number):
        ref_file = self.directory + "batch%i_ref.txt.gz"%(batch_number)
        alt_file = self.directory + "batch%i_alt.txt.gz"%(batch_number)
        ref_tensor, alt_tensor = self.prep_dna(ref_file, alt_file)
        return ref_tensor, alt_tensor

    def count_lines(self, in_file):
        counter = 0
        with gzip.open(in_file, 'r') as f:
            for line in f:
                counter += 1
        return counter

    def prep_dna(self, ref, alt):
        N = self.count_lines(ref)
        idx = 0
        ref_tensor = np.zeros((N,1,4,300))
        with gzip.open(ref,'r') as f:
            for line in f:
                split_line = line.rstrip().split()
                _N, _W = divmod(idx, 4)
                ref_tensor[_N, 0, _W, :] = np.asarray(split_line, dtype=int)
                idx+=1
        idx2 = 0
        alt_tensor = np.zeros((N,1,4,300))
        with gzip.open(alt,'r') as f:
            for line in f:
                split_line = line.rstrip().split()
                _N, _W = divmod(idx, 4)
                alt_tensor[_N, 0, _W, :] = np.asarray(split_line, dtype=int)
                idx2+=1
        assert idx == idx2, "Number of lines in ref and alt must equal"
        return ref_tensor, alt_tensor

    def generate_RC_tensor(self, input_tensor): #input is regular tensors
        output_tensor = np.zeros_like(input_tensor)
        for input_idx in range(input_tensor.shape[0]):
            cur_mat = input_tensor[input_idx,0,:,:]
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
            output_tensor[input_idx,0,:,:] = copy_mat
        return output_tensor

    def generate_current_batch(self, current_index, ref_tensor, alt_tensor):
        batch_size = ref_tensor.shape[-1] - 150 + 1
        r = np.zeros((batch_size,1,4,150), dtype=np.float32)
        a = np.zeros((batch_size,1,4,150), dtype=np.float32)
        for idx in range(batch_size):
            start = idx
            end = idx + 150
            r[idx] = ref_tensor[current_index, :, :, start:end]
            a[idx] = alt_tensor[current_index, :, :, start:end]
        return r, a, self.generate_RC_tensor(r), self.generate_RC_tensor(a) 


class PredictorObj(object):
    def __init__(self, model_name):
        self.model = load(model_name)
        #self.model = load_network(model_name)
        self.results = []
        self.results_per_batch = []

    def predict(self, r_batch, a_batch, r_rc_batch, a_rc_batch):
        r_pred = self.model.predict([r_batch, r_rc_batch], verbose = 0)
        a_pred = self.model.predict([a_batch, a_rc_batch], verbose = 0)
        r_sum = np.sum(r_pred.flatten())
        a_sum = np.sum(a_pred.flatten())
        diff = r_sum - a_sum
        return diff

    def clean(self):
        self.results = []
        self.results_per_batch = []

class Engine(object): 
    def __init__(self, filedir="./seqs/", debug=False):
        #filedir = /broad/compbio/liyue/Projects/mpra/data/comVarSeq/
        self.filedir = filedir
        self.BatchObj = BatchObj(self.filedir)
        self.debug=debug
        self.models = ["./models/K562_Simple.h5", "./models/HepG2_Simple.h5", "./models/LCL_Simple.h5"]
        self.predObjs, self.model_names = self.process_models()
        print len(self.predObjs)

    def count_lines(self, in_file):
        counter = 0
        with gzip.open(in_file, 'r') as f:
            for line in f:
                counter += 1
        return counter

    def process_models(self):
        predObjs = []
        model_names = []
        for idx, model_name in enumerate(self.models):
            predObj = PredictorObj(model_name)
            predObjs.append(predObj)
            p = re.search("/models/(.*).h5", model_name)
            if p:
                model_name = p.group(1)
                model_names.append(model_name)
        return predObjs, model_names

    def all(self, output_file = ""):
        pass

    def one(self, batch_num, output_dir = "./comVar_test_results/", debug=False):
        start=time.time()
        output_name = output_dir + "%i_results.txt.gz"%(batch_num)
        ref_tensor, alt_tensor = self.BatchObj.process_batch(batch_num)
        mat_x = ref_tensor.shape[0]
        mat_y = len(self.predObjs)
        results = np.zeros((mat_x, mat_y))
        with gzip.open(output_name, 'wb') as f:
            for i in range(ref_tensor.shape[0]):
                r_t, r_rc_t, a_t, a_rc_t = self.BatchObj.generate_current_batch(i, ref_tensor, alt_tensor)
                for j, predObj in enumerate(self.predObjs):
                    results[i,j] = predObj.predict(r_t, a_t, r_rc_t, a_rc_t)
                    if debug:
                        print results[i,j]
                if i % 50 == 0 and i > 0:
                    print("Index %i done in %0.03f seconds"%(i, time.time()-start))
                    if debug:
                        break 
            joint_names = "\t".join(self.model_names)
            f.write("%s\n"%(joint_names))
            for i in range(results.shape[0]):
                joint_str = "\t".join(map(str,results[i]))
                f.write("%s\n"%(joint_str))
        
def main():
    #total = range(1,2489)
    start = time.time()
    batch_num = int(sys.argv[1])
    eng = Engine()
    eng.one(batch_num, debug=False)
    print("Completed in %0.04f"%(time.time()-start))



if __name__ == "__main__":
    main()
