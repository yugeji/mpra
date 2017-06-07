import os
import sys 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import keras
from hyperopt import hp, fmin, tpe, Trials, STATUS_FAIL, STATUS_OK
from hyperopt.mongoexp import MongoTrials
import motif_processor
import common_utils
import cv_engine
from math import log
import pickle

HepG2MotifProcObj = motif_processor.MotifProcessor(ENCODE_only=False, 
                                                   cell_line="HepG2", 
                                                   meme_file="/home/alvin/Dropbox/Lab/CNN/data/meme/HepG2_pos_steme/steme.txt")
HepG2_motif_tensor, HepG2_motif_names = HepG2MotifProcObj.generate_custom_CNN_filters(max_length = 18, padding = 0, truncate = False)

K562MotifProcObj = motif_processor.MotifProcessor(ENCODE_only=False, 
                                                   cell_line="K562", 
                                                   meme_file="/home/alvin/Dropbox/Lab/CNN/data/meme/K562_pos_steme/steme.txt")
K562_motif_tensor, K562_motif_names = K562MotifProcObj.generate_custom_CNN_filters(max_length = 18, padding = 0, truncate = False)


LCLMotifProcObj = motif_processor.MotifProcessor(ENCODE_only=False, 
                                                   cell_line="LCL", 
                                                   meme_file="/home/alvin/Dropbox/Lab/CNN/data/meme/LCL_pos_steme/steme.txt")
LCL_motif_tensor, LCL_motif_names = LCLMotifProcObj.generate_custom_CNN_filters(max_length = 18, padding = 0, truncate = False)


CrossValProcObj = common_utils.CrossValProcessor(output_dirs = ["./HEPG2_act_1_25/", "./HEPG2_rep_1_25/", 
                                                                      "./K562_act_1_25/", "./K562_act_1_25/",
                                                                      "./LCL_act_alt_1_25/"])

HepG2_binObj = CrossValProcObj.HepG2_binObj
K562_binObj = CrossValProcObj.K562_binObj
LCL_binObj = CrossValProcObj.LCL_binObj
binObjs = [HepG2_binObj, K562_binObj, LCL_binObj]
CvDicts = [CrossValProcObj.HepG2_dict, CrossValProcObj.K562_dict, CrossValProcObj.LCL_dict]
motif_tensors = [HepG2_motif_tensor, K562_motif_tensor, LCL_motif_tensor]
motif_names = [HepG2_motif_names, K562_motif_names, LCL_motif_names]
names=["HepG2", "K562", "LCL"]
dirs = ["./HepG2_1_25/", "./K562_1_25/", "./LCL_1_25/"]
#Testing

idx = int(sys.argv[1])
if idx == 0:
    other_indices = [1,2]
elif idx == 1:
    other_indices = [0,2]
else:
    other_indices = [0,1]

cvObj = cv_engine.CvEngine(binObjs[idx], CvDicts[idx], dirs[idx], names[idx], motif_tensors[idx], motif_names[idx])
def func(kwargs):
    try:
        print('starting func')
        print kwargs
        cvObj = cv_engine.CvEngine(binObjs[idx], CvDicts[idx], dirs[idx], names[idx], motif_tensors[idx], 
                           motif_names[idx], debug=True)        
        cvObj.start_CV_NN(kwargs)
        other1 = cvObj.predOther(binObjs[other_indices[0]], name = names[other_indices[0]])
        other2 = cvObj.predOther(binObjs[other_indices[1]], name = names[other_indices[1]])
        other1_name = names[other_indices[0]]
        other2_name = names[other_indices[1]]
        cvObj.summarize(prefix="%0.05f_%0.05f_%0.05f"%(float(kwargs['dropout']),float(kwargs['L1']),float(kwargs['L2'])))
        print cvObj.return_combined_auroc()
        return_dict ={'loss': -1 * cvObj.return_combined_auroc(), 'status': STATUS_OK, 'attachments':{other1_name:other1,other2_name:other2}}
        #return_dict ={'loss': -1 * cvObj.return_combined_auroc(), 'status': STATUS_OK, 'attachments':{other1_name:other1}}
        cvObj.wipe_results()
        print "I'm returning!"
        return return_dict
    except Exception as e:
        print "Exception"
        print e
        return {'status': STATUS_FAIL}

spaces = (
    hp.uniform('dropout', 0.0, 0.25),
    hp.loguniform('L1', log(1e-7), log(10)),
    hp.loguniform('L2', log(1e-7), log(10)),
    hp.loguniform('L1_W1', log(1e-7), log(10)),
    hp.loguniform('L2_W2', log(1e-7), log(10)),
    hp.quniform('pool_width', 2,25,1)
)

#trials=MongoTrials('mongo://localhost:1234/motif_db2/jobs', exp_key='test_exp1')
trials=Trials()
best = fmin(func, space=spaces,algo=tpe.suggest,max_evals=2, trials=trials)
pickle.dump(trials, open('./hyperopt_%i/%0.04f_%0.04f_%0.04f.pickle'%(idx, best['dropout'], best['L1'], best['L2']), 'wb'))
print best
print trials.best_trial
