import rpy2.robjects as robjects
import numpy as np


mat = robjects.r['load']("../data/ATAC_STARRdata/SHARPR/ForAlvinATACSTARR_GM12878Lib_150_600_merged_0.2RPM_RNAtoDNA_FC_result_sa333_autoF.RData")

a = np.array(robjects.r['whole_re']) #this takes forever.

print len(a) #22
print a[0] #7
print len(a[1]) #7




