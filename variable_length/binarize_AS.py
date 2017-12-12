import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import normalize, StandardScaler

def motif_normalize(filename):
    f = open(filename, 'r')
    new_file = open("normalized_output0.txt", 'wb')

    vector = []
    for line in f:
        vector.append(float(line.split("\t")[1]))

    print len(vector)

    scaler = StandardScaler()
    scaler.fit(np.log(vector))
    normed = scaler.transform(np.log(vector))
    #normed = vector #temp
        
#    print "vc1", vector[0], "mean: ", np.mean(vector), "std: ", np.std(vector)
#    vector = vector - np.mean(vector)
#    print "shifted mean:", np.mean(vector)
#    normed = stats.norm.pdf(vector, np.mean(vector), np.std(vector)) #(vector-np.mean(vector))/(np.std(vector))
#    print "std after pdf:", np.std(normed), "mean:", np.mean(normed)
#    print normed[0], np.mean(normed), np.std(vector)
#    normed = normed/np.std(vector)

#    normed = np.log(vector)
#    normed = normed - np.mean(normed)
    std = np.std(normed)*0
    mean = np.mean(normed)

    print normed[0], mean, std

    f.close()
    f = open(filename, 'r')
    lines = f.readlines()

    above_count = 0
    below_count = 0
    for i in range(len(lines)):
        if abs(normed[i]) > std:
            if normed[i] > std:
                new_file.write(lines[i].split("\t")[0] + "\t1.0" + "\n")
                above_count += 1
            else:
                new_file.write(lines[i].split("\t")[0] + "\t-1.0" + "\n")
                below_count += 1

    print above_count, below_count

    
    
motif_normalize("../data/ATAC_STARRdata/output.txt")
