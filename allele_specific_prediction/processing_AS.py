from pyfaidx import Fasta
import sys
import h5py
import gzip
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import numpy as np
import rpy2.robjects as robjects

DEBUG = False
#DEBUG = True

atac_dir="../data/ATAC_STARRdata/"
hg19_dir="../data/hg19/"
read_file=sys.argv[1] #"normalized_output0.txt" #format: location, score, length of region
length_index=5 #which index the length measurement is
if DEBUG:
    read_file = "test_file2.txt"
print ("reading sequences from " + read_file)
        
#available chromosomes
indices = range(1, 23)
indices = [str(i) for i in indices]
#indices.append("X")
#indices.append("Y")

#separate file out by chromosome
chrs = {k:[] for k in indices} #initialize dictionary of sequences by chrm

#sort reads by chromosome
with open(read_file) as f:
    next(f)
    for line in f:
        vals = line.split("\t")
        chrm = vals[0].split(":")[0][3:]
        try:
            chrs[chrm].append(line)
        except KeyError:
            pass

if DEBUG:
    indices = ["1"]
for i in indices:
    chr_name = "chr" + i
    chr_f = Fasta(hg19_dir + chr_name + ".fa")
    write_f_name = 'chr' + i + 'alt_binary.txt.gz'
    if DEBUG:
        write_f_name = "chr0.txt.gz"
    write_f = gzip.GzipFile(write_f_name, 'wb') #file that gets written to, placed in current directory

    print ("working on chromosome " + i)
    print ("writing into " + write_f_name)

    dataset = []
    values = []
    for line in chrs[i]:
        #gathering values
        vals = line.strip().split("\t")
        values.append(vals[1])
        #retrieving sequences from Fasta
        loc = vals[0].split(":")
        s_index = int(float(loc[1].split("-")[0]))-1
        e_index = int(float(loc[1].split("-")[1]))
        dataset.append(str(chr_f[chr_name][s_index:e_index]).upper())

    write_f.write(str(values)) #Y data

    #one-hot encoding
    one_hot_mat = []
    order = "ACGT"
    try:
        for idx in range(len(dataset)):
            write_f.write("/")

            s = dataset[idx]
            #convert to ints
            try:
                char_to_int = dict((b, i) for i, b in enumerate(order))
                integer_encoded = [char_to_int[b] for b in s]
                #convert to matrix
                encoded = to_categorical(integer_encoded)
                #set print options so numpy converts large 2-D arrays to string format with ellipsis
                np.set_printoptions(threshold = np.prod(encoded.T.shape))
                #        write_f.write(",".join(" ".join(str(i) for i in encoded.T)))
                write_f.write(np.array2string(encoded.T, max_line_width = 2000))
            except KeyError:
                print ("ACGT not found")

    finally:
        write_f.close()

    
#setting numpy string printing options back
np.set_printoptions(threshold = 1000)


