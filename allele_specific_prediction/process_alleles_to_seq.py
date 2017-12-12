from pyfaidx import Fasta
import sys
import h5py
import gzip
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import numpy as np
import rpy2.robjects as robjects

append = "set9"
#read in coding regions
f_loc = "../data/HiDRA/"
ref_file = open(f_loc + "refseq_coding_genes_hg19.txt")
alt_f_name = "common_snps_alt" + append + ".txt.gz"
ref_f_name = "common_snps_ref" + append + ".txt.gz"


#process out exons first
ref_lines = ref_file.readlines()

coding_regions = {} #keys = chrms, values = set of exons
#set up dictionary
for i in range(23):
    coding_regions["chr" + str(i)] = set()
#going through lines of reference file
for l in ref_lines:
    chrm = l.split("\t")[2]
    exon_starts = l.split("\t")[9].split(",")[:-1] #going to have an extra at the end
    exon_ends = l.split("\t")[10].split(",")[:-1]
    try:
        for x in (exon_starts + exon_ends):
            coding_regions[chrm].add(x)
    except KeyError:
        chrm = l.split("\t")[2].split("_")[0]
        try:
            for x in (exon_starts + exon_ends):
                coding_regions[chrm].add(x)
        except KeyError:
            if chrm != "chrY" and chrm != "chrX" and chrm != "chrUn":
                print "KeyError: loc, chrm ", x, chrm

ref_file.close()


hg19 = Fasta("../data/hg19/genes.fa")
#hg19 = Fasta("../data/hg19/chr1.fa") #for testing
#returns a sequence of length length centered at loc with snp as the SNP
def get_sequence(chrm, loc, length, ref, alt):
    s_index = loc - length/2
    e_index = loc + length/2
    if contains_exon(chrm, s_index, e_index):
        return None
    
    first_half = hg19[chrm][s_index:loc-1].seq.upper()
    second_half = hg19[chrm][loc:e_index].seq.upper()
    return first_half + ref.upper() + second_half, first_half + alt.upper() + second_half


def contains_exon(chrm, start, end): #returns True if the region between start and end contain an exon
    for x in range(start, end+1):
        if x in coding_regions[chrm]:
            return True
    return False

def one_hot_encode(s):
    one_hot_mat = []
    order = "ACGT"
                
    #convert to ints
    try:
        char_to_int = dict((b, i) for i, b in enumerate(order))
        integer_encoded = [char_to_int[b] for b in s]
        #convert to matrix
        encoded = to_categorical(integer_encoded)
        #set print options so numpy converts large 2-D arrays to string format with ellipsis
        np.set_printoptions(threshold = np.prod(encoded.T.shape))
        return np.array2string(encoded.T, max_line_width = 2000)
    except KeyError:
        print ("ACGT not found")
        return None



#go through clinvar file
print ("writing into " + alt_f_name + " and " + ref_f_name)
alt_f = gzip.GzipFile(alt_f_name, 'wb') #file that gets written to, placed in current directory
ref_f = gzip.GzipFile(ref_f_name, 'wb') #file that gets written to, placed in current directory
alt_f.write(str([0]*100)) #dummy Y data
ref_f.write(str([0]*100)) #dummy Y data

count = 0
with open(sys.argv[1]) as allele_file:
    for l in allele_file:
        count += 1
        if count%1000 == 0:
            print "lines processed ", count
        if l[0] != "#":
            try:
                line = l.split("\t")
                chrm = "chr" + str(line[0])
                pos = int(line[1])
                if len(line[3]) > 1 or len(line[4]) > 1:
                    continue
                else:
                    ref, alt = get_sequence(chrm, pos, 500, line[3], line[4])
                    if np.array(ref).shape != np.array(alt).shape:
                        print "difference between ref and alt!!", ref, alt
                    else:
                        #one-hot encoding
                        mata = one_hot_encode(alt)
                        matr = one_hot_encode(ref)
                        if mata != None and matr != None:
                            alt_f.write("/")
                            alt_f.write(mata)
                            ref_f.write("/")
                            ref_f.write(matr)

            except KeyError:
                pass
            except Exception as e:
                print (e)
                
                                    
alt_f.close()
ref_f.close()
            
