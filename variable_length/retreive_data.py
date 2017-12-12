import gzip
import numpy as np

#simple zero-padding function
def zero_padding(input):
        max_num = 500

        l = len(input[0])
        output = np.zeros((4, max_num))
        size = min(l, max_num)
        output[:4, (max_num-size)/2:(max_num+size)/2] = input[:4, (l-size)/2:(l+size)/2] #zero pads the two ends and truncates if too long
        return output

def no_change(input):
        return input

#unpacking gzip into non-ragged format
def unpack_gzip(filename, func = zero_padding, ragged = False):
        mat = []
        y_data = []
        with gzip.open(filename, 'rb') as f:
                seq = ""
                for line in f:
                        seq += line
                        if "/" in line: #end of a sequence
                                seq = seq.replace("[", '').replace("]", '').replace("\n", '').split("/") #should figure out a better way to do this
                                if len(y_data) == 0: #y_data is the first line
                                        y_data = np.fromstring(seq[0].replace("'", ''), dtype=float, sep=",")
                                else:
                                        arr = np.fromstring(seq[0], sep='.', dtype=int)
                                        input = arr.reshape(4, -1)
                                        mat.append(func(input))
                                seq = seq[1] #we're done with 1 seq, maybe we have the tailend of the next though
                #at the very end of the file, there is no "/"
                seq = seq.replace("[", '').replace("]", '').replace("\n", '').split("/") #should figure out a better way to do this
                arr = np.fromstring(seq[0], sep='.', dtype=int)
                input = arr.reshape(4, -1)
                mat.append(func(input))
                                
                        
                                

        
#        input_file = gzip.open(filename, 'rb')
#        try:
#                seqs = input_file.read()
#        finally:
#                input_file.close()
#
#        mat = []
        
#        seqs = str(seqs).replace("[", '').replace("]", '').replace("\n", '').split("/") #should figure out a better way to do this
#        y_data = np.fromstring(seqs[0].replace("'", ''), dtype=float, sep=",")
        

#        for i in range(1, len(seqs)):
#                arr = np.fromstring(seqs[i], sep='.', dtype=int)
#                input = arr.reshape(4, -1)
#                mat.append(func(input))

        if ragged:
                return mat, y_data
        else:
                return np.asarray(mat), np.asarray(y_data)

def unpack_gzip_generator(filename, func = zero_padding, ragged = False):
#        mat = []
        y_data = []
        with gzip.open(filename, 'rb') as f:
                seq = ""
                for i in range(len(f)):
                        line = f[i]
                        seq += line
                        if "/" in line: #end of a sequence
                                seq = seq.replace("[", '').replace("]", '').replace("\n", '').split("/") #should figure out a better way to do this
                                if len(y_data) == 0: #y_data is the first line
                                        y_data = np.fromstring(seq[0].replace("'", ''), dtype=float, sep=",")
                                else:
                                        arr = np.fromstring(seq[0], sep='.', dtype=int)
                                        input = arr.reshape(4, -1)
                                        yield func(input), y_data[i]
                                        seq = seq[1] #we're done with 1 seq, maybe we have the tailend of the next though
                #at the very end of the file, there is no "/"
                seq = seq.replace("[", '').replace("]", '').replace("\n", '').split("/") #should figure out a better way to do this
                arr = np.fromstring(seq[0], sep='.', dtype=int)
                input = arr.reshape(4, -1)
                yield func(input), y_data[-1]
                                        
        
#print unpack_gzip("chr0.txt.gz")[0].shape
