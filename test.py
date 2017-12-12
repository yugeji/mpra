import numpy as np
import h5py

f = h5py.File('test.hdf5', 'w')
group = f.create_group("somegroup")

a = np.arange(100)
dset = group.create_dataset("data", data=a)

print group["data"][:]
