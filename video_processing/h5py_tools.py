import h5py
import numpy as np


def save_matrix(matrix, name):
    hf = h5py.File('%s.h5' % name, 'w')
    hf.create_dataset('dataset', data=matrix, compression="gzip", compression_opts=9)
    hf.close()


def read_matrix(name):
    hf = h5py.File('%s.h5' % name, 'r')
    dataset = hf.get('dataset')
    matrix = np.array(dataset)
    hf.close()
    return matrix