import warnings
warnings.filterwarnings('ignore')
import numpy as np
import h5py

def save_array(data, file):
	h5f = h5py.File(file, "w")
	h5f.create_dataset('data', data=data)
	h5f.close()

def load_array(file):
	h5f = h5py.File(file, "r")
	data = h5f['data'][:]
	h5f.close()
	return data
