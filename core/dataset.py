import os, sys
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import torch.utils.data as thd
from torchvision import transforms
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import core.mel_features as mf
import core.spectrum as spectrum
import core.utils as utils
import multiprocessing as mp
import itertools
from copy import deepcopy
import cv2

DATA_PATH = os.path.join(os.getcwd(), 'data')

data_transforms = {
	'train': transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize([-2.07057395], [2.12761937])
	]),
	'test': transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize([-2.07057395], [2.12761937])
	])
}


def cache_spectrogram(filename: str):
	pcm = mf.read_wav(os.path.join('data', 'train', filename))
	spec = spectrum.mel(pcm)
	name, file_extension = os.path.splitext(filename)
	utils.save_array(spec, os.path.join('data', 'cache', name + '.h5'))


def load_and_slice(entry: dict):
	spec = utils.load_array(entry['fpath'])
	sample = []
	for sample_slice in spectrum.sliding_window_split(spec):
		slice_entry = deepcopy(entry)
		pad = sample_slice.shape[0] - sample_slice.shape[1]
		sample_slice = cv2.copyMakeBorder(sample_slice, 0, 0, pad, 0, cv2.BORDER_WRAP)
		slice_entry['data'] = np.expand_dims(sample_slice, axis=-1)
		sample.append(slice_entry)
	return sample


class SoundData(object):
	def __init__(self, test_size=0.2, num_processes=8, seed=42):
		self.df = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
		self.cache_dir = os.path.join(DATA_PATH, 'cache')
		if not os.path.exists(self.cache_dir):
			os.mkdir(self.cache_dir)
		self.num_processes = num_processes
		self.unique_label = np.sort(self.df.label.unique()).tolist()
		self.label2idx = dict(zip(self.unique_label, range(len(self.unique_label))))
		self.df.loc[:, 'target'] = self.df.label.apply(lambda x: self.label2idx[x])
		self.df.loc[:, 'fpath'] = self.df.fname.apply(lambda x: os.path.join(DATA_PATH, 'train', x))
		self.df.loc[:, 'h5f'] = self.df.fname.apply(lambda x: os.path.join(self.cache_dir, os.path.splitext(x)[0] + '.h5'))
		self.num_classes = len(self.unique_label)
		self.cache_samples()
		self.train_idx, self.test_idx = train_test_split(self.df.index.values, test_size=test_size, random_state=seed)

	def cache_samples(self):
		if not os.listdir(self.cache_dir):
			pool = mp.Pool(processes=self.num_processes)
			pool.map(cache_spectrogram, self.df.fname.tolist())
			print("DATASET cached")

	def reset_index(self, train, test):
		self.train_idx, self.test_idx = train, test

	def get_train_test_split(self):
		return self.df.loc[self.train_idx], self.df.loc[self.test_idx]


class Dset(thd.Dataset):
	def __init__(self, df, num_processes=8, transform=None, phase=""):
		"""
		Args:
			:transform:     PyTorch transforms
			:phase:         Description string (optional)
		"""
		self.df = df
		self.data_dir = os.path.join(DATA_PATH, 'train')
		self.samples = []
		pool = mp.Pool(processes=num_processes)

		for i in tqdm(self.df.index, desc=f'Loading {phase} samples'):
			fpath = os.path.join(self.data_dir, self.df.loc[i, 'fname'])
			if os.path.exists(fpath):
				entry = {}
				entry['fpath'] = self.df.loc[i, 'h5f']
				entry['target'] = self.df.loc[i, 'target']
				entry['index'] = i
				self.samples.append(entry)

		self.data = pool.map(load_and_slice, self.samples)
		self.data = list(itertools.chain.from_iterable(self.data))

		self.transform = transform

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		spec = self.data[idx]['data']

		if self.transform is not None:
			spec = self.transform(spec)

		return spec, self.data[idx]['target'], self.data[idx]['index']


if __name__ == '__main__':
	from time import time
	t0 = time()
	sound_data = SoundData(num_processes=6)
	train_df, test_df = sound_data.get_train_test_split()
	trainset = Dset(train_df, num_processes=6, transform=data_transforms['train'])
	testset = Dset(test_df, num_processes=6, transform=data_transforms['test'])
	print(time() - t0)
