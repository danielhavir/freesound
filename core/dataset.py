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
	'mel256_train': transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize([-2.07057395], [2.12761937])
	]),
	'mel256_test': transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize([-2.07057395], [2.12761937])
	]),
	'wavelet_train': transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize([-7.5517883], [7.686689])
	]),
	'wavelet_test': transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize([-7.5517883], [7.686689])
	]),
	'44mel256_train': transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize([-2.44529629], [1.96563387])
	]),
	'44mel256_test': transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize([-2.44529629], [1.96563387])
	]),
	'24mel256_train': transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize([-2.1824522], [2.08129025])
	]),
	'24mel256_test': transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize([-2.1824522], [2.08129025])
	]),
}


def cache_spectrogram(filename: str):
	pcm = mf.read_wav(os.path.join('data', 'train', filename), target_sample_rate=24000)
	spec = spectrum.mel(pcm)
	name, file_extension = os.path.splitext(filename)
	utils.save_array(spec, os.path.join('data', 'cache', '24mel256_train', name + '.h5'))


def cache_test_spectrogram(filename: str):
	pcm = mf.read_wav(os.path.join('data', 'test', filename), target_sample_rate=24000)
	spec = spectrum.mel(pcm)
	name, file_extension = os.path.splitext(filename)
	utils.save_array(spec, os.path.join('data', 'cache', '24mel256_test', name + '.h5'))


def load_and_slice(entry: dict):
	spec = utils.load_array(entry['fpath'])
	sample = []
	for sample_slice in spectrum.sliding_window_split(spec, split_width=spec.shape[0]):
		slice_entry = deepcopy(entry)
		pad = sample_slice.shape[0] - sample_slice.shape[1]
		sample_slice = cv2.copyMakeBorder(sample_slice, 0, 0, pad, 0, cv2.BORDER_WRAP)
		slice_entry['data'] = np.float32(np.expand_dims(sample_slice, axis=-1))
		sample.append(slice_entry)
	return sample


def load_and_slice_test(entry: dict):
	spec = utils.load_array(entry['fpath'])
	sample = []
	if spec.shape[1] < spec.shape[0]:
		pad = spec.shape[0] - spec.shape[1]
		sample_slice = cv2.copyMakeBorder(spec, 0, 0, pad, 0, cv2.BORDER_WRAP)
		entry['data'] = np.float32(np.expand_dims(sample_slice, axis=-1))
		return [entry]
	for sample_slice in spectrum.sliding_window_split(spec, split_width=spec.shape[0]):
		slice_entry = deepcopy(entry)
		pad = sample_slice.shape[0] - sample_slice.shape[1]
		sample_slice = cv2.copyMakeBorder(sample_slice, 0, 0, pad, 0, cv2.BORDER_WRAP)
		slice_entry['data'] = np.float32(np.expand_dims(sample_slice, axis=-1))
		sample.append(slice_entry)
	return sample


class SoundData(object):
	def __init__(self, cache_prefix='mel256', test_size=0.2, num_processes=8, seed=42, prevent_cache=False):
		self.df = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
		self.cache_dir = os.path.join(DATA_PATH, 'cache', f'{cache_prefix}_train')
		if not os.path.exists(self.cache_dir):
			os.mkdir(self.cache_dir)
		self.num_processes = num_processes
		self.prevent_cache = prevent_cache
		self.unique_label = np.sort(self.df.label.unique()).tolist()
		self.label2idx = dict(zip(self.unique_label, range(len(self.unique_label))))
		self.idx2label = dict(zip(range(len(self.unique_label)), self.unique_label))
		self.df.loc[:, 'target'] = self.df.label.apply(lambda x: self.label2idx[x])
		self.df.loc[:, 'fpath'] = self.df.fname.apply(lambda x: os.path.join(DATA_PATH, 'train', x))
		self.df.loc[:, 'h5f'] = self.df.fname.apply(lambda x: os.path.join(self.cache_dir, os.path.splitext(x)[0] + '.h5'))
		self.num_classes = len(self.unique_label)
		self.cache_samples()
		self.idxs = self.df.index.values
		self.train_idx, self.test_idx = train_test_split(self.idxs, test_size=test_size, random_state=seed)

	def cache_samples(self):
		if not os.listdir(self.cache_dir) and not self.prevent_cache:
			print(f"Caching in {self.num_processes} processes...")
			pool = mp.Pool(processes=self.num_processes)
			pool.map(cache_spectrogram, (self.df.fname).tolist())
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


class TestDset(thd.Dataset):
	def __init__(self, cache_prefix='mel256', num_processes=3, transform=None):
		self.data_dir = os.path.join(DATA_PATH, 'test')
		self.cache_dir = os.path.join(DATA_PATH, 'cache', f'{cache_prefix}_test')
		if not os.path.exists(self.cache_dir):
			os.mkdir(self.cache_dir)
		self.h5fs = sorted(os.listdir(self.cache_dir))
		self.idx2fname = dict(zip(range(len(self.h5fs)), [os.path.splitext(h5f)[0] + '.wav' for h5f in self.h5fs]))
		
		pool = mp.Pool(processes=num_processes)

		if not os.listdir(self.cache_dir):
			pool.map(cache_test_spectrogram, os.listdir(self.data_dir))
		
		self.samples = []
		for i, fname in enumerate(tqdm(self.h5fs, desc=f'Loading test spectrograms')):
			entry = {}
			entry['fname'] = os.path.splitext(fname)[0]
			entry['index'] = i
			entry['fpath'] = os.path.join(self.cache_dir, fname)
			self.samples.append(entry)
		
		self.data = pool.map(load_and_slice_test, self.samples)
		self.data = list(itertools.chain.from_iterable(self.data))

		self.transform = transform
	
	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, idx):
		spec = self.data[idx]['data']

		if self.transform is not None:
			spec = self.transform(spec)

		return spec, self.data[idx]['index']


if __name__ == '__main__':
	from time import time
	t0 = time()
	#sound_data = SoundData(cache_prefix='24mel256', num_processes=6)
	#train_df, test_df = sound_data.get_train_test_split()
	#trainset = Dset(sound_data.df, num_processes=6, transform=data_transforms['24mel256_train'])
	#valset = Dset(test_df, num_processes=6, transform=data_transforms['test'])
	testset = TestDset(cache_prefix='24mel256', num_processes=4, transform=data_transforms['44mel256_test'])
	print(time() - t0)
	print(len(os.listdir(testset.cache_dir)))
	print(testset[0][0].size())
