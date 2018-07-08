import multiprocessing as mp

import core.mel_features as mf
import numpy as np
import os

import pickle as pkl


def mel(data):
	return mf.mel_spectrogram(data=data, sample_rate=16000, log_offset=0.01, window_length_secs=0.1, hop_length_secs=0.01,
							  logarithmic=True, num_mel_bins=256, lower_edge_hertz=100, upper_edge_hertz=10000).T


def spectrogram(data):
	return mf.spectrogram(data=data, window_length_secs=0.1, hop_length_secs=0.01, sample_rate=16000, logarithmic=True).T


def continuous_wavelet_transform(data):
	cwt = mf.continuous_wavelet_transform(data, octave_exponent=10, sub_octaves=25, starting_scale=2, dt=1)
	return np.flipud(cwt)


def sliding_window_split(data, overlap=20, split_width=256):
	sample_width = data.shape[-1]
	index = 0
	splits = []

	while index + split_width < sample_width:
		splits.append(data[..., index:index + split_width])
		index += split_width - overlap

	if sample_width - index > split_width / 8:
		slice_from = -split_width if sample_width > split_width else 0
		splits.append(data[..., slice_from:])

	return np.array(splits)
