import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as thd
import core.dataset as cd
from tqdm import tqdm
from collections import defaultdict
import argparse

CKPT_DIR = os.path.join('checkpoints')

# Collect arguments (if any)
parser = argparse.ArgumentParser()

# Cache prefix
parser.add_argument('cache_prefix', nargs='?', type=str, choices=['mel256', 'wavelet'], default='mel256', help="Mel spectrogram or wavelets.")
# Checkpoint directory
parser.add_argument('-dir', '--ckpt_dir', type=str, choices=os.listdir(CKPT_DIR), default=sorted(os.listdir(CKPT_DIR))[-1], help="Checkpoints dir.")
# Checkpoint directory
parser.add_argument('-dir2', '--ckpt_dir2', type=str, choices=os.listdir(CKPT_DIR), default=sorted(os.listdir(CKPT_DIR))[-2], help="Checkpoints dir.")
# Type of evaluation
parser.add_argument('-t', '--type', type=str, choices=['all', 'last', 'combine-last', 'combine-all'], default='last', help="Type of experiment evaluation.")
# Batch size
parser.add_argument('-bs', '--batch_size', type=int, default=64, help='Batch size.')
# Number of processes
parser.add_argument('-nw', '--num_workers', type=int, default=6, help='Number of processes (workers).')
# Select device "cuda" for GPU or "cpu"
parser.add_argument('--device', type=str, default=("cuda" if torch.cuda.is_available() else "cpu"), choices=['cuda', 'cpu'], help='Device to use. Choose "cuda" for GPU or "cpu".')
# Select GPU device
parser.add_argument('--gpu_device', type=int, default=None, help='ID of a GPU to use when multiple GPUs are available.')
# Use multiple GPUs?
parser.add_argument('--multi_gpu', action='store_true', help='Flag whether to use all available GPUs.')
args = parser.parse_args()

print(f"Loading snapshots from experiment: {args.ckpt_dir}")

idx2label = cd.SoundData().idx2label
#sound_data = cd.SoundData(phase='test', num_processes=args.num_workers)
testset = cd.TestDset(num_processes=args.num_workers, transform=cd.data_transforms[f'{args.cache_prefix}_test'])
testloader = thd.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
device = torch.device(args.device)
RES_DIR = os.path.join(CKPT_DIR, args.ckpt_dir)
snaps_dir = os.path.join(RES_DIR, 'snaps')
runs = [os.path.join(snaps_dir, run_name) for run_name in sorted(os.listdir(snaps_dir))]
if args.type.startswith('combine'):
	RES_DIR2 = os.path.join(CKPT_DIR, args.ckpt_dir2)
	snaps_dir2 = os.path.join(RES_DIR2, 'snaps')
	runs += [os.path.join(snaps_dir2, run_name) for run_name in sorted(os.listdir(snaps_dir2))]
is_ensemble = len(runs) > 1

def eval_model(loader, model, model_num):
	predictions = defaultdict(list)
	pbar = tqdm(loader, total=(len(loader.dataset)//args.batch_size + 1), desc=f'Evaluation model {model_num}')
	with torch.no_grad():
		for inputs, ids in pbar:
			inputs = inputs.to(device).float()

			outputs = F.softmax(model(inputs), dim=1)

			for i in range(len(ids)):
				predictions[ids[i].item()].append(outputs[i].cpu())
	
	predicted = {}
	for idx in sorted(predictions.keys()):
		predicted[idx] = torch.stack(predictions[idx], dim=0).numpy().mean(axis=0)
	
	return predicted

# List of dictionaries
results = []
for split_num, run_dir in enumerate(runs):
	if args.type.endswith('last'):
		for model_num, mname in enumerate(os.listdir(run_dir)):
			if mname.endswith('last.model'):
				print(f"Evaluating model {mname}")
				model = torch.load(os.path.join(run_dir, mname))
				if args.multi_gpu:
					model = nn.DataParallel(model)
				results.append(eval_model(testloader, model, split_num))
	elif args.type.endswith('all'):
		for model_num, mname in enumerate(os.listdir(run_dir)):
			if mname.endswith('.model'):
				print(f"Evaluating model {mname}")
				model = torch.load(os.path.join(run_dir, mname))
				if args.multi_gpu:
					model = nn.DataParallel(model)
				results.append(eval_model(testloader, model, f'{split_num} / {model_num}'))

# Dictionary of lists / np.arrays
results = {k: np.array([dic[k] for dic in results]) for k in results[0]}

for key, value in results.items():
	results[key] = value.mean(axis=0)

preds = {}
for key, value in results.items():
	preds[key] = value.argsort()[-3:][::-1].tolist()

subm = pd.DataFrame.from_dict(preds, columns=[f'label{i}' for i in range(3)], orient='index')
subm['fname'] = subm.index
subm['fname'] = subm.fname.apply(lambda x: testset.idx2fname[x])
subm = subm[['fname', 'label0', 'label1', 'label2']]
for i in range(3):
	subm[f'label{i}'] = subm[f'label{i}'].apply(lambda x: idx2label[x])
subm['label'] = subm.label0 + ' ' + subm.label1 + ' ' + subm.label2
subm = subm.drop([f'label{i}' for i in range(3)], axis=1)
for fname in ['0b0427e2.wav', '6ea0099f.wav', 'b39975f5.wav']:
	subm.loc[subm.shape[0], 'fname'] = fname
	subm.loc[subm.shape[0]-1, 'label'] = 'Laughter Hi-Hat Flute'
if not subm.shape[0] == 9400:
	import pdb; pdb.set_trace()
subm.to_csv(os.path.join(RES_DIR, f'submission-{args.type}.csv'), index=False)
