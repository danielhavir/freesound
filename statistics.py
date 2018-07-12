import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as thd
from sklearn.model_selection import KFold, StratifiedKFold
import core.dataset as cd
import core.eval_utils as evaluate
from tqdm import tqdm
from collections import defaultdict
import argparse

CKPT_DIR = os.path.join('checkpoints')

# Collect arguments (if any)
parser = argparse.ArgumentParser()

# Pretrained model
parser.add_argument('model', type=str, help="Model to run.")
# Checkpoint directory
parser.add_argument('-dir', '--ckpt_dir', type=str, choices=os.listdir(CKPT_DIR), default=sorted(os.listdir(CKPT_DIR))[-1], help="Checkpoints dir.")
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
# Seed
parser.add_argument('-s', '--seed', type=int, default=42, help='Random state seed.')
args = parser.parse_args()

print(f"Loading snapshots from experiment: {args.ckpt_dir}")

sound_data = cd.SoundData(num_processes=args.num_workers, seed=args.seed)
device = torch.device(args.device)
RES_DIR = os.path.join(CKPT_DIR, args.ckpt_dir)
snaps_dir = os.path.join(RES_DIR, 'snaps')
runs = [os.path.join(snaps_dir, run_name) for run_name in sorted(os.listdir(snaps_dir))]
kfold = StratifiedKFold(n_splits=len(runs), random_state=args.seed)

def eval_model(loader, model, model_num, phase):
	total_loss = 0.0
	predictions = defaultdict(list)
	labels = defaultdict(list)
	pbar = tqdm(loader, total=(len(loader.dataset)//args.batch_size + 1), desc=f'Evaluation model {model_num}')
	with torch.no_grad():
		for inputs, targets, ids in pbar:
			inputs, targets = inputs.to(device).float(), targets.to(device)

			outputs = model(inputs)
			# No Mixup -> use CrossEntropy
			loss = F.cross_entropy(outputs, targets)
			total_loss += loss.item()

			preds = F.softmax(outputs, dim=1)

			for i in range(len(ids)):
				predictions[ids[i].item()].append(preds[i].cpu())
				labels[ids[i].item()].append(targets[i].item())
	
	predicted = []; true = []
	for idx in sorted(predictions.keys()):
		predicted.append( torch.stack(predictions[idx], dim=0).numpy().mean(axis=0).argsort()[-3:][::-1] )
		true.append( min(labels[idx]) )
	predicted = np.array(predicted)
	true = np.array(true)
	total_loss /= (len(loaders[phase].dataset)//args.batch_size + 1)
	acc, precision, recall = evaluate.print_results(true, predicted[:,0], total_loss, phase, cm=False)

# List of dictionaries
for split_num, (train, test) in enumerate(kfold.split(sound_data.idxs, sound_data.df.target)):
	sound_data.reset_index(train, test)
	train_df, test_df = sound_data.get_train_test_split()
	trainset = cd.Dset(train_df, args.num_workers, transform=cd.data_transforms['train'], phase='train')
	testset = cd.Dset(test_df, args.num_workers, transform=cd.data_transforms['test'], phase='test')
	model = torch.load(os.path.join(runs[split_num], args.model + '-last.model'))
	loaders = {'train': thd.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
	'test': thd.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)}
	for phase in ['train', 'test']: eval_model(loaders[phase], model, split_num, phase)
