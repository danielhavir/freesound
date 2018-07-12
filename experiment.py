import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as thd
import core.dataset as cd
import core.eval_utils as evaluate
import core.logger as log
from sklearn.model_selection import KFold, StratifiedKFold
from model.densenet import *
from model.resnet import *
from core.mixup import Mixup, OneHotCrossEntropy
from core.snap_scheduler import SnapScheduler
from tqdm import tqdm
from collections import defaultdict
from copy import deepcopy
import argparse
import logging

logger = logging.getLogger()
logger, RUN_DIR = log.setup_logger(logger)

pretrained_models = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'densenet121': densenet121,
    'densenet169': densenet169,
    'densenet201': densenet201,
    'densenet161': densenet161,
}

class Experiment(object):
	def __init__(self, model: str, batch_size: int, epochs: int, lr: float, eval_interval: int=1,
	optimizer: str='sgd', schedule: str=None, step_size: int=10, gamma: float=0.5, use_mixup: bool=True,
	mixup_alpha: float=0.5, conv_fixed: bool=False, weighted: bool=False, cross_validate: bool=False,
	n_splits: int=5, seed: int=42, metric: str='accuracy', no_snaps: bool=False, debug_limit: int=None,
	device: str=('cuda' if torch.cuda.is_available() else 'cpu'), num_processes: int=8, multi_gpu: bool=False, **kwargs):
		self.set_seed(seed)
		self.model_str = model
		logger.info(f"Starting experiment with {self.model_str.capitalize()}")
		self.batch_size = batch_size
		self.epochs = epochs
		self.lr = lr
		self.eval_interval = eval_interval
		self.schedule = schedule
		self.step_size = step_size
		self.gamma = gamma
		self.optimizer_str = optimizer
		self.use_mixup = use_mixup
		self.conv_fixed = conv_fixed
		self.weighted = weighted
		self.cross_validate = cross_validate
		self.n_splits = n_splits
		self.metric = metric
		self.no_snaps = no_snaps
		self.debug_limit = debug_limit
		self.device = torch.device(device)
		self.num_processes = num_processes
		self.multi_gpu = multi_gpu

		self.sound_data = cd.SoundData(num_processes=self.num_processes, seed=seed)
		self.num_classes = self.sound_data.num_classes

		if not self.cross_validate:
			self.loaders = self.get_loaders()
		
		self.initial_best_threshold = 0.8
		self.emptystats = {
			'train': {
				'loss': [],
				'accuracy': [],
				'precision': [],
				'recall': []
			},
			'test': {
				'loss': [],
				'accuracy': [],
				'precision': [],
				'recall': []
			},
			'snaps': []
		}

		if self.use_mixup:
			self.criterion = OneHotCrossEntropy(self.device)
		else:
			self.criterion = nn.CrossEntropyLoss()

		if self.use_mixup:
			logger.info(f'Using mixup with alpha={mixup_alpha}')
			self.eye = torch.eye(self.num_classes).to(self.device)
			self.mixup = Mixup(mixup_alpha, self.device)
		
		self.model = self.load_model()
		
		if optimizer == 'sgd':
			if self.conv_fixed:
				self.optimizer = optim.SGD(self.model.fc.parameters(), lr=self.lr, momentum=0.9)
			else:
				self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
		elif optimizer == 'adam':
			if self.conv_fixed:
				self.optimizer = optim.Adam(self.model.fc.parameters(), lr=self.lr, amsgrad=False)
			else:
				self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, amsgrad=False)
		
		if self.schedule is not None:
			if self.schedule.lower() == 'step':
				logger.info(f"Scheduling learning rate every {self.step_size} with gamma = {self.gamma}")
				self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)
			elif self.schedule.lower() == 'snap':
				logger.info("Scheduling learning rate every using Snap Scheduler")
				self.scheduler = SnapScheduler(self.optimizer, num_epochs=self.epochs, num_snaps=4, init_lr=self.lr)
			elif self.schedule.lower() == 'exponential':
				logger.info(f"Scheduling learning rate exponentially with gamma = {self.gamma}")
				scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
		
		if self.cross_validate:
			logger.info(f"Using Stratified K-Folds Cross Validation with {self.n_splits} splits")
			self.kfold = StratifiedKFold(n_splits=self.n_splits, random_state=seed)
		
		if not os.path.exists(os.path.join(RUN_DIR, 'snaps')):
			os.mkdir(os.path.join(RUN_DIR, 'snaps'))

	def set_seed(self, seed):
		logger.info(f"Setting seed {seed}")
		if torch.cuda.is_available():
			torch.cuda.manual_seed(seed)
			torch.cuda.manual_seed_all(seed)
			torch.cuda.synchronize()

		torch.manual_seed(seed)
		np.random.seed(seed)
		# random.seed(seed)
	
	def get_loaders(self, num_workers=8):
		train_df, test_df = self.sound_data.get_train_test_split()
		if self.debug_limit is not None:
			logger.warning(f"Limiting dataset to {self.debug_limit} entries.")
			train_df = train_df.loc[:self.debug_limit]
			test_df = test_df.loc[:self.debug_limit]

		logging.info(f"Loading spectrograms")
		self.trainset = cd.Dset(train_df, self.num_processes, transform=cd.data_transforms['train'], phase='train')
		self.testset = cd.Dset(test_df, self.num_processes, transform=cd.data_transforms['test'], phase='test')

		return {'train': thd.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_processes),
				'test': thd.DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_processes)}
		
	def load_model(self):
		model = pretrained_models[self.model_str](pretrained=True)
		if self.conv_fixed:
			logger.warning("Fixing weights")
			for param in model.parameters():
				param.requires_grad = False

		classifier = lambda num_features: nn.Linear(num_features, self.num_classes)

		if self.model_str.startswith('densenet'):
			num_ftrs = model.classifier.in_features
			model.classifier = classifier(num_ftrs)
		elif self.model_str.startswith('resnet'):
			num_ftrs = model.fc.in_features
			model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
			model.fc = classifier(num_ftrs)
		else:
			raise ValueError(f'Invalid model string. Received {self.model_str}.')
		
		logger.info(f"Num params: {sum([np.prod(p.size()) for p in model.parameters()])}")
		logger.info(f"Num trainable params: {sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])}")

		if self.multi_gpu:
			model = nn.DataParallel(model)
		
		return model.to(self.device)
	
	def train_loop(self, epoch):
		train_loader = tqdm(self.loaders['train'], desc=f'TRAIN Epoch {epoch}',
							total=(len(self.trainset)//self.batch_size + 1))

		total_loss = 0.0
		correct = 0; total = 0;
		for inputs, targets, ids in train_loader:
			inputs, targets = inputs.to(self.device).float(), targets.to(self.device)

			if self.use_mixup:
				# OneHot encode
				targets = self.eye[targets]
				inputs, targets = self.mixup(inputs, targets)
			
			self.optimizer.zero_grad()

			outputs = self.model(inputs)

			loss = self.criterion(outputs, targets)
			loss.backward()
			self.optimizer.step()

			total_loss += loss.item()
			
			preds = torch.max(outputs, 1)[1].cpu()
			if self.use_mixup:
				targets = torch.max(targets, 1)[1].cpu()
			else:
				targets = targets.cpu()
			correct += (preds == targets).sum().item()
			total += targets.size(0)
			accuracy = correct / total

			train_loader.set_postfix(loss=round(total_loss / total, 4), acc=round(accuracy, 2))
		
		train_loader.close()
	
	def eval_loop(self, epoch, phase):
		total_loss = 0.0
		predictions = defaultdict(list)
		labels = defaultdict(list)
		eval_loader = tqdm(self.loaders[phase], desc=f'EVALUATION Epoch {epoch}', total=(len(self.loaders[phase].dataset)//self.batch_size + 1))

		with torch.no_grad():
			for inputs, targets, ids in eval_loader:
				inputs, targets = inputs.to(self.device).float(), targets.to(self.device)

				outputs = self.model(inputs)
				# No Mixup -> use CrossEntropy
				loss = F.cross_entropy(outputs, targets)
				total_loss += loss.item()

				preds = F.softmax(outputs, dim=1)

				for i in range(len(ids)):
					predictions[ids[i].item()].append(preds[i].cpu())
					labels[ids[i].item()].append(targets[i].item())
		
		predicted = []; true = []
		for idx in sorted(predictions.keys()):
			predicted.append( np.argmax(torch.stack(predictions[idx], dim=0).numpy().mean(axis=0)) )
			true.append( min(labels[idx]) )
		predicted = np.array(predicted)
		true = np.array(true)
		total_loss /= (len(self.loaders[phase].dataset)//self.batch_size + 1)
		acc, precision, recall = evaluate.print_results(true, predicted, total_loss, phase, epoch, cm=False, print_fc=logger.info)
		self.stats[phase]['accuracy'].append(acc)
		self.stats[phase]['loss'].append(total_loss)
		self.stats[phase]['precision'].append(precision)
		self.stats[phase]['recall'].append(recall)
	
	def save_model(self, snaps_dir, snap_fname):
		torch.save(self.model, os.path.join(snaps_dir, snap_fname))
	
	def single_run(self, run_fname='run'):
		self.stats = deepcopy(self.emptystats)
		self.stats['epochs'] = self.epochs

		snaps_dir = os.path.join(RUN_DIR, 'snaps', run_fname)
		if not os.path.exists(snaps_dir):
			os.mkdir(snaps_dir)

		best_score = self.initial_best_threshold

		for epoch in range(1, self.epochs+1):
			self.train_loop(epoch)
			if epoch % self.eval_interval == 0:
				for phase in ['train', 'test']: self.eval_loop(epoch, phase)
		
				test_score = self.stats['test'][self.metric][-1]

				if not self.no_snaps and test_score > best_score:
					best_score = test_score
					self.save_model(snaps_dir, self.model_str + f'-{epoch}-{int(test_score*100)}.model')
					self.stats['snaps'].append(epoch)
			
			if self.schedule is not None:
				if self.schedule == 'snap':
					if not self.no_snaps and self.scheduler.save_model(epoch):
						self.save_model(snaps_dir, self.model_str + f'snap-{epoch}.model')
				self.scheduler.step()
		
		stats_fname = 'stats-' + run_fname + '.json'
		log.write_json(self.stats, filepath=os.path.join(RUN_DIR, stats_fname))
		
		if not self.no_snaps:
			self.save_model(snaps_dir, self.model_str + '-last.model')
	
	def split_run(self):
		for split_num, (train, test) in enumerate(self.kfold.split(self.sound_data.idxs, self.sound_data.df.target)):
			logging.info(2*'#' + f" Running split {split_num+1}/{self.n_splits} " + 2*'#')
			self.sound_data.reset_index(train, test)
			self.loaders = self.get_loaders()

			if split_num > 0:
				# Reinitialize model and optimizer

				logger.info('Reinitializing model')
				self.model = self.load_model()
				
				if self.optimizer_str == 'sgd':
					if self.conv_fixed:
						self.optimizer = optim.SGD(self.model.fc.parameters(), lr=self.lr, momentum=0.9)
					else:
						self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
				elif self.optimizer_str == 'adam':
					if self.conv_fixed:
						self.optimizer = optim.Adam(self.model.fc.parameters(), lr=self.lr, amsgrad=False)
					else:
						self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, amsgrad=False)
			
			self.single_run(run_fname=f'run-{split_num}')
	
	def run(self):
		if self.no_snaps:
			logger.info('Preventing from snapshots')
			
		if self.cross_validate:
			self.split_run()
		else:
			self.single_run()
			


if __name__ == '__main__':
	# Collect arguments (if any)
	parser = argparse.ArgumentParser()

	# Pretrained model
	parser.add_argument('model', type=str, choices=pretrained_models.keys(), help="Model to run.")
	# Batch size
	parser.add_argument('-bs', '--batch_size', type=int, default=64, help='Batch size.')
	# Epochs
	parser.add_argument('-e', '--epochs', type=int, default=50, help='Number of epochs.')
	# Optimizer
	parser.add_argument('-o', '--optim', type=str, choices=['sgd', 'adam'], default='sgd', help='Optimizer.')
	# Learning rate
	parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='Learning rate.')
	# Scheduler
	parser.add_argument('--scheduler', type=str, choices=['step', 'snap', 'exponential'], default=None, help='Scheduler.')
	# Gamma argument for scheduler (only applies to step and exponential)
	parser.add_argument('--gamma', type=float, default=0.5, help='Gamma argument for scheduler (only applies to step and exponential).')
	# Prevent from using mixup
	parser.add_argument('--no_mixup', action='store_true', help='Flag whether to use mixup.')
	# Fix weights of convolutional layers
	parser.add_argument('--conv_fixed', action='store_true', help='Flag whether to fix weights of convolutional layers.')
	# Weight classes to tackle inbalance
	parser.add_argument('-w', '--weighted', action='store_true', help='Flag whether to weight classes.')
	# Use cross validation
	parser.add_argument('-cv', '--cross_validate', action='store_true', help='Flag whether to use cross validation.')
	# Alpha parameter for Mixup's Beta distribution
	parser.add_argument('-alpha', '--mixup_alpha', type=float, default=0.8, help="Alpha parameter for Mixup's Beta distribution.")
	# Prevent from storing snapshots
	parser.add_argument('--no_snaps', action='store_true', help='Flag whether to prevent from storing snapshots.')
	# Debug limit to decrease size of dataset
	parser.add_argument('--debug_limit', type=int, default=None, help='Debug limit to decrease size of dataset.')
	# Seed
	parser.add_argument('-s', '--seed', type=int, default=42, help='Random state seed.')
	# Number of processes
	parser.add_argument('-nw', '--num_workers', type=int, default=6, help='Number of processes (workers).')
	# Select device "cuda" for GPU or "cpu"
	parser.add_argument('--device', type=str, default=("cuda" if torch.cuda.is_available() else "cpu"), choices=['cuda', 'cpu'], help='Device to use. Choose "cuda" for GPU or "cpu".')
	# Select GPU device
	parser.add_argument('--gpu_device', type=int, default=None, help='ID of a GPU to use when multiple GPUs are available.')
	# Use multiple GPUs?
	parser.add_argument('--multi_gpu', action='store_true', help='Flag whether to use all available GPUs.')
	args = parser.parse_args()

	if args.gpu_device is not None:
		torch.cuda.set_device(args.gpu_device)

	exp = Experiment(args.model, args.batch_size, args.epochs, args.learning_rate, use_mixup=(not args.no_mixup),
	mixup_alpha=args.mixup_alpha, conv_fixed=args.conv_fixed, weighted=args.weighted, cross_validate=args.cross_validate, schedule=args.scheduler,
	seed=args.seed, no_snaps=args.no_snaps, debug_limit=args.debug_limit, num_processes=args.num_workers, multi_gpu=args.multi_gpu)
	exp.run()
