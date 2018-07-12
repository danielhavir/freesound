import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse

RESULTS_DIR = os.path.join('checkpoints')

parser = argparse.ArgumentParser()

parser.add_argument('run_dir', nargs='?', type=str, default=sorted(os.listdir(RESULTS_DIR))[-1], help="Directory with statistics. (Last experiment by default)")
parser.add_argument('--no_show', action='store_true', help='Flag whether to show a matplotlib window.')
parser.add_argument('--no_save', action='store_true', help='Flag whether to save figure.')

args = parser.parse_args()

print(f'Loading experiment {args.run_dir}')
RUN_DIR = os.path.join(RESULTS_DIR, args.run_dir)

stats_files = [i for i in os.listdir(RUN_DIR) if i.startswith('stats')]
num_splits = len(stats_files)

stats = {}
if num_splits == 1:
    with open(os.path.join(RUN_DIR, 'stats-run.json'), 'r') as f:
        stats[0] = json.load(f)
else:
    for i in range(num_splits):
        with open(os.path.join(RUN_DIR, f'stats-run-{i}.json'), 'r') as f:
            stats[i] = json.load(f)
num_epochs = stats[0]['epochs']

metrics = ['accuracy', 'loss', 'recall', 'precision']
colors = ['b', 'r']
statistics = defaultdict(list)
snaps = []
for i in range(num_splits):
    for metric in metrics:
        for phase in ['train', 'test']:
            statistics[phase + "_" + metric].append(stats[i][phase][metric])
    
    if 'snaps' in stats[i]:
        snaps += stats[i]['snaps']

for key in statistics:
    statistics[key] = np.array(statistics[key])

min_precision_recall = min([statistics[phase + "_" + metric].min() for metric in ['precision', 'recall'] for phase in ['train', 'test']]) - 0.05
snaps = set(snaps)

plt.figure(figsize=(20,15))
for i, metric in enumerate(metrics):
    plt.subplot(3, 2, i+1)
    for j, phase in enumerate(['train', 'test']):
        plt.plot(statistics[phase + "_" + metric].mean(axis=0), color=colors[j])
        plt.fill_between(list(range(num_epochs)), statistics[phase + "_" + metric].min(axis=0), statistics[phase + "_" + metric].max(axis=0), color=colors[j], alpha=0.2)
    for x_coord in snaps:
        plt.axvline(x=x_coord-1, color='g', alpha=0.75, linewidth=0.9, linestyle='--')
    plt.grid()
    if metric in ['precision', 'recall']:
        plt.ylim(min_precision_recall, 1)
    plt.xlabel("Epoch number")
    plt.ylabel(metric.capitalize())
    plt.title(metric.capitalize())

if not args.no_save:
    plt.savefig(os.path.join(RUN_DIR, 'statistics.png'), bbox_inches='tight', pad_inches=0.2)
if not args.no_show:
    plt.show()
