import os, sys
import time
import json
import logging

RESULTS_DIR = os.path.join(os.getcwd(), 'checkpoints')
if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)

def setup_logger(logger: logging.Logger):
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)-8s %(message)s')
    std_handler = logging.StreamHandler(sys.stdout)
    std_handler.setFormatter(formatter)
    std_handler.setLevel(logging.INFO)
    logger.addHandler(std_handler)

    RUN_DIR = os.path.join(RESULTS_DIR, time.strftime('%Y%m%d-%X'))
    if not os.path.exists(RUN_DIR):
        os.mkdir(RUN_DIR)
    file_handler = logging.FileHandler(os.path.join(RUN_DIR, 'experiment.log'), encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger, RUN_DIR

def write_json(stats: dict, filepath: str='stats.json'):
    with open(filepath, 'w') as f:
        json.dump(stats, f, indent=2)
