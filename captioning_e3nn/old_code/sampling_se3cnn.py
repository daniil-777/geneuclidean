
import argparse
import config
from sampling.sampler import Sampler

import argparse
import config
import multiprocessing

import numpy as np
from numpy import savetxt
import torch

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import Utils
import argparse
import sys
import config
from py3nvml import py3nvml

import json
import os
import pickle

from sklearn.model_selection import KFold
import numpy as np

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from build_vocab import Vocabulary
from data_loader import get_loader, Pdb_Dataset, collate_fn, collate_fn_masks
from sampling.sampler import Sampler
# from utils import Utils



def main():
    parser = argparse.ArgumentParser(
    description='sample from trained model'
)
    parser.add_argument('config', type=str, help='Path to config file.')
    args = parser.parse_args()
    

    cfg = config.load_config(args.config, 'configurations/config_lab/default.yaml')
    savedir =  cfg['output_parameters']['savedir']
    encoder_path = os.path.join(savedir, "models", cfg['training_params']['encoder_name']) 
    decoder_path = os.path.join(savedir, "models", cfg['training_params']['decoder_name']) 

    
    split = 0
    # regimes = ["simple_probabilistic", "max", "temp_sampling", "simple_probabilistic_topk"]
    regimes = ["beam_1", "beam_3", "beam_10", "max", "temp_sampling_0.7", "probabilistic",
                "simple_probabilistic_topk_10"]

    for regim in regimes:
        print("doing sampling... ", regim)
        sampler = Sampler(cfg, regim)
        sampler.analysis_cluster(split, encoder_path, decoder_path)

if __name__ == "__main__":
    main()