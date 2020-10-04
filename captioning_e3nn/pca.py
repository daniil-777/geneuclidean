
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


def main():
    parser = argparse.ArgumentParser(
    description='sample from trained model'
)
    parser.add_argument('config', type=str, help='Path to config file.')
    args = parser.parse_args()
    

    cfg = config.load_config(args.config, 'configurations/config_lab/default.yaml')
    savedir =  cfg['output_parameters']['savedir']
    # encoder_path = os.path.join(savedir, "models", cfg['training_params']['encoder_name']) 
    # decoder_path = os.path.join(savedir, "models", cfg['training_params']['decoder_name']) 

    encoder_path = os.path.join(savedir, "models", "encoder_best_" + str(cfg['splitting']['id_fold']) + '.ckpt') 
    decoder_path = os.path.join(savedir, "models", "decoder_best_" + str(cfg['splitting']['id_fold']) + '.ckpt') 
    
    split = cfg['splitting']['id_fold']
    sampler = Sampler(cfg, 'max')
    sampler.save_encodings_all('test')

  
 

if __name__ == "__main__":
    main()