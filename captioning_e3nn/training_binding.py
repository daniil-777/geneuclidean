
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
from data_loader_binding import Pdb_Dataset
# from training.trainer import train_loop, train_loop_mask
from training.train import Trainer
from training.binding import Trainer_Binding_Fold
from utils import Utils

# from training.train import Trainer
# from utils import Utils



def main():
    parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model.'
)
    parser.add_argument('config', type=str, help='Path to config file.')

    args = parser.parse_args()


    cfg = config.load_config(args.config, 'configurations/config_lab/default.yaml')
    trainer = Trainer_Binding_Fold(cfg)
    trainer.train_epochs()
   

if __name__ == "__main__":
    main()