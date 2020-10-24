import multiprocessing

import numpy as np
from numpy import savetxt
import torch

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
# from torch.utils.tensorboard import SummaryWriter


import argparse
import sys
import config
from rdkit import Chem
import json
import os
import csv
import pickle
import time 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
import numpy as np

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from build_vocab import Vocabulary
from data_loader import Pdb_Dataset
from Contrib.statistics import analysis_to_csv, analysis_to_csv_test
from visualisation import Visualisation



def main():
    parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model.'
)
    parser.add_argument('config', type=str, help='Path to config file.')

    args = parser.parse_args()

    cfg = config.load_config(args.config, 'configurations/config_lab/default.yaml')
    savedir =  cfg['output_parameters']['savedir']
    encoder_path = os.path.join(savedir, "models", cfg['training_params']['encoder_name']) 
    decoder_path = os.path.join(savedir, "models", cfg['training_params']['decoder_name'])

    visualiser = Visualisation(cfg, "beam")

    visualiser.save_for_vis(0, encoder_path, decoder_path)
    

if __name__ == "__main__":
    main()