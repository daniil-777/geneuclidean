import os, sys

import argparse

# from release import *
from src.utils import config 
# import utils.config as config
import multiprocessing

import numpy as np
from numpy import savetxt
import torch

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# from utils import Utils
import argparse
import sys
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
from src.utils.build_vocab import Vocabulary
from src.datasets.data_loader import get_loader, Pdb_Dataset, collate_fn, collate_fn_masks
from src.training.train_check_att_vis import Trainer_Attention_Check_Vis
from src.tests.training.train_checkpoint import Trainer_Fold
from src.sampling.sampler import Sampler
from src.datasets.split import Splitter
from src.training.utils import save_checkpoint_sampling
from src.evaluation.analysis import plot_all
from src.tests.datasets.feature import Featuring
import warnings

def test_Feature_exists():
    parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model.'
)
    
    parser.add_argument('--config', type=str, help='Path to config file.')
    parser.add_argument('--radious', type=int , default=8, help='dimension of word embedding vectors')
    parser.add_argument('--type_feature', type=str , default='mass_charge', help='type_feature')
    parser.add_argument('--type_filtering', type=str , default = 'all', help='type_filtering')
    parser.add_argument('--h_filterig', type=str , default='without_h', help='h')
    parser.add_argument('--type_fold', type=str, help='type_fold')
    parser.add_argument('--idx_fold', type=str, help='Path to config file.')
    args = parser.parse_args()
                         
    cfg = config.load_config(args.config, 'configurations/config_lab/default.yaml')
    type_fold = args.type_fold
    idx_fold = args.idx_fold
    savedir = cfg["output_parameters"]["savedir"]
    model_name = cfg["model_params"]["model_name"]
    num_epoches = cfg["model_params"]["num_epochs"]


    #features generation
    print("Checking saved features!")
    names_prot_exceptions = []
    Feature_gen = Featuring(cfg, args.radious, args.type_feature, args.type_filtering, args.h_filterig)
    for pdbid in Feature_gen.idx_files_refined:
        name_protein = Feature_gen.files_refined[pdbid]
        files = os.listdir(os.path.join(Feature_gen.init_refined, name_protein))
        array_feat_names = [name_protein, "feature", "r", str(args.radious), args.type_feature, args.type_filtering, args.h_filterig]
        name_feature = "_".join(array_feat_names) + ".pt"
        if name_feature in files:
            path_feat = os.path.join(Feature_gen.init_refined, name_protein, name_feature)
            feature_filt = torch.load(path_feat, map_location=torch.device('cpu')).long()
            if feature_filt.shape[1] == 3:
                print("exception! - ", name_protein)
                names_prot_exceptions.append(name_protein)
        else:
            print("no feature! - ", name_protein)
    print(names_prot_exceptions)

if __name__ == "__main__":
    test_Feature_exists()