
import os, sys

import argparse
import multiprocessing
from multiprocessing import Pool
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
from src.datasets.feature import Featuring
import warnings
warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model.')
    
    parser.add_argument('--config', type=str, help='Path to config file.')
    parser.add_argument('--radious', type=int , default=8, help='dimension of word embedding vectors')
    parser.add_argument('--type_feature', type=str , default='mass_charge', help='type_feature')
    parser.add_argument('--type_filtering', type=str , default = 'all', help='type_filtering')
    parser.add_argument('--h_filterig', type=str , default='without_h', help='h')
    parser.add_argument('--type_fold', type=str, help='type_fold')
    parser.add_argument('--idx_fold', type=str, help='idx fold')
    args = parser.parse_args()
                         
    cfg = config.load_config(args.config, 'configurations/config_local/default.yaml')
    type_fold = args.type_fold
    idx_fold = args.idx_fold
    savedir = cfg["output_parameters"]["savedir"]
    model_name = cfg["model_params"]["model_name"]
    num_epoches = cfg["model_params"]["num_epochs"]
    #features generation
    Feature_gen = Featuring(cfg, args.radious, args.type_feature, args.type_filtering, args.h_filterig)
    
    def get_unique_elems(pid):
        all_elems = list(set(Feature_gen._get_all_elem_general(pid)))
        return list(set(Feature_gen._get_all_elems(pid)))

    with Pool(processes=8) as pool:
        all_elems = pool.map(get_unique_elems, Feature_gen.idx_files_refined)
    all_elems = list(set(all_elems))
    print("all_elems - ", all_elems)
        # with tqdm(total=len(Feature_gen.idx_files_refined)) as pbar:
        #     for i, res in tqdm(enumerate(pool.imap_unordered(Feature_gen._get_length, Feature_gen.idx_files_refined))):
        #         all_elems.append(res)
        #         pbar.update()
        # all_elems = list(set(all_elems))
        # print("all_elems - ", all_elems)
    # for pid in Feature_gen.idx_files_refined:
    #     all_elems = list(set(Feature_gen._get_all_elems(pid)))
    #     # print("all_elems", list(set(all_elems)))
    #     elems_to_add = [elem for elem in all_elems if elem not in all_elems]
    #     all_elems.append(elems_to_add)
    # print("all_elems", all_elems)

if __name__ == "__main__":
    main()