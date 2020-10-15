
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
# from training.trainer import train_loop, train_loop_mask
from training.train import Trainer
# from training.train_fold import Trainer_Fold
from training.train_check_att_vis import Trainer_Attention_Check_Vis
from training.train_checkpoint import Trainer_Fold
from sampling.sampler import Sampler


def main():
    parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model.'
)
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('type_fold', type=str, help='type_fold')
    parser.add_argument('idx_fold', type=str, help='Path to config file.')
    args = parser.parse_args()
                         
    cfg = config.load_config(args.config, 'configurations/config_lab/default.yaml')
    type_fold = args.type_fold
    idx_fold = args.idx_fold
    savedir = cfg["output_parameters"]["savedir"]

    if(cfg['training_params']['mode'] == "no_attention"):
        trainer = Trainer_Fold(cfg)
        trainer.train_epochs()
    elif(cfg['training_params']['mode'] == "attention"):
        trainer = Trainer_Attention_Check_Vis(cfg)
        trainer.train_epochs()
    
    encoder_path = os.path.join(savedir, "models", "encoder_best_" + str(idx_fold) + '.ckpt') 
    decoder_path = os.path.join(savedir, "models", "decoder_best_" + str(idx_fold) + '.ckpt') 

    split = cfg['splitting']['id_fold']
    # regimes = ["simple_probabilistic", "max", "temp_sampling", "simple_probabilistic_topk"]
    # regimes = ["beam_1", "beam_3", "beam_10", "max", "temp_sampling_0.7", "probabilistic",
    #             "simple_probabilistic_topk_10"]
    regimes = ["probabilistic", "max"]
    for regim in regimes:
        print("doing sampling... ", regim)
        sampler = Sampler(cfg, regim)
        sampler.analysis_cluster(idx_fold, type_fold, encoder_path, decoder_path)


if __name__ == "__main__":
    main()