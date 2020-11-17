
import argparse
import json
import multiprocessing
import os
import pickle
# from utils import Utils
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy import savetxt
from py3nvml import py3nvml
from sklearn.model_selection import KFold
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import src.utils.config as config
from src.datasets.data_loader import (Pdb_Dataset, collate_fn,
                                      collate_fn_masks, get_loader)
from src.datasets.feature import Featuring
from src.datasets.split import Splitter
from src.evaluation.analysis import plot_all
from src.evaluation.evaluator import Evaluator
from src.sampling.sampler import Sampler
from src.training.train_check_att_vis import Trainer_Attention_Check_Vis
from src.training.train_checkpoint import Trainer_Fold
# from src.training.training_feature import Trainer_Fold_Feature
# from src.training.training_feature_att import Trainer_Fold_Feature_Attention
from src.training.trainer import Trainer_Fold_Feature
from src.training.trainer_att import Trainer_Fold_Feature_Attention
from src.training.utils import save_checkpoint_sampling
from src.utils.build_vocab import Vocabulary
from src.utils.checkpoint import Checkpoint_Eval, Checkpoint_Fold


def main():
    parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model.'
)
    parser.add_argument('--loc', type=str, help='Location of running')
    parser.add_argument('--config', type=str, help='Path to config file.')
    parser.add_argument('--radious', type=int , default=8, help='dimension of word embedding vectors')
    parser.add_argument('--type_feature', type=str , default='mass_charge', help='type_feature')
    parser.add_argument('--type_filtering', type=str , default = 'all', help='type_filtering')
    parser.add_argument('--h_filterig', type=str , default='without_h', help='h')
    parser.add_argument('--type_fold', type=str, help='type_fold')
    # parser.add_argument('--idx_fold', type=str, help='Path to config file.')
    args = parser.parse_args()
    
    if args.loc == 'lab':
        config_file_path = 'configurations/config_lab/default.yaml'
    else:
        config_file_path = 'configurations/config_local/default.yaml'
                  
    cfg = config.load_config(args.config, config_file_path)
    type_fold = args.type_fold
    savedir = cfg["output_parameters"]["savedir"]
    cfg["sampling_params"]["type_fold"] = type_fold
    model_name = cfg["model_params"]["model_name"] + "_" + args.type_feature + "_" + str(args.radious) + "_" + args.type_filtering + "_" + args.h_filterig
    cfg["model_params"]["model_name"] = model_name
    num_epoches = cfg["model_params"]["num_epochs"]

    #features generation
    print("**********Checking features**************")
    Feature_gen = Featuring(cfg, args.radious, args.type_feature, args.type_filtering, args.h_filterig)

    cfg['model']['encoder_kwargs']['natoms'] = Feature_gen.max_length
    print("number of atoms: ", cfg['model']['encoder_kwargs']['natoms'])


    file_folds_checkpoint_path = os.path.join(savedir, model_name,  "checkpoints", "folds.csv")
    os.makedirs(os.path.join(savedir, model_name,  "checkpoints"), exist_ok=True)
    checkpoint_fold = Checkpoint_Fold(file_folds_checkpoint_path, type_fold)
    start_idx_fold = checkpoint_fold._get_current_fold()
    
    pipeline_checkpoint_path = os.path.join(savedir, model_name,  "checkpoints", 'pipeline.txt')
    file_pipeline_checkpoint = open(pipeline_checkpoint_path, "a+")
       
    # get split folds file
    file_idx_split = os.path.join(cfg['output_parameters']['savedir'], model_name,  "logs", "idxs", type_fold)
    print("file_idx_split", file_idx_split)
    if not os.path.exists(file_idx_split):
        print("doing split...")
        splitter = Splitter(cfg)
        splitter.split(type_fold)

    #training + validation + pca
    for idx_fold in range(start_idx_fold, 5):
        print("Doing Train/Val on the fold - ",idx_fold)
        if(cfg['training_params']['mode'] == "no_attention"):
            trainer = Trainer_Fold_Feature(cfg, idx_fold)
            trainer.train_epochs(Feature_gen)
        elif(cfg['training_params']['mode'] == "attention"):
            trainer = Trainer_Fold_Feature_Attention(cfg, idx_fold)
            trainer.train_epochs(Feature_gen)
        #pca
        encoder_path = os.path.join(savedir, model_name, "models", "encoder-" + str(idx_fold) + "-" + str(num_epoches) + "-" + str(type_fold) + '.ckpt') 
        decoder_path = os.path.join(savedir, model_name, "models", "decoder-" + str(idx_fold) + "-" + str(num_epoches) + "-" + str(type_fold) + '.ckpt')
        sampler = Sampler(cfg, 'max', Feature_gen)
        print("Doing pca on the fold - ",idx_fold)
        sampler.save_encodings_all('test', idx_fold, encoder_path, decoder_path)
        sampler.save_encodings_all('train', idx_fold, encoder_path, decoder_path)
        #write fold id to checkpoint
        checkpoint_fold.write_checkpoint(idx_fold + 1)
  

    #Evaluation
    range_epochs = [1, 10, cfg['model_params']['num_epochs']]
    # regimes = ["probabilistic", "max", "beam_3", "beam_10"]
    regimes = ['beam_10']
    # regimes = ["probabilistic"]
    print("Evaluation starts!...")
    for regim in regimes:
        evaluator = Evaluator(cfg, regim, type_fold, range_epochs, Feature_gen)
        print("start run evaluation!...")
        evaluator.run_evaluation()

    #Plot similarities & Mol dostributions
    if "plot" not in file_pipeline_checkpoint.readlines():
        plot =  plot_all(cfg, num_epoches - 1)
        plot.run()
        #plot for every epoch
        # for epoch in range(num_epoches):
        #     plot =  plot_all(cfg, num_epoches - 1)
        #     plot.run()
        file_pipeline_checkpoint.write("plot")
        file_pipeline_checkpoint.flush()

if __name__ == "__main__":
    main()