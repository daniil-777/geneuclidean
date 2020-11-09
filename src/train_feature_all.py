
import argparse
import src.utils.config as config
import multiprocessing

import numpy as np
from numpy import savetxt
import torch

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# from utils import Utils
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
from src.training.training_feature_att import Trainer_Fold_Feature_Attention
from src.training.train_checkpoint import Trainer_Fold
from src.training.training_feature import Trainer_Fold_Feature
from src.sampling.sampler import Sampler
from src.datasets.split import Splitter
from src.training.utils import save_checkpoint_sampling
from src.evaluation.analysis import plot_all
from src.datasets.feature import Featuring


def main():
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
    model_name = cfg["model_params"]["model_name"] + "_" + args.type_feature + "_" + str(args.radious) + "_" + args.type_filtering + "_" + args.h_filterig
    cfg["model_params"]["model_name"] = model_name
    num_epoches = cfg["model_params"]["num_epochs"]
    
    #features generation
    print("**********Checking features**************")
    Feature_gen = Featuring(cfg, args.radious, args.type_feature, args.type_filtering, args.h_filterig)
    cfg['model']['encoder_kwargs']['natoms'] = Feature_gen.max_length
    print("number of atoms: ", cfg['model']['encoder_kwargs']['natoms'])


    # get split folds file
    dir_idx_split = os.path.join(cfg['output_parameters']['savedir'], model_name,  "logs", "idxs", cfg['splitting']['file_folds'])
    if not os.path.exists(dir_idx_split):
        print("***********doing split...***********")
        splitter = Splitter(cfg)
        splitter.split(type_fold)



    #training + evaluation
    if(cfg['training_params']['mode'] == "no_attention"):
        trainer = Trainer_Fold_Feature(cfg, idx_fold)
        trainer.train_epochs(Feature_gen)
    elif(cfg['training_params']['mode'] == "attention"):
        trainer = Trainer_Fold_Feature_Attention(cfg, idx_fold)
        trainer.train_epochs()
    
    # encoder_path = os.path.join(savedir, "models", "encoder_best_" + str(idx_fold) + '.ckpt') 
    # decoder_path = os.path.join(savedir, "models", "decoder_best_" + str(idx_fold) + '.ckpt')
    encoder_path = os.path.join(savedir, model_name, "models", "encoder-" + str(idx_fold) + "-" + str(num_epoches) + '.ckpt') 
    decoder_path = os.path.join(savedir, model_name, "models", "decoder-" + str(idx_fold) + "-" + str(num_epoches) + '.ckpt')
    checkpoint_sampling_path = os.path.join(savedir, model_name,  "checkpoints", str(idx_fold) + '_sample.pkl')
    pipeline_checkpoint_path = os.path.join(savedir, model_name,  "checkpoints", str(idx_fold) + 'pipeline.txt')
    file_pipeline_checkpoint = open(pipeline_checkpoint_path, "a+")
   
    # regimes = ["simple_probabilistic", "max", "temp_sampling", "simple_probabilistic_topk"]
    # regimes = ["beam_1", "beam_3", "beam_10", "max", "temp_sampling_0.7", "probabilistic",
    #             "simple_probabilistic_topk_10"]
    #sampling
    # regimes = ["probabilistic", "max", "beam_1", "beam_3", "beam_10"]
    if "pca" not in file_pipeline_checkpoint.readlines():
        print("*****doing pca********")
        sampler = Sampler(cfg, 'max', Feature_gen)
        sampler.save_encodings_all('test', idx_fold, encoder_path, decoder_path)
        sampler.collect_all_encodings()
        sampler.save_encodings_all('train', idx_fold, encoder_path, decoder_path)
        sampler.collect_all_encodings()
        file_pipeline_checkpoint.write("pca")
    


    regimes = ["probabilistic", "max", "beam_1", "beam_3", "beam_10", "beam_20"]
    end_sampling_ind = len(regimes)
    if (os.path.exists(checkpoint_sampling_path)):
        print("loading sample ids...")
        checkpoint_sampling = torch.load(checkpoint_sampling_path)
        start_sampling_ind = checkpoint_sampling['idx_sample_regime_start']
        print("************start_sampling_ind***********", start_sampling_ind)
    else:
        start_sampling_ind = 0
        save_checkpoint_sampling(checkpoint_sampling_path, 0, 0)

    for sampling_ind in range(start_sampling_ind, end_sampling_ind):
        sample = regimes[sampling_ind]
        print("*********sample regim*********** ", sample)
        sampler = Sampler(cfg, sample, Feature_gen)
        sampler.analysis_cluster(idx_fold, type_fold, encoder_path, decoder_path)
    
    if "plot" not in file_pipeline_checkpoint.readlines():
        plot =  plot_all(cfg)
        plot.run()
        file_pipeline_checkpoint.write("plot")
    # for regim in regimes:
    #     print("doing sampling... ", regim)
    #     sampler = Sampler(cfg, regim)
    #     sampler.analysis_cluster(idx_fold, type_fold, encoder_path, decoder_path)


if __name__ == "__main__":
    main()