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
from data_loader import get_loader, Pdb_Dataset, collate_fn
from models_new import DecoderRNN, Encoder_se3ACN, MyDecoderWithAttention
from captioning_e3nn.training.trainer import train_loop, train_loop_mask
# from utils import Utils



DATA_PATH = os.path.realpath(os.path.dirname(__file__))
N_SPLITS = 5

# args = str(sys.argv[1])
# print(args)

# Arguments
parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model.'
)
parser.add_argument('config', type=str, help='Path to config file.')

args = parser.parse_args()


cfg = config.load_config(args.config, 'configurations/config_lab/default.yaml')


# configuration = utils.parse_configuration()

# model params
num_epochs = cfg['model_params']['num_epochs']
batch_size = cfg['model_params']['batch_size']
learning_rate = cfg['model_params']['learning_rate']
num_workers = cfg['model_params']['num_workers']

# training params
protein_dir = cfg['training_params']['image_dir']
caption_path = cfg['training_params']['caption_path']
log_step = cfg['training_params']['log_step']
save_step = cfg['training_params']['save_step']
vocab_path = cfg['preprocessing']['vocab_path']

#output files
savedir = cfg['output_parameters']['savedir']
tesnorboard_path = savedir
model_path = os.path.join(savedir, "models")
log_path = os.path.join(savedir, "logs")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Encoder, Decoder = config.get_model(cfg, device=device)

nparameters_enc = sum(p.numel() for p in Encoder.parameters())
nparameters_dec = sum(p.numel() for p in Decoder.parameters())
print(Encoder)
print('Total number of parameters: %d' % (nparameters_enc + nparameters_dec))

if not os.path.exists(log_path):
    os.makedirs(log_path)

if not os.path.exists(model_path):
    os.makedirs(model_path)

    
test_idx_file = open(os.path.join(log_path, "test_idx.txt"), "w")
log_file = open(os.path.join(log_path, "log.txt"), "w")
log_file_tensor = open(os.path.join(log_path, "log_tensor.txt"), "w")
writer = SummaryWriter(tesnorboard_path)



# Load vocabulary wrapper
with open(vocab_path, "rb") as f:
    vocab = pickle.load(f)


if __name__ == "__main__":
    # get indexes of all complexes and "nick names"
    featuriser = Pdb_Dataset(cfg, vocab=vocab)
    # data_ids, data_names = utils._get_refined_data()
    files_refined = os.listdir(protein_dir)
    # data_ids = np.array([i for i in range(len(files_refined) - 3)])
    data_ids = np.array([i for i in range(20)])

    #cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=2)
    my_list = list(kf.split(data_ids))
    test_idx = []
    # output memory usage
    py3nvml.nvmlInit()
    for split_no in range(N_SPLITS):
        train_id, test_id = my_list[split_no]
        train_data = data_ids[train_id]
        test_data = data_ids[test_id]
        with open(os.path.join(savedir, 'test_idx_' + str(split_no)), 'wb') as fp:
            pickle.dump(test_data, fp)
        
        test_idx.append(test_data)
        test_idx_file.write(str(test_data) + "\n")
        test_idx_file.flush()

        feat_train = [featuriser[data] for data in train_data]

        loader_train = DataLoader(feat_train, batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=num_workers,
                                    collate_fn=collate_fn,)

        total_step = len(loader_train)
        print("total_step", total_step)
        encoder = Encoder
        decoder = Decoder

        criterion = nn.CrossEntropyLoss()
        # params_encoder = filter(lambda p: p.requires_grad, encoder.parameters())

        caption_params = list(decoder.parameters()) + list(encoder.parameters())
        caption_optimizer = torch.optim.Adam(caption_params, lr=learning_rate)

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(caption_optimizer, 'min')
        for epoch in range(num_epochs):
            if(cfg['preprocessing']['mask'] == True):
                train_loop_mask(loader_train, encoder, decoder,caption_optimizer, split_no, epoch, total_step)
            else:
                train_loop(loader_train, encoder, decoder,caption_optimizer, split_no, epoch, total_step)

            
       
    




        



        





