import multiprocessing

import numpy as np
from numpy import savetxt
import torch
from torchsummary import summary
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from tqdm import tqdm


import argparse
import sys
import config
from py3nvml import py3nvml
import tqdm
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
from data_loader_binding import Pdb_Dataset, Loss
from models_new import DecoderRNN, Encoder_se3ACN, MyDecoderWithAttention
from sampling.sampler import Sampler
from utils import Utils
import sys
import numpy as np
from numpy import savetxt





class Trainer_Binding_Fold():
    def __init__(self, cfg):
        # model params
        self.cfg = cfg
        self.original_stdout = sys.stdout
        #folds data
        self.name_file_folds = cfg['splitting']['file_folds']
        self.fold_number = cfg['splitting']['id_fold']
        
        self.num_epochs = cfg['model_params']['num_epochs']
        self.N_EPOCHS = cfg['model_params']['num_epochs']
        self.BATCH_SIZE = cfg['model_params']['batch_size']
        self.learning_rate = cfg['model_params']['learning_rate']
        self.NUM_WORKERS = cfg['model_params']['num_workers']

        # training params
        self.protein_dir = cfg['training_params']['image_dir']
        self.caption_path = cfg['training_params']['caption_path']
        self.log_step = cfg['training_params']['log_step']
        self.save_step = cfg['training_params']['save_step']
        self.vocab_path = cfg['preprocessing']['vocab_path']
        self.n_splits = cfg['training_params']['n_splits']
        self.loss_best = np.inf

        #output files
        self.savedir = cfg['output_parameters']['savedir']
        self.name_plot = cfg['output_parameters']['name_plot']
        self.tesnorboard_path = os.path.join(self.savedir, "tensorboard")
        self.model_path = os.path.join(self.savedir, "models")
        self.log_path = os.path.join(self.savedir, "logs")
        self.PKD_PATH = os.path.join(self.savedir, "logs")
        self.PATH_PLOTS = os.path.join(self.savedir, "plots")
        self.idx_file = os.path.join(self.log_path, "idxs")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir_smiles = os.path.join(self.savedir, "statistics")
        if not os.path.exists(self.tesnorboard_path):
            os.makedirs(self.tesnorboard_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        if not os.path.exists(self.PATH_PLOTS):
            os.makedirs(self.PATH_PLOTS)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.idx_file):
            os.makedirs(self.idx_file)

        if not os.path.exists(self.save_dir_smiles):
            os.makedirs(self.save_dir_smiles)
        #log files 
        self.test_idx_file = open(os.path.join(self.idx_file, "test_idx.txt"), "w")
        self.log_file = open(os.path.join(self.log_path, "log.txt"), "w")
        self.log_file_tensor = open(os.path.join(self.log_path, "log_tensor.txt"), "w")
        self.writer = SummaryWriter(self.tesnorboard_path)
        
        self.Encoder  =  config.get_model_binding(self.cfg, device=self.device)
        # self.input = self.cfg.get_shape_input(self.cfg)
        # print(summary(self.Encoder, self.input))
        # print(summary(self.Decoder))
        print(self.Encoder)
        with open(os.path.join(self.log_path, "model.txt"), 'w') as f:
            sys.stdout = f # Change the standard output to the file we created.
            # print(summary(self.Encoder, self.input))
            # print(summary(self.Decoder))
            print(self.Encoder)
  
            sys.stdout = self.original_stdout
        self.tils = Utils(self.cfg)

        #print all params
        nparameters_enc = sum(p.numel() for p in self.Encoder.parameters())
        with open(os.path.join(self.log_path, "model.txt"), 'w') as f:
            f.write('Total number of parameters: %d' % (nparameters_enc))

        with open(self.vocab_path, "rb") as f:
            self.vocab = pickle.load(f)
        self.criterion = nn.CrossEntropyLoss()

    def train_loop_mask(self, model, loss_cl, opt, epoch):

        target_pkd_all = []
        model = model.train()
        progress = tqdm(loader)
        all_rmsd = []
        pkd_pred = []
        for idx, features, geometry, masks, target_pkd in progress:
            idx = idx.to(self.device)
            features = features.to(self.device)
            geometry = geometry.to(self.device)
            masks = masks.to(self.device)
            # num_atoms= num_atoms.to(self.device)
            target_pkd = target_pkd.to(self.device)
            target_pkd_all.append(target_pkd)
            opt.zero_grad()

            # out1 = model(features, geometry)
            out1 = model(geometry, features, masks)
            pkd_pred.append(out1.cpu())
            # print(out1.cpu())
            loss_rmsd_pkd = loss_cl(out1, target_pkd).float()

            self.writer.add_scalar("training_loss", loss_rmsd_pkd.item(), epoch)

            loss_rmsd_pkd.backward()
            opt.step()
            all_rmsd.append(loss_rmsd_pkd.item())
        return torch.cat(target_pkd_all), torch.cat(pkd_pred), sum(all_rmsd) / len(all_rmsd)


    def eval_loop(self, loader, model, epoch):
        """
        Evaluation loop using `model` and data from `loader`.
        """
        model = model.eval()
        progress = tqdm(loader)
        
        target_pkd_all = []
        pkd_pred = []
        all_rmsd = []
        for idx, features, geometry, target_pkd in progress:
            with torch.no_grad():
                features = features.to(self.device)
                geometry = geometry.to(self.device)
                masks = masks.to(self.device)

                out1 = model(geometry, features, masks).to(self.device)
                target_pkd = target_pkd.to(self.device)
                target_pkd_all.append(target_pkd)
                pkd_pred.append(out1.cpu())

                loss_rmsd_pkd = loss_cl(out1, target_pkd).float()
                self.writer.add_scalar("test_loss", loss_rmsd_pkd.item(), epoch)
                all_rmsd.append(loss_rmsd_pkd.item())
        return torch.cat(target_pkd_all), torch.cat(pkd_pred), sum(all_rmsd) / len(all_rmsd)

            
         

    def train_epochs(self):
        featuriser = Pdb_Dataset(self.cfg, vocab=self.vocab)
        files_refined = os.listdir(self.protein_dir)
        idx_folds = pickle.load( open(os.path.join(self.idx_file, self.name_file_folds), "rb" ) )
        split_no = self.fold_number
        test_idx = []
        py3nvml.nvmlInit()
 
        train_id, test_id = idx_folds[split_no]
        train_data = train_id
        test_data = test_id
        with open(os.path.join(self.idx_file, 'test_idx_' + str(split_no)), 'wb') as fp:
            pickle.dump(test_data, fp)

        feat_train = [featuriser[data] for data in train_data]
        feat_test = [featuriser[data] for data in test_data]

        loader_train = DataLoader(
            feat_train, batch_size=self.BATCH_SIZE, num_workers= self.NUM_WORKERS, shuffle=True
        )

        loader_test = DataLoader(
            feat_test, batch_size=self.BATCH_SIZE, num_workers=self.NUM_WORKERS, shuffle=False
        )


        loss_cl = Loss()
        opt = Adam(model.parameters(),
                    lr=config["model_params"]["learning_rate"])
        scheduler = ExponentialLR(opt, gamma=0.95)

        print("Training model...")
        losses_to_write_train = []
        for i in range(N_EPOCHS):

            print("Epoch {}/{}...".format(i + 1, self.N_EPOCHS))
            epoch = i + 1
            target_pkd_all, pkd_pred, loss = training_loop(
                loader_train, self.Encoder, loss_cl, opt, epoch
            )
            print("pkd_pred", pkd_pred)
            losses_to_write_train.append(loss)

            if i == self.N_EPOCHS - 1:
              
                np.save(
                    os.path.join(self.PKD_PATH, "pkd_pred_train_{}.npy".format(str(i))),
                    arr=pkd_pred.detach().cpu().clone().numpy(),
                )
            scheduler.step()
        losses_to_write_train = np.asarray(losses_to_write_train, dtype=np.float32)
        # save losses for the train
        np.savetxt(
            os.path.join(self.PATH_LOSS, "losses_train_2016.out"),
            losses_to_write_train,
            delimiter=",",
        )
        # save true values of training target
        savetxt(
            os.path.join(self.PKD_PATH, "target_pkd_all_train.csv"),
            target_pkd_all.detach().cpu().clone().numpy(),
        )

        np.save(
            os.path.join(self.PKD_PATH, "target_pkd_all_train"),
            arr=target_pkd_all.detach().cpu().clone().numpy(),
        )

        print("Evaluating model...")
        target_pkd_all_test, pkd_pred_test, loss_test_to_write = eval_loop(
            loader_test, model, epoch
        )
        print("pkd_pred", pkd_pred_test)
        loss_test_to_write = np.asarray(loss_test_to_write, dtype=np.float32)
        loss_test_to_write = np.asarray([loss_test_to_write])
        np.savetxt(
            os.path.join(self.PATH_LOSS, "losses_test_2016.out"),
            loss_test_to_write,
            delimiter=",",
        )

        os.makedirs(self.PKD_PATH, exist_ok=True)


        np.save(
            os.path.join(self.PKD_PATH, "target_pkd_all_test"),
            arr=target_pkd_all_test.detach().cpu().clone().numpy(),
        )
        np.save(
            os.path.join(self.PKD_PATH, "pkd_pred_test"),
            arr=pkd_pred_test.detach().cpu().clone().numpy(),
        )

        with open(os.path.join(self.PKD_PATH, "split_pdbids.pt"), "wb") as handle:
            pickle.dump(split_pdbids, handle)

        self.utils.plot_statistics(
            self.PKD_PATH,
            self.PATH_PLOTS,
            self.N_EPOCHS,
            self.name_plot,
            "train",
            losses_to_write_train[-1],
            loss_test_to_write[0],
        )

        self.utils.plot_statistics(
            self.PKD_PATH,
            self.PATH_PLOTS,
            self.N_EPOCHS,
            self.name_plot,
            "test",
            losses_to_write_train[-1],
            loss_test_to_write[0],
        )

        self.utils.plot_losses(
            self.PATH_LOSS, self.PATH_PLOTS, self.N_EPOCHS, self.name_plot
        )


       