import multiprocessing

import numpy as np
from numpy import savetxt
import torch
# from torchsummary import summary
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


import argparse
import sys
import utils.config as config
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
from utils.build_vocab import Vocabulary
from datasets.data_loader import get_loader, Pdb_Dataset, collate_fn, collate_fn_masks
from datasets.data_loader_feature import Pdb_Dataset_Feature
from sampling.sampler import Sampler
from training.utils import save_checkpoint
from torch.utils import model_zoo


class Trainer_Fold():
    def __init__(self, cfg, split_no):
        # model params
        self.cfg = cfg
        self.split_no = split_no
        self.original_stdout = sys.stdout
        #folds data
        self.name_file_folds = cfg['splitting']['file_folds']
        self.fold_number = cfg['splitting']['id_fold']
        self.model_name = cfg['model_params']['model_name']
        self.num_epochs = cfg['model_params']['num_epochs']
        self.batch_size = cfg['model_params']['batch_size']
        self.learning_rate = cfg['model_params']['learning_rate']
        self.num_workers = cfg['model_params']['num_workers']

        # training params
        self.protein_dir = cfg['training_params']['image_dir']
        self.caption_path = cfg['training_params']['caption_path']
        self.log_step = cfg['training_params']['log_step']
        self.save_step = cfg['training_params']['save_step']
        self.vocab_path = cfg['preprocessing']['vocab_path']
        self.n_splits = cfg['training_params']['n_splits']
        self.loss_best = np.inf

        #output files
        # self.savedir = cfg['output_parameters']['savedir']
        self.savedir = os.path.join(cfg['output_parameters']['savedir'], self.model_name)
        self.tesnorboard_path_train = os.path.join(self.savedir, "logs", "tensorboard_" + self.model_name, 'train')
        self.tesnorboard_path_eval = os.path.join(self.savedir, "logs", "tensorboard_" + self.model_name, 'eval')
        self.model_path = os.path.join(self.savedir, "models")
        self.log_path = os.path.join(self.savedir, "logs")
        self.idx_file = os.path.join(self.log_path, "idxs")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir_smiles = os.path.join(self.savedir, "statistics")
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.idx_file, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.save_dir_smiles, exist_ok=True)
        os.makedirs(self.tesnorboard_path_train, exist_ok=True)
        os.makedirs(self.tesnorboard_path_eval, exist_ok=True)
        # if not os.path.exists(self.log_path):
        #     os.makedirs(self.log_path)
        # if not os.path.exists(self.idx_file):
        #     os.makedirs(self.idx_file)
        # if not os.path.exists(self.model_path):
        #     os.makedirs(self.model_path)

        # if not os.path.exists(self.save_dir_smiles):
        #     os.makedirs(self.save_dir_smiles)
        #log files 
        self.test_idx_file = open(os.path.join(self.idx_file, "test_idx.txt"), "w")
        self.log_file = open(os.path.join(self.log_path, "log.txt"), "w")
        self.log_file_tensor = open(os.path.join(self.log_path, "log_tensor.txt"), "w")
        self.writer_train = SummaryWriter(self.tesnorboard_path_train)
        self.writer_eval = SummaryWriter(self.tesnorboard_path_eval)
        
        self.Encoder, self.Decoder = config.get_model(cfg, device=self.device)
        # self.input = config.get_shape_input(self.cfg)
        print(self.Encoder)
        print(self.Decoder)
        with open(os.path.join(self.log_path, "model.txt"), 'w') as f:
            sys.stdout = f # Change the standard output to the file we created.
            # print(summary(self.Encoder, self.input))
            # print(summary(self.Decoder))
            print(self.Encoder)
            print(self.Decoder)
            sys.stdout = self.original_stdout
           
        # print(model)
        # self.name_file_stat = cfg["sampling_params"]["name_all_stat"]
        # self.file_statistics = open(os.path.join(self.save_dir_smiles, self.name_file_stat), "w")
        #the file of the whole stat
        # self.file_statistics.write("name,fold,type_fold, orig_smile, gen_smile, gen_NP, gen_logP,gen_sa,gen_qed,gen_weight,gen_similarity, orig_NP, orig_logP, orig_sa, orig_qed, orig_weight, frequency, sampling" + "\n")
        # self.file_statistics.flush()

        #print all params
        nparameters_enc = sum(p.numel() for p in self.Encoder.parameters())
        nparameters_dec = sum(p.numel() for p in self.Decoder.parameters())
        print('Total number of parameters: %d' % (nparameters_enc + nparameters_dec))

        with open(os.path.join(self.log_path, "model.txt"), 'w') as f:
            f.write('Total number of parameters: %d' % (nparameters_enc + nparameters_dec))

        with open(self.vocab_path, "rb") as f:
            self.vocab = pickle.load(f)
        self.criterion = nn.CrossEntropyLoss()
        self.model_name = 'e3nn'
        self.checkpoint_path = os.path.join(self.savedir, 'checkpoints')
        os.makedirs(self.checkpoint_path, exist_ok=True)
        self.checkpoint_path_training =  os.path.join(self.savedir, 'checkpoints', str(self.split_no) + '_training.pkl')

        self.eval_check_path = os.path.join(self.savedir, 'checkpoints', 'eval.txt')
        if not os.path.exists(self.eval_check_path):
            with open(self.eval_check_path, 'w') as file:
                file.write('0')
        
        #loading checkpoint
        if (os.path.exists(self.checkpoint_path_training)):
            checkpoint = torch.load(self.checkpoint_path_training)
            
            print("loading model...")
            self.start_epoch = checkpoint['start_epoch'] + 1
            self.Encoder, self.Decoder = config.get_model(cfg, device=self.device)
      
            self.Encoder.load_state_dict(checkpoint['encoder'])
            self.Decoder.load_state_dict(checkpoint['decoder'])
            self.encoder_best, self.decoder_best = self.Encoder, self.Decoder
            self.caption_optimizer = checkpoint['caption_optimizer']
            # self.scheduler = ExponentialLR(self.caption_optimizer, gamma=0.95)
            # self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.split_no = checkpoint['split_no']
         
        else:
            print("initialising model...")
            self.start_epoch = 0
            self.Encoder, self.Decoder = config.get_model(cfg, device=self.device)
            self.encoder_best, self.decoder_best = self.Encoder, self.Decoder
            caption_params = list(self.Encoder.parameters()) + list(self.Decoder.parameters())
            self.caption_optimizer = torch.optim.Adam(caption_params, lr = self.learning_rate)
            # self.scheduler = ExponentialLR(self.caption_optimizer, gamma=0.95)
            self.split_no = self.fold_number


    def train_loop_mask(self, loader, caption_optimizer, split_no, epoch, total_step):
        self.Encoder.train()
        self.Decoder.train()
        progress = tqdm(loader)
        for i, (features, geometry, masks, captions, lengths) in enumerate(progress):
            # Set mini-batch dataset

            features = features.to(self.device)
            geometry = geometry.to(self.device)
            captions = captions.to(self.device)
            masks = masks.to(self.device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            caption_optimizer.zero_grad()
            # Forward, backward and optimize
            feature = self.Encoder(features, geometry, masks)
            outputs = self.Decoder(feature, captions, lengths)

            loss = self.criterion(outputs, targets)
            # scheduler.step(loss)

            self.Decoder.zero_grad()
            self.Encoder.zero_grad()
            loss.backward()
            caption_optimizer.step()  #!!! figure out whether we should leave that 
     
            name = "training_loss_" + str(split_no + 1)
            self.writer_train.add_scalar(name, loss.item(), epoch)

            # writer.add_scalar("training_loss", loss.item(), epoch)
            self.log_file_tensor.write(str(loss.item()) + "\n")
            self.log_file_tensor.flush()
            # handle = py3nvml.nvmlDeviceGetHandleByIndex(0)
            # fb_mem_info = py3nvml.nvmlDeviceGetMemoryInfo(handle)
            # mem = fb_mem_info.used >> 20
            mem = 0
            # print('GPU memory usage: ', mem)
            self.writer_train.add_scalar('val/gpu_memory', mem, epoch)
            # Print log info
            if i % self.log_step == 0:
                result = "Split [{}], Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}".format(
                    split_no, epoch, self.num_epochs, i, total_step, loss.item(), np.exp(loss.item())
                )
                # print(result)
                self.log_file.write(result + "\n")
                self.log_file.flush()

            progress.set_postfix({'epoch': epoch,
                                  'loss': loss.item(),
                                  'Perplexity': np.exp(loss.item()),
                                  'mem': mem})
            if (self.loss_best - loss > 0):
                # print("The best loss " + str(loss.item()) + "; Split-{}-Epoch-{}-Iteration-{}_best.ckpt".format(split_no, epoch + 1, i + 1))
                self.log_file.write("The best loss " + str(loss.item()) + "; Split-{}-Epoch-{}-Iteration-{}_best.ckpt".format(split_no, epoch + 1, i + 1) + "\n")
                self.enoder_best = self.Encoder
                self.decoder_best = self.Decoder
                self.encoder_best_name =  os.path.join(
                        self.model_path, "encoder_best_" + str(split_no) + ".ckpt"
                    )
                self.decoder_best_name =  os.path.join(
                        self.model_path, "decoder_best_" + str(split_no) + ".ckpt")
                torch.save(
                    self.Encoder.state_dict(),
                    self.encoder_best_name,
                )
                torch.save(
                    self.Decoder.state_dict(),
                    self.decoder_best_name,
                )
                self.loss_best = loss
        self.log_file_tensor.write("\n")
        self.log_file_tensor.flush()


    def eval_loop(self, loader, epoch):
        """
        Evaluation loop using `model` and data from `loader`.
        """
        self.Encoder.eval()
        self.Decoder.eval()
        progress = tqdm(loader)
        # print("Evaluation starts...") 
        for step, (features, geometry, masks, captions, lengths) in enumerate(progress):
            with torch.no_grad():
                features = features.to(self.device)
                geometry = geometry.to(self.device)
                captions = captions.to(self.device)
                masks = masks.to(self.device)
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
                # Forward, backward and optimize
                feature = self.Encoder(features, geometry, masks)
                outputs = self.Decoder(feature, captions, lengths)
                loss = self.criterion(outputs, targets)
                name = "eval_loss_" + str(self.split_no + 1)
                self.writer_eval.add_scalar(name, loss.item(), epoch)
                # self.writer.add_scalar("test_loss", loss.item(), step)
                # handle = py3nvml.nvmlDeviceGetHandleByIndex(0)
                # fb_mem_info = py3nvml.nvmlDeviceGetMemoryInfo(handle)
                # mem = fb_mem_info.used >> 20
                mem = 20
                progress.set_postfix({'epoch': epoch,
                                      'l_ev': loss.item(),
                                      'Perplexity': np.exp(loss.item()),
                                      'mem': mem})
        # with open(self.eval_check_path, 'w') as file:
        #     file.write('1')
      

    def train_epochs(self, Feature_loader):
        featuriser = Pdb_Dataset_Feature(self.cfg, Feature_loader)
        # data_ids, data_names = utils._get_refined_data()
        files_refined = os.listdir(self.protein_dir)
        # data_ids = np.array([i for i in range(len(files_refined) - 3)])

        #cross validation
        
        idx_folds = pickle.load( open(os.path.join(self.idx_file, self.name_file_folds), "rb" ) )
        test_idx = []
        # output memory usage
        # py3nvml.nvmlInit()
        # sampling = self.cfg['sampling_params']['sampling']
        # sampler = Sampler(self.cfg, sampling)
 
        train_id, test_id = idx_folds[self.split_no]
        train_data = train_id
        test_data = test_id
        with open(os.path.join(self.idx_file, 'test_idx_' + str(self.split_no)), 'wb') as fp:
            pickle.dump(test_data, fp)
        
        test_idx.append(test_data)
        self.test_idx_file.write(str(test_data) + "\n")
        self.test_idx_file.flush()

        feat_train = [featuriser[data] for data in train_data]
        feat_test = [featuriser[data] for data in test_data]
        
        loader_train = DataLoader(feat_train, batch_size=self.batch_size,
                                    shuffle=True,
                                    num_workers=self.num_workers,
                                    collate_fn=collate_fn_masks,)

        loader_test = DataLoader(feat_test, batch_size=self.batch_size,
                                    shuffle=False,
                                    num_workers=self.num_workers,
                                    collate_fn=collate_fn_masks,)
        # loader_train = config.get_loader(cfg, feat_train, batch_size, num_workers,)

        total_step = len(loader_train)
        print("total_step", total_step)

        # params_encoder = filter(lambda p: p.requires_grad, encoder.parameters())        
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(caption_optimizer, 'min')
        for epoch in range(self.start_epoch, self.num_epochs):
            # config.get_train_loop(cfg, loader_train, encoder, decoder,caption_optimizer, split_no, epoch, total_step)
            #if add masks everywhere call just train_loop
            self.train_loop_mask(loader_train, self.caption_optimizer, self.split_no, epoch, total_step)
            # self.scheduler.step()
            self.eval_loop(loader_test, epoch)
            save_checkpoint(self.checkpoint_path_training, epoch, self.Encoder, self.Decoder,
                            self.encoder_best, self.decoder_best, self.caption_optimizer, self.split_no)
            # save_checkpoint(self.checkpoint_path_training, epoch, self.Encoder, self.Decoder,
                            # self.encoder_best, self.decoder_best, self.caption_optimizer, self.scheduler, self.split_no)

            self.encoder_name =  os.path.join(
                        self.model_path, "encoder-{}-{}.ckpt".format(self.split_no, epoch + 1)
                    )
            self.decoder_name =  os.path.join(
                        self.model_path, "decoder-{}-{}.ckpt".format(self.split_no, epoch + 1)
                    )
            torch.save(
                    self.Encoder.state_dict(),
                    self.encoder_name,
                )
            torch.save(
                    self.Decoder.state_dict(),
                    self.decoder_name,
                )
        # check_eval = open(self.eval_check_path).readlines()[0]
        # print("check_eval", check_eval)
        # if check_eval == '0':
        #     self.eval_loop(loader_test)
        # print("Evaluation is done already!")

        #run sampling for the test indxs
            
        # sampler.analysis_cluster(split_no, self.encoder_best_name, self.decoder_best_name)


       



