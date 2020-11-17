import argparse
import csv
import json
import multiprocessing
import os
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
# from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import torch
import torch.nn as nn
from numpy import savetxt
from rdkit import Chem
from sklearn.model_selection import KFold
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

import utils.config as config
from src.datasets.data_loader import Pdb_Dataset
from src.evaluation.Contrib.statistics import (analysis_to_csv,
                                               analysis_to_csv_test)
from src.sampling.sampler import Sampler
from src.training.utils import save_checkpoint_sampling
from src.utils.build_vocab import Vocabulary
from src.utils.checkpoint import Checkpoint_Eval


class Evaluator():
    def __init__(self, cfg, sampling, type_fold, epochs_array, Feature_Loader):
        self.cfg = cfg
        self.Feature_Loader = Feature_Loader
        self.type_fold = type_fold
        self.path_root = cfg['preprocessing']['path_root']
        # self.init_refined = self.path_root + "/data/new_refined/"
        self.init_refined = cfg['training_params']['image_dir']
        self.files_refined = os.listdir(self.init_refined)
        self.files_refined = [file for file in self.files_refined if file[0].isdigit()]
        self.files_refined.sort()
        
        self.attention = self.cfg['training_params']['mode']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.sampling = sampling
        self.epochs_array = epochs_array
        self.num_epochs = len(self.epochs_array)
        # print("num of epoches", self.num_epochs)
        self.model_encoder =  cfg['model']['encoder']
        # print(self.model_encoder)
        self.model_decoder =  cfg['model']['decoder']
        self.sampling_data = cfg['sampling_params']['sampling_data']
        self.protein_dir = cfg["training_params"]["image_dir"]
        if not self.sampling.startswith('beam'):
            self.number_smiles = cfg["sampling_params"]["number_smiles"]
        else:
            self.number_smiles = int(self.sampling.split("_")[1])
        # if (self.sampling == "max"):
        #     self.number_smiles = 1
        self.time_waiting = cfg["sampling_params"]["time_waiting"]
        # model params
        self.model_name = cfg['model_params']['model_name']
        # self.num_epochs = cfg['model_params']['num_epochs']
        self.batch_size = cfg['model_params']['batch_size']
        self.learning_rate = cfg['model_params']['learning_rate']
        self.num_workers = cfg['model_params']['num_workers']

        # training params
        self.protein_dir = cfg['training_params']['image_dir']
        self.caption_path = cfg['training_params']['caption_path']
        self.log_step = cfg['training_params']['log_step']
        self.save_step = cfg['training_params']['save_step']
        self.vocab_path = cfg['preprocessing']['vocab_path']
        #output files
        self.savedir = os.path.join(cfg['output_parameters']['savedir'], self.model_name)
        self.save_dir_smiles = os.path.join(self.savedir, "statistics")
        self.tesnorboard_path = self.savedir
        self.log_path = os.path.join(self.savedir, "logs")
        self.idx_file = os.path.join(self.log_path, "idxs")
        self.save_dir_encodings = os.path.join(self.savedir, "encodings", self.model_name)
        #sampling params
        os.makedirs(self.save_dir_smiles, exist_ok=True)
        os.makedirs(self.save_dir_encodings, exist_ok=True)
        os.makedirs(os.path.join(self.log_path, "checkpoints"), exist_ok=True)
        self.path_data = os.path.join(cfg["output_parameters"]["savedir"], cfg["model_params"]["model_name"], "statistics")
        with open(self.vocab_path, "rb") as f:
            self.vocab = pickle.load(f)
        self.dataset = Pdb_Dataset(cfg, self.vocab)
        self.path_smiles_train = os.path.join(self.log_path, "checkpoints", "smiles_train")
        if not os.path.exists(self.path_smiles_train):
            self.smiles_train = self._get_train_smiles()
        else:
            with open(self.path_smiles_train, 'rb') as smiles:
                self.smiles_train = pickle.load(smiles)
        self.path_vis = os.path.join(cfg["output_parameters"]["savedir"], self.model_name, 'results_' + self.model_name)
        self.path_plot = os.path.join(self.path_vis, self.type_fold)
        os.makedirs(self.path_plot, exist_ok=True)
        self._n_folds = 5
        self.path_checkpoint_evaluator = os.path.join(self.savedir, "checkpoints", "checkpoint_evaluator.csv")
        self.checkpoint_evaluation = Checkpoint_Eval(self.path_checkpoint_evaluator, self.type_fold, self.sampling)
        self.start_rec_fold, self.start_rec_epoch, self.start_eval_fold, self.start_eval_epoch = self.checkpoint_evaluation._get_data()

        self.path_novel = os.path.join(self.log_path, "checkpoints", "novel.npy")
        self.path_valid = os.path.join(self.log_path, "checkpoints", "valid.npy")
        self.path_unique = os.path.join(self.log_path, "checkpoints", "unique.npy")
        if not os.path.isfile(self.path_novel):
            self.valid = np.zeros((self._n_folds, self.num_epochs))
            self.unique = np.zeros((self._n_folds, self.num_epochs))
            self.novel = np.zeros((self._n_folds, self.num_epochs))
            np.save(self.path_novel, self.novel[0])
            np.save(self.path_valid, self.valid[0])
            np.save(self.path_unique, self.unique[0])
        else:
            self.novel = np.load(self.path_novel, allow_pickle=True)
            self.valid = np.load(self.path_valid, allow_pickle=True)
            self.unique = np.load(self.path_unique, allow_pickle=True)
            # print("shape of unique array first- ", self.unique.shape)

    def run_evaluation(self):
        self.record_all_mol()
        self.evaluate_all_mol()
        
    def record_all_mol(self):
        for idx_fold in range(self.start_rec_fold, self._n_folds):
            for epoch in range(self.start_rec_epoch, self.num_epochs):
                epoch_absolute = self.epochs_array[epoch]
                encoder_path = os.path.join(self.savedir,  "models", "encoder-" + str(idx_fold) + "-" + str(epoch + 1) + '-' + str(self.type_fold) + '.ckpt') 
                decoder_path = os.path.join(self.savedir, "models", "decoder-" + str(idx_fold) + "-" + str(epoch + 1) + '-' + str(self.type_fold) + '.ckpt')
                # print("encoder_path!!", encoder_path)
                sampler = Sampler(self.cfg, self.sampling, self.Feature_Loader)
                sampler.analysis_cluster(idx_fold, epoch_absolute, self.type_fold, encoder_path, decoder_path)
                self.checkpoint_evaluation.write_record_checkpoint(idx_fold + 1, epoch + 1)


    def evaluate_all_mol(self):
        for idx_fold in range(self.start_eval_fold, self._n_folds):
            for epoch in range(self.start_eval_epoch, self.num_epochs):
                epoch_absolute = self.epochs_array[epoch]
                self.name_file_stat = self.sampling + "_" + str(self.type_fold) + "_" + str(idx_fold) + ".csv"
                file_mols = pd.read_csv(os.path.join(self.save_dir_smiles, self.name_file_stat))
                # print("file_mols, - ", file_mols)
                mol = file_mols.loc[file_mols['epoch_no'] == str(epoch_absolute), 'gen_smile'].to_list()
                number_mols = len(mol) 
                # print("mol!!, ", mol)
                # Compute unique molecules
                # print("shape of unique array - ", self.unique.shape)
                self.unique[idx_fold, epoch] = len(set(mol)) / (number_mols + 1)
                # Remove duplicates
                mol = np.array(list(set(mol)))
                number_mols = mol.shape[0] 
                # Check validity and remove non-valid molecules
                to_delete = []
                for k, m in enumerate(mol):
                    if not self.check_valid(m):
                        to_delete.append(k)
                valid_mol = np.delete(mol, to_delete)
                self.valid[idx_fold, epoch] = len(valid_mol) / (number_mols + 1)


                # Compute molecules unequal to training data
                if valid_mol.size != 0:
                    print("not equal to 0!")
                    new_m = self.check_with_training_data(list(valid_mol), idx_fold)
                    self.novel[idx_fold, epoch] = len(new_m) / number_mols

                #save arrays of novel/valid/unique
                np.save(self.path_novel, self.novel)
                np.save(self.path_valid, self.valid)
                np.save(self.path_unique, self.unique)
                self.checkpoint_evaluation.write_eval_checkpoint(idx_fold + 1, epoch + 1)

        # Get percentage
        self.unique *= 100
        self.novel *= 100
        self.valid *= 100

        # Get mean values
        mean_unique = np.mean(self.unique, axis=0)
        mean_valid = np.mean(self.valid, axis=0)
        mean_novel = np.mean(self.novel, axis=0)

        # Get standard deviation
        std_unique = np.std(self.unique, axis=0)
        std_valid = np.std(self.valid, axis=0)
        std_novel = np.std(self.novel, axis=0)

        # PLot
        plt.figure(1)
        array_epoches = np.asarray(self.epochs_array)
        # print("array_epoches, - ", array_epoches)
        plt.errorbar(array_epoches, mean_unique, yerr=std_unique, capsize=3, label='unique')
        plt.errorbar(array_epoches, mean_valid, yerr=std_valid, capsize=3,
                     label='valid & unique')
        plt.errorbar(array_epoches, mean_novel, yerr=std_novel, capsize=3,
                     label='novel, valid & unique', linestyle=':')


        # plt.errorbar(np.arange(1, self.num_epochs + 1), mean_unique, yerr=std_unique, capsize=3, label='unique')
        # plt.errorbar(np.arange(1, self.num_epochs + 1), mean_valid, yerr=std_valid, capsize=3,
        #              label='valid & unique')
        # plt.errorbar(np.arange(1, self.num_epochs + 1), mean_novel, yerr=std_novel, capsize=3,
        #              label='novel, valid & unique', linestyle=':')
      
        plt.yticks(np.arange(0, 110, step=10))
        plt.legend(loc=3)
        plt.ylim(0, 105)
        plt.title('SMILES at ' + str(self.sampling) + ', ' + str(self.type_fold))
        plt.ylabel('% SMILES')
        plt.xlabel('Epoch')
        path_save = os.path.join(self.path_plot, self.sampling + '_' + 'novel_valid_unique_molecules.png')
        plt.savefig(path_save)
    
        # data = np.vstack((mean_unique, std_unique, mean_valid, std_valid, mean_novel, std_novel))
        # pd.DataFrame(data).to_csv(self._experiment_name + '/molecules/' + self._experiment_name + '_data.csv')

        # # Create output for last epoch
        # data = np.vstack((unique[:,self._epochs-1],valid[:,self._epochs-1],novel[:,self._epochs-1]))
        # pd.DataFrame(data).to_csv(self._experiment_name + '/molecules/' + self._experiment_name + 'final_epoch_data.csv')
        #plt.show()
        plt.close()

    def check_with_training_data(self, mol, id_fold):
        '''Remove molecules that are within the training set and return number
        :return mol:    SMILES not contained in the training
        '''
        to_delete = []
        can_mol = []

        for i, m in enumerate(mol):
            if m in self.smiles_train[id_fold]:
                to_delete.append(i)
        mol = np.delete(mol, to_delete)
        return mol

    def check_valid(self, smile):
        m = Chem.MolFromSmiles(smile)
        if m is None or smile == '' or smile.isspace() == True:
            return False
        else:
            return True

    def _get_train_smiles(self):
        print("smiles training start...")
        smiles_train = []
        for id_split in range(5):
            smiles_split = []
            # self.file_folds = os.path.join(self.idx_file, "test_idx_" + self.type_fold + "_" + str(id_split))
            self.file_folds = os.path.join(self.idx_file, self.type_fold)
            idx_all = [i for i in range(len(self.files_refined))]
            with (open(self.file_folds, "rb")) as openfile:
                idx_folds = pickle.load(openfile)
                _, idx_test = idx_folds[id_split]
            #take indx of proteins in the training set
            idx_proteins_train = np.setdiff1d(idx_all, idx_test)
            for pid in idx_proteins_train:
                smile = self._get_caption(pid)
                smiles_split.append(smile)
            smiles_train.append(smiles_split)
        with open(self.path_smiles_train, 'wb') as fp:
            pickle.dump(smiles_train, fp)
        return smiles_train


    def _get_caption(self, id):
        """get caption as a row of a smile by id
        """

        protein_name = self.files_refined[id]
        path_to_smile = os.path.join(
            self.init_refined, protein_name, protein_name + "_ligand.smi"
        )
        with open(path_to_smile, "r") as file:
            caption = file.read()
        return caption

