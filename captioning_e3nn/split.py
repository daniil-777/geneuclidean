import itertools as IT
import json
import os
import pickle
import time
from distutils.dir_util import copy_tree
from functools import partial
from multiprocessing import Pool
from shutil import copyfile
import itertools
import _pickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial.distance as dist
import torch
from matplotlib import pyplot as plt
from numpy import mean, std
# from openbabel import openbabel
from scipy import spatial as spatial
from scipy.stats import pearsonr
import argparse
import sys
import config
from py3nvml import py3nvml

import json
import os
import pickle

from sklearn.model_selection import KFold
import numpy as np

from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.model_selection import KFold
from sklearn.cluster import MiniBatchKMeans


from build_vocab import Vocabulary


number_atoms = 22

FP_SIZE = 1024 

class Splitter:
    # def __init__(self, path_pocket: str, path_ligand: str):
    def __init__(self, cfg):
        self.cfg = cfg
        self.name_file_folds = cfg['splitting']['file_folds']
        self.num_epochs = cfg['model_params']['num_epochs']
        self.batch_size = cfg['model_params']['batch_size']
        self.learning_rate = cfg['model_params']['learning_rate']
        self.num_workers = cfg['model_params']['num_workers']
        
        self.path_root = cfg['preprocessing']['path_root']
        self.init_refined = self.path_root + "/data/new_refined/"
        # training params
        self.protein_dir = cfg['training_params']['image_dir']
        self.files_refined = os.listdir(self.protein_dir)
        self.files_refined.sort()
        self.caption_path = cfg['training_params']['caption_path']
        self.log_step = cfg['training_params']['log_step']
        self.save_step = cfg['training_params']['save_step']
        self.vocab_path = cfg['preprocessing']['vocab_path']
        self.n_splits = cfg['training_params']['n_splits']
        self.loss_best = np.inf
        self.n_samples = len(self.files_refined) - 3
        #output files
        self.savedir = cfg['output_parameters']['savedir']
        self.tesnorboard_path = self.savedir
        self.model_path = os.path.join(self.savedir, "models")
        self.log_path = os.path.join(self.savedir, "logs")
        self.idx_file = os.path.join(self.log_path, "idxs")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir_smiles = os.path.join(self.savedir, "statistics")
        self.file_prot_chain = "../data/bc-95.txt"
        self.random_state = 1337
        if not os.path.exists(self.save_dir_smiles):
            os.makedirs(self.save_dir_smiles)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        if not os.path.exists(self.idx_file):
            os.makedirs(self.idx_file)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

       
    def _get_random_split(self):
        data_ids = np.array([i for i in range(self.n_samples)])
   
        #cross validation
        kf = KFold(n_splits=5, shuffle=True, random_state=2)
        my_list = list(kf.split(data_ids))
        with open(os.path.join(self.idx_file, self.name_file_folds), 'wb') as fp:
            pickle.dump(my_list, fp)

    def _ligand_scaffold_split(self):
        """
        Ligand-based scaffold split using Morgan fingerprints
        and k-means clustering.
        """
        km = MiniBatchKMeans(n_clusters=self.n_splits, random_state=self.random_state)
        feat = np.zeros((self.n_samples, 1024), dtype=np.uint8)

        for idx in range(self.n_samples):
            smile = self._get_caption(idx)
            mol = Chem.MolFromSmiles(smile)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=FP_SIZE)
            arr = np.zeros((1,), dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            feat[idx] = arr.copy()

        labels = km.fit_predict(feat)
        splits = []
        for split_no in range(self.n_splits):
            indices_train = np.where(labels != split_no)[0]
            indices_test = np.where(labels == split_no)[0]
            splits.append((indices_train, indices_test))
        splits = np.asarray(splits)
        with open(os.path.join(self.idx_file, self.name_file_folds), 'wb') as fp:
            pickle.dump(splits, fp)
        return splits

    def chain_split(self):
        self.files_refined = self.files_refined[:-3]
        with open(self.file_prot_chain, 'r') as file: 
            lines = file.read().splitlines()
        words_all = []
        for line in lines:
            words_line = [word.split("_")[0].lower() for word in line.split()]
            words_all.append(words_line)
            
        word_dict = {}
        for idx, line in enumerate(words_all):
            for word in line:
                word_dict[word] = idx
                

        refined_orig_dict = {key: value for value, key in enumerate(self.files_refined)}
        refined_dict_chain = {}
        for id in self.files_refined:
            if id in word_dict.keys():
                refined_dict_chain[id] = word_dict[id]
                
        inverted_ref_chain = {}
        for key, value in refined_dict_chain.items():
            inverted_ref_chain.setdefault(value, list()).append(key)
            
        values_unique = list(set(refined_dict_chain.values())) #idx of line (subgroup) from prot chain
        kf = KFold(n_splits=5, shuffle=True, random_state=2)
        my_list = list(kf.split(values_unique))

        split_refined_all = []
        for split in my_list: # folds
            split_refined = [] 
            for sub_split in split: #train/test
                idx_first = [values_unique[idx] for idx in sub_split] #train/test
                names_prot = list(itertools.chain.from_iterable([inverted_ref_chain[ind] for ind in idx_first])) #names of refined
                id_orig = [refined_orig_dict[name] for name in names_prot] #id of prot from original list
                split_refined.append(id_orig)
            split_refined_all.append(split_refined)

        with open(os.path.join(self.idx_file, self.name_file_folds), 'wb') as fp:
            pickle.dump(split_refined_all, fp)
    

    def _get_caption(self, id):
        """get caption as a row of a smile by id
        """

        protein_name = self.files_refined[id]
        # print("current protein", protein_name)
        path_to_smile = os.path.join(
            self.init_refined, protein_name, protein_name + "_ligand.smi"
        )
        with open(path_to_smile, "r") as file:
            caption = file.read()
        return caption
        


def main():
    parser = argparse.ArgumentParser(
    description='Get Splits File'
)
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('type_fold', type=str, help='type_fold')

    args = parser.parse_args()
                         
    cfg = config.load_config(args.config, 'configurations/config_lab/default.yaml')
    type_fold = args.type_fold
    args = parser.parse_args()


    cfg = config.load_config(args.config, 'configurations/config_local/default.yaml')
    
    splitter = Splitter(cfg)
    
    if(type_fold == 'random'):
        splitter._get_random_split()
    elif(type_fold == 'morgan'):
        splitter._ligand_scaffold_split()
    elif(type_fold == 'chain'):
        splitter.chain_split()


    # if(cfg['splitting']['split'] == 'random'):
    #     splitter._get_random_split()
    # elif(cfg['splitting']['split'] == 'morgan'):
    #     splitter._ligand_scaffold_split()
    # elif(cfg['splitting']['split'] == 'chain'):
    #     splitter.chain_split()

if __name__ == "__main__":
    main()