import multiprocessing

import numpy as np
from numpy import savetxt
import torch

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
# from torch.utils.tensorboard import SummaryWriter


import argparse
import sys
import config
from rdkit import Chem
import json
import os
import csv
import pickle
import time 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
import numpy as np

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from build_vocab import Vocabulary
from data_loader import Pdb_Dataset
from Contrib.statistics import analysis_to_csv, analysis_to_csv_test


class Sampler():
    def __init__(self, cfg, sampling):
        # model params
        #sampling params
        # self.idx_fold = idx_fold
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.sampling = sampling
        # self.sampling = cfg['sampling_params']['sampling']

        self.model_encoder =  cfg['model']['encoder']
        print(self.model_encoder)
        self.model_decoder =  cfg['model']['decoder']
        self.sampling_data = cfg['sampling_params']['sampling_data']
        self.protein_dir = cfg["training_params"]["image_dir"]
        self.number_smiles = cfg["sampling_params"]["number_smiles"]
        if (self.sampling == "max"):
            self.number_smiles = 1
        self.time_waiting = cfg["sampling_params"]["time_waiting"]
        self.type_fold = cfg["sampling_params"]["type_fold"]
        # self.file_folds = cfg["sampling_params"]["folds"]
        
        # self.file_folds = os.path.join()
    
        # model params
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
        #output files
        self.savedir = cfg['output_parameters']['savedir']
        self.save_dir_smiles = os.path.join(self.savedir, "statistics")
        self.tesnorboard_path = self.savedir
        self.log_path = os.path.join(self.savedir, "logs")
        self.idx_file = os.path.join(self.log_path, "idxs")
        
        #encoder/decoder path
        # self.encoder_path = os.path.join(self.savedir, "models", cfg['training_params']['encoder_name']) 
        # self.decoder_path = os.path.join(self.savedir, "models", cfg['training_params']['decoder_name'])
        self.save_dir_encodings = os.path.join(self.savedir, "encodings")
        #sampling params
        
        if not os.path.exists(self.save_dir_smiles):
            os.makedirs(self.save_dir_smiles)

        if not os.path.exists(self.save_dir_encodings):
            os.makedirs(self.save_dir_encodings)

        self.file_long_proteins = open(os.path.join(self.save_dir_smiles, "exceptions_long.txt"), "w")
        self.name_all_statistics = cfg['sampling_params']['name_all_stat']
        self.file_all_stat = open(os.path.join(self.save_dir_smiles, self.name_all_statistics), "w")
        # self.file_statistics = file_statistics
    
        # self.file_statistics = open(os.path.join(self.save_dir_smiles, self.name_file_stat), "w")
        # #the file of the whole stat
        # self.file_statistics.write("name,fold,type_fold,orig_smile,gen_smile,gen_NP,gen_logP,gen_sa,gen_qed,gen_weight,gen_similarity,orig_NP,orig_logP,orig_sa,orig_qed,orig_weight,frequency,sampling,encoder,decoder" +  "\n")
        # self.file_statistics.flush()

        with open(self.vocab_path, "rb") as f:
            self.vocab = pickle.load(f)

        self.dataset = Pdb_Dataset(cfg, self.vocab)
        # self.encoder_path, self.decoder_path = self._get_model_path()
        # self.encoder, self.decoder = config.eval_model_captioning(cfg, self.encoder_path, self.decoder_path, device = self.device)
    

    def analysis_cluster(self, split_no, encoder_path, decoder_path):
        # encoder, decoder = self._get_model_path(idx_fold)
        self.idx_fold = split_no
        self.name_file_stat = self.sampling + "_" + str(self.idx_fold) + "_" + self.cfg["sampling_params"]["name_all_stat"] 
        self.file_statistics = open(os.path.join(self.save_dir_smiles, self.name_file_stat), "w")
        #the file of the whole stat
        self.file_statistics.write("name,fold,type_fold,orig_smile,gen_smile,gen_NP,gen_logP,gen_sa,gen_qed,gen_weight,gen_similarity,orig_NP,orig_logP,orig_sa,orig_qed,orig_weight,frequency,sampling,encoder,decoder" +  "\n")
        self.file_statistics.flush()
        # self.encoder_path, self.decoder_path = self._get_model_path()
        self.encoder, self.decoder = config.eval_model_captioning(self.cfg, encoder_path, decoder_path, device = self.device)
        self.file_folds = os.path.join(self.idx_file, "test_idx_" + str(self.idx_fold))
        with (open(self.file_folds, "rb")) as openfile:
            idx_proteins = pickle.load(openfile)
        # idx_proteins = [1,2,3,4]
        files_refined = os.listdir(self.protein_dir)
        idx_all = [i for i in range(len(files_refined) - 3)]
        #take indx of proteins in the training set
        if (self.sampling_data == "train"):
            idx_to_generate = np.setdiff1d(idx_all, idx_proteins)
        else:
            idx_to_generate = idx_proteins
        for id_protein in idx_to_generate:
            self.generate_smiles(id_protein)


    def _get_models(self, idx_fold):
        encoder_path, decoder_path = self._get_model_path(idx_fold)
        encoder, decoder = config.eval_model_captioning(cfg, encoder_path, decoder_path, device = self.device)
        return encoder, decoder

    def _get_model_path(self):
        encoder_name = "encoder-" + str(self.idx_fold) + "-1-2.ckpt"
        decoder_name = "decoder-" + str(self.idx_fold) + "-1-2.ckpt"
        encoder_path = os.path.join(self.savedir, "models", encoder_name)
        decoder_path = os.path.join(self.savedir, "models", decoder_name)
        return encoder_path, decoder_path


    def load_pocket(self, id_protein, transform=None):
        print("loading data of a protein", self.dataset._get_name_protein(id_protein))
        features, masks = self.dataset._get_features_complex(id_protein)
        geometry = self.dataset._get_geometry_complex(id_protein)
        features = features.to(self.device).unsqueeze(0)
        geometry = geometry.to(self.device).unsqueeze(0)
        masks = masks.to(self.device).unsqueeze(0)
        return features, geometry, masks


    def generate_encodings(self, idx_):
        #generate features of encoder and writes it to files
        protein_name =  self.dataset._get_name_protein(id)
        features, geometry = self.load_pocket(id)
        # Generate a caption from the image
        feature = self.encoder(features, geometry)
        torch.save(feature, os.path.join(self.save_dir_encodings, protein_name + "_feature_encoding.pt"))

    def printing_smiles(self, sampled_ids, list_smiles_all):
    
        sampled_caption = []
        for word_id in sampled_ids:
            word = self.vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == "<end>":
                break
        sentence = "".join(sampled_caption)
        sentence = sentence[7:-5]
        print(sentence)
        m = Chem.MolFromSmiles(sentence)
        if m is None or sentence == '' or sentence.isspace() == True:
            print('invalid')
            # list_smiles_all.append(sentence)
            return 1
        else:
            print(sentence)
            # smiles.append(sentence)
            list_smiles_all.append(sentence)
            return 1

    def smiles_all_txt(self):
        file_all_smiles = open(os.path.join(self.save_dir_smiles, "all_smiles_lig.txt"), "w")
        files_refined =  os.listdir(self.caption_path)
        files_refined.remove(".DS_Store")
        for protein_name in files_refined:
            init_path_smile =  os.path.join(
                self.caption_path, protein_name, protein_name + "_ligand.smi"
            )
            with open(init_path_smile) as fp: 
                initial_smile = fp.readlines()[0]
                file_all_smiles.write(initial_smile + "\n")
                file_all_smiles.flush()

    def generate_smiles(self, id):
        #original + gen smiles
        print("current id - ", id)
        smiles = []
        protein_name =  self.dataset._get_name_protein(id)
        print("current protein ", protein_name)
        #path of the real smile
        init_path_smile =  os.path.join(
                self.caption_path, protein_name, protein_name + "_ligand.smi"
            )
        
        with open(init_path_smile) as fp: 
            initial_smile = fp.readlines()[0] #write a true initial smile
        smiles.append(initial_smile)
        amount_val_smiles = 0
        
        iter = 0
        start = time.time()
        if (self.sampling != "beam"):
            while (amount_val_smiles < self.number_smiles):
                end = time.time()
                print("time elapsed", end - start)
                if((end - start) > self.time_waiting):
                    #stop generating if we wait for too long till 50 ligands
                    self.file_long_proteins.write(protein_name + "\n") #write a protein with long time of generating
                    self.file_long_proteins.flush()
                    break
                iter += 1
                # Build models
                # Load the trained model parameters            
                # # Prepare features and geometry from pocket
                features, geometry, masks = self.load_pocket(id)

                # Generate a caption from the image
                feature = self.encoder(features, geometry, masks)
                #print("feature", feature)
                
                if (self.sampling == "probabilistic"):
                    sampled_ids = self.decoder.sample_prob(feature)
                elif (self.sampling == "max"):
                    sampled_ids = self.decoder.sample_max(feature)
               
                sampled_ids = ( sampled_ids[0].cpu().numpy() )
                idx =  self.printing_smiles(sampled_ids, smiles)
                amount_val_smiles += idx
           
        
        elif (self.sampling == "beam"):
            features, geometry, masks = self.load_pocket(id)
            feature = self.encoder(features, geometry, masks)
            sampled_ids = self.decoder.sample_beam_search(feature)
            for sentence in sampled_ids:
                iter += 1
                idx =  self.printing_smiles(np.asarray(sentence[1:]), smiles)
                amount_val_smiles += idx
        else:
            raise ValueError("Unknown sampling...")

           # sampled_ids = (
            #   sampled_ids[0].cpu().numpy()
           # )  # (1, max_seq_length) -> (max_seq_length)
            # Convert word_ids to wordsi
           # for sentence in sampled_ids:
           #     idx =  self.printing_smiles(np.asarray(sentence[1:]), smiles)
           #     amount_val_smiles += idx
        
        if (amount_val_smiles > 0):
            # save_dir_analysis = os.path.join(save_dir_smiles, str(id_fold), protein_name)
            stat_protein = analysis_to_csv_test(smiles,  protein_name, self.idx_fold, self.type_fold) #get the list of lists of statistics
            # stat_protein = np.transpose(np.vstack((stat_protein, np.asarray(amount_val_smiles * [amount_val_smiles /iter]))))
            stat_protein.append(amount_val_smiles * [amount_val_smiles /iter])
            stat_protein.append(amount_val_smiles * [self.sampling])
            stat_protein.append(amount_val_smiles * [self.model_encoder])
            stat_protein.append(amount_val_smiles * [self.model_decoder])
            # print("shape all_stat", all_stat.shape)
            # file_statistics.write(str(list(map(list, zip(*stat_protein)))) + "\n")
            wr = csv.writer(self.file_statistics)
            wr.writerows(list(map(list, zip(*stat_protein))))
            self.file_statistics.flush()
            

    def analysis_all():
        #for every fold takes indicies for the test, generates smiles and builds statistics
        num_folds = 3
        # all_stat = np.empty((1, 8))
        for id_fold in range(num_folds):
            file_freq = open(os.path.join(save_dir_smiles, str(id_fold), str(id_fold) + "_freq.txt"), "w")
            file_idx = os.path.join(save_dir_folds, "test_idx_" + str(id_fold))
            with (open(file_idx, "rb")) as openfile:
                idx_proteins = pickle.load(openfile)
            for id_protein in idx_proteins:
                self.generate_smiles(id_protein)
            
    
    def test_analysis_all():
        #for every fold takes indicies for the test, generates smiles and builds statistics
        num_folds = 3
        all_stat = []
        # idx_array = [[11,12], [14, 15]]
        idx_array = [[11], [14]]
        for id_fold in range(2):
            file_freq = open(os.path.join(save_dir_smiles, str(id_fold), str(id_fold) + "_freq.txt"), "w")
            idx_proteins = idx_array[id_fold]
            for id_protein in idx_proteins:
                self.generate_smiles(id_protein)


        # all_stat = np.array(all_stat)
        # print("shape all_stat", len(all_stat))
        # print("all_stat", all_stat)
        df = pd.DataFrame(all_stat, columns = ['name', 'fold', 'logP','sa','qed','weight','similarity', 'orig_logP', 'orig_sa', 'orig_qed', 'orig_weight','frequency'])
        df.to_csv(os.path.join(save_dir_smiles, "all_stat_new.csv"))
    

    def save_encodings_all(self):
        r'''For every protein id in rain/test generates feature and saves it
        '''
        #writes encodings to .pt files
        with (open(self.file_folds, "rb")) as openfile:
            idx_proteins_test = pickle.load(openfile)
        # Load the trained model parameters
        files_refined = os.listdir(self.protein_dir)
        idx_all = [i for i in range(len(files_refined) - 3)]
        #take indx of proteins in the training set
        idx_train =  np.setdiff1d(idx_all, idx_proteins_test)
        # for id_protein in idx_train:
        for id_protein in idx_proteins_test:    
            self.generate_encodings(id_protein)

    def collect_all_encodings(self):
        r''' Writes all saved features to 1 file
        '''
        files_encodings =  os.listdir(self.save_dir_encodings)
        all_encodings = []
        for file_enc in files_encodings:
            if(file_enc[0].isdigit()):
                path_to_enc = os.path.join(self.save_dir_encodings, file_enc)
                enc_from_torch = torch.load(path_to_enc, map_location=torch.device('cpu')).view(-1).detach().numpy() 
                print(type(enc_from_torch))
                all_encodings.append(enc_from_torch)
        all_encodings = np.asarray(all_encodings)
        np.savetxt(os.path.join(self.save_dir_encodings, 'all_encodings.csv'), all_encodings, delimiter=',') 



        # df = pd.DataFrame(all_stat, columns = ['name', 'fold', 'logP','sa','qed','weight','similarity', 'frequency'])
        # df = pd.DataFrame(all_stat, columns = ['name', 'fold', 'logP','sa','qed','weight','similarity', 'orig_logP', 'orig_sa', 'orig_qed', 'orig_weight','frequency'])
        # df.to_csv(os.path.join(save_dir_smiles, "all_stat_new.csv"))


        # all_stat = np.vstack((all_stat, stat_protein))
        # all_stat += map(list, zip(*stat_protein))
