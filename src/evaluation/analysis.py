import csv
import os
import pickle
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm, pyplot
# from matplotlib import pyplot
from numpy import mean, std
# from captioning_e3nn.Contrib.NP_Score.npscorer_my import processMols
from rdkit import Chem, DataStructs, RDConfig
from rdkit.Chem import QED, AllChem, Descriptors
from scipy import stats
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
matplotlib.use('TkAgg')
matplotlib.rcParams.update({'font.size': 10})
from sklearn.preprocessing import MinMaxScaler


class Tree_Analysis(dict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value
    def get_mean(self, sampling: str, param: str):
        mean_array = np.asarray(list(d[sampling][param].values()))
        mean = np.mean(mean_array, axis = 0)
        return mean



class plot_all():
    def __init__(self, cfg, epoch):
        self.path_data = os.path.join(cfg["output_parameters"]["savedir"], cfg["model_params"]["model_name"], "statistics")
        self.names_gen_properties = ["gen_NP", "gen_weight", "gen_logP", "gen_sa"]
        self.names_orig_properties = ['orig_NP', 'orig_weight', 'orig_logP', 'orig_sa']
        self.num_epochs = cfg['model_params']['num_epochs']
        self.epoch = epoch
        self.files = os.listdir(self.path_data)
        self.dict_analysis = Tree_Analysis()
        self.dict_orig = Tree_Analysis()
        self.dict_sim = Tree_Analysis()
        self.rand_sim = self.get_random_perm()
        self.gen_to_orig = {"gen_NP": 'orig_NP',
                            "gen_weight": 'orig_weight',
                            "gen_logP":  'orig_logP',
                            "gen_sa": 'orig_sa'}
        
        self.colors = ['b', 'r', 'c', 'm', 'k', 'y', 'w']
        self.model_name = cfg['model_params']['model_name']
        self.path_vis = os.path.join(cfg["output_parameters"]["savedir"], self.model_name, 'results_' + self.model_name)
        self.path_sim = os.path.join(self.path_vis, 'similarity')
        self.path_prop = os.path.join(self.path_vis, 'properties')
        os.makedirs(self.path_sim, exist_ok=True)
        os.makedirs(self.path_prop, exist_ok=True)
        

    def get_random_perm(self):
        with open("../data/all_smiles_lig.txt") as f:
            list_smiles = f.read().splitlines()
   
        #random permutation
        perm = list(range(len(list_smiles)))
        random.shuffle(perm)
        perm_smiles = [list_smiles[index] for index in perm]

        mol_orig = [AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smile), 2) for smile in list_smiles] #for original
        mol_perm = [AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smile), 2) for smile in perm_smiles] #for permuted

        similarity = [DataStructs.DiceSimilarity(mol_orig[i],mol_perm[i]) for i in range(len(mol_orig))] #array of similarities
        return similarity

    def get_array(self, file: str, name: str):
        data = pd.read_csv(os.path.join(self.path_data, file))
        array = data.loc[data['epoch_no'] ==  self.epoch, name].to_list()
        # array = data[name].to_list()
        return array
    
    
    def get_dim(self):
        self.dim_splits = len(self.dict_analysis)
    
    def allign_dict(self, dict_values: list):
        dim_splits = len(dict_values)
        max_length = 0
        for i in range(dim_splits):
            if(len(dict_values[i]) > max_length):
                max_length = len(dict_values[i])
        for i in range(dim_splits):
            dict_values[i] += [0] * (max_length - len(dict_values[i]))
            
    def _get_average_property(self, method, name_split, property_mol):
        all_l = list(self.dict_analysis[name_split][property_mol][method].values())
        self.allign_dict(all_l)
        lst = np.asarray(all_l)
        mean = np.mean(lst, axis = 0)
        return mean
    
    def _get_average_sim(self, method, name_split, property_mol):
        all_l = list(self.dict_sim[name_split][property_mol][method].values())
        self.allign_dict(all_l)
        lst = np.asarray(all_l)
        mean = np.mean(lst, axis = 0)
        return mean
    
    def _get_average_orig(self, name_split, property_mol):
        all_l = list(self.dict_orig[name_split][property_mol].values())
        self.allign_dict(all_l)
        lst = np.asarray(all_l)
        mean = np.mean(lst, axis = 0)
        return mean
        
    def build_dict(self):
        methods = []
        files_exception = [".ipynb_checkpoints", "exceptions_long.csv", "exceptions_long.txt", "stat_e3nn_prob_0.csv", "temp_sampling_0.8_random_0"]
        for file in self.files:
            if file not in files_exception:
                print("file", file)
                parts = file.split("_")
                if (len(parts) < 4):
                    method = parts[0]
                    id_fold = parts[-1]
                    name_split = parts[1]
                else:
                    method = parts[1] + parts[0]
                    id_fold = parts[-1]
                    # print("id_fold", id_fold)
                    name_split = parts[2]
                    # name_split = parts[2][:-4]
                if (method not in methods):
                    methods.append(method)
                for property_name in self.names_gen_properties:           
                    self.dict_analysis[name_split][property_name][method][id_fold] = self.get_array(file, property_name)
                for property_name in self.names_orig_properties:
                    self.dict_orig[name_split][property_name][id_fold] = self.get_array(file, property_name)
                self.dict_sim[name_split]["gen_similarity"][method][id_fold] = self.get_array(file, "gen_similarity")
          
        self.num_splits = len(self.dict_analysis)
#         self.num_methods = len(self.dict_analysis['random'])
        self.num_methods = len(methods)
        print("num plits", self.num_splits)
        print("num_methods", self.num_methods)

            
    def plot_similarity(self):
        fig, axs = plt.subplots(nrows = 1, ncols = self.num_splits)
        fig.set_figheight(15)
        fig.set_figwidth(40)
        for id_split, name_split in enumerate(list(self.dict_sim)):
#             ax_all = axs[id_split]
            ax_all = axs
            fig1, axs1 = plt.subplots(nrows = 1, ncols = self.num_methods) #for local file for every fold type split
            plt.title = 'Histogram of Shear Strength'
            fig1.set_figheight(7)
            fig1.set_figwidth(25)
            fig1.suptitle(name_split, fontsize=26)
            plt.ylabel('Density')
            plt.xlabel('Similarity')
            pyplot.legend(loc='upper right')
            sns.distplot(self.rand_sim, color='green', hist=True, rug=False, label= 'Shuffled pairs', ax = ax_all);
            for id_method, method_name in enumerate(list(self.dict_sim[name_split]['gen_similarity'])):
                print("sim_method_name - ", method_name)
                sim_array = self._get_average_sim(method_name, name_split, "gen_similarity")
                color = self.colors[id_method]
                color_rand = self.colors[-1]
                ax1 = axs1[id_method]                
                mean_sim = mean(sim_array)
                mean_sim_rand = mean(self.rand_sim)
                
                sns.distplot(sim_array, color=color, hist=True, rug=False, label= method_name, ax = ax_all);
                ax_all.axvline(mean_sim, color='blue', linestyle='--')
                ax_all.axvline(mean_sim_rand, color='green', linestyle='--')
                ax_all.set_title(name_split + ' sampling', fontsize=70)
                ax_all.set_xlabel('Distance Similarity', fontsize=70)
                ax_all.set_ylabel('Density', fontsize=70)
                ax_all.legend(loc='upper right', fontsize=70)
                
                sns.distplot(self.rand_sim, color='green', hist=True, rug=False, label= 'Shuffled pairs', ax = ax1);
                sns.distplot(sim_array,  color='blue', hist=True, rug=False, label= 'Generated pairs', ax = ax1);
                ax1.axvline(mean_sim, color='blue', linestyle='--')
                ax1.axvline(mean_sim_rand, color='green', linestyle='--')
                ax1.set_title(method_name)
                ax1.set_ylabel('Density')
                ax1.set_xlabel('Distance Similarity')
                ax1.legend(loc='upper right')
                
                
                fig_local, axs_local = plt.subplots(nrows = 1, ncols =1)
                sns.distplot(sim_array, color='blue', hist=True, rug=False, label= 'Generated pairs', ax = axs_local);
                sns.distplot(self.rand_sim, color='green', hist=True, rug=False, label= 'Shuffled pairs', ax = axs_local);
                axs_local.set_title(method_name)
                axs_local.axvline(mean_sim_rand, color='green', linestyle='--')
                axs_local.axvline(mean_sim, color = color, linestyle='--')

                axs_local.set_ylabel('Density')
                axs_local.set_xlabel('Distance Similarity')
                axs_local.legend(loc='upper right')
                os.makedirs(os.path.join(self.path_sim, name_split), exist_ok=True)
                fig_local.savefig(os.path.join(self.path_sim, name_split, method_name + '.pdf'), dpi=600)
                
            name = name_split + "_epoch_" + str(self.epoch) + "_sim.pdf"
            fig1.savefig(os.path.join(self.path_sim, name), dpi = 600)
        name_all = "sim_all.pdf"
        fig.savefig(os.path.join(self.path_sim, name_all), dpi=600)

        
    def plot_properties(self):
        num_splits = len(self.dict_analysis)
        #iterate over random/chain/scaffold split
        for id_split, name_split in enumerate(list(self.dict_analysis)):
            fig1, axs = plt.subplots(nrows = 1, ncols = 4)
            fig1.suptitle(name_split, fontsize=20)
            fig1.set_figheight(7)
            fig1.set_figwidth(25)
            #iterate over NP, weight...
            for id_property, property_name in enumerate(list(self.dict_analysis[name_split])):
                fig_local, axs_local = plt.subplots(nrows = 1, ncols =1)
                os.makedirs(os.path.join(self.path_prop, name_split), exist_ok=True)
                
                ax1 = axs[id_property]
                orig_name = self.gen_to_orig[property_name]
                print("orig name", orig_name)
                prop_array_orig = self._get_average_orig(name_split, orig_name)
                
                sns.distplot(prop_array_orig, hist=True, color = 'black', rug=False, label= orig_name, ax = ax1);
                sns.distplot(prop_array_orig, hist=True, color = 'green', rug=False, label= orig_name, ax = axs_local)
                mean_array_orig = mean(prop_array_orig)
                ax1.axvline(mean_array_orig, color = 'green', linestyle='--')
                
                axs_local.axvline(mean_array_orig, color = 'green', linestyle='--')
                axs_local.set_ylabel('Density')
                axs_local.set_title(property_name + ' sampling')
                #iterate over sampling (max, probabilistic)...
                for id_method, method_name in enumerate(list(self.dict_analysis[name_split][property_name])):
                    prop_array = self._get_average_property(method_name, name_split, property_name)
                    color = self.colors[id_method]
                    sns.distplot(prop_array, hist=True, color = color, rug=False, label= method_name, ax = ax1)
                    mean_prop_array = mean(prop_array)

                    ax1.axvline(mean_prop_array, color = color, linestyle='--')
                    ax1.set_ylabel('Density')
                    ax1.set_title(property_name)
                    ax1.legend(loc='upper right')
        
                    sns.distplot(prop_array, hist=True, color = color, rug=False, label= method_name, ax = axs_local);
                    axs_local.axvline(mean_prop_array, color = color, linestyle='--')
                    axs_local.legend(loc= 'upper right')
                fig_local.savefig(os.path.join(self.path_prop, name_split, property_name + '.pdf'), dpi=600)    
            name = name_split + "_prop.pdf"
            fig1.savefig(os.path.join(self.path_prop, name), dpi=600)

    def run(self):
        self.build_dict()
        self.plot_similarity()
        self.plot_properties()
        
