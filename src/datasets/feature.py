import os
import multiprocessing
import shutil
from distutils.dir_util import copy_tree
from multiprocessing import Pool
from functools import partial
import re
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from moleculekit.molecule import Molecule
import pandas as pd
# from moleculekit.smallmol.smallmol import SmallMol
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from moleculekit.tools.atomtyper import prepareProteinForAtomtyping, getFeatures
from moleculekit.tools.voxeldescriptors import getChannels
# import dictionary of atoms' types and hot encoders
from src.datasets.dictionaries import atom_most_common, dict_atoms_hot, dict_atoms_simple, dict_atoms_masses, dict_atoms_charges
from src.utils.checkpoint import save_checkpoint_feature
import src.utils.config as config
import argparse

# from dict 
class Featuring():
    def __init__(self, cfg, radious, type_feature, type_filtering, h_filterig):
        """uses cfg file which is given as arg in "python train_captioning.py"
        """
        self.path_root = cfg['preprocessing']['path_root']
        self.path_data = cfg['data']['path']
        self.path_checkpoint = os.path.join(self.path_data,  "preprocess_checkpoint.csv")
        self.file_checkpoint_data = open(self.path_checkpoint,  "a+").close()
        # self.file_checkpoint_data.close()
        if (len(open(self.path_checkpoint).readlines()) == 0):
            print("creating the file...")
            with open(self.path_checkpoint,  "a+") as f:
                f.write('radious,type_feature,type_filtering,h_filterig'+ "\n")
        self.init_refined = self.path_root + "/data/new_refined/"
        self.init_casf = self.path_root + "/data/new_core_2016/"
        self.dict_atoms = dict_atoms_hot
        self.dict_atoms_simple = dict_atoms_simple
        self.dict_words = atom_most_common
        self.dict_atoms_masses = dict_atoms_masses
        self.dict_atoms_charges = dict_atoms_charges
        self.radious = radious
        self.type_feature = type_feature
        self.type_filtering = type_filtering
        self.h_filterig = h_filterig
        ##################refined files###################
        self.files_refined = os.listdir(self.init_refined)
        self.files_refined = [file for file in self.files_refined if file[0].isdigit()]
        self.files_refined.sort()
        self.idx_files_refined = list(range(0, len(self.files_refined)))
        # self.idx_files_refined = [0, 1]
        self.max_length = 0
        if not self.check_featuring():
            print("calculating max length...")
            self.run_parallel_max_length()
            print("writing to files...")
            self.run_parallel_write()
        else:
            f, m, g = self._get_feat_geo_from_file(0)
            self.max_length = f.shape[0]
        # array_names = [str(radious), self.type_feature, self.type_filtering, self.h_filterig]
        # self.name_checkpoint_features = '_'.join(array_names)
        # os.makedirs(os.path.join(self.path_data, "checkpoints"), exist_ok=True)
        # self.path_checkpoint_features = os.path.join(self.path_data, "checkpoints", self.name_checkpoint_features + ".pkl")
        # if (os.path.exists(self.path_checkpoint_features)):
        #     print("loading feature ids...")
        #     checkpoint_features = torch.load(self.path_checkpoint_features)
        #     self.idx_max_length = checkpoint_features['idx_max_length']
        #     self.max_length = checkpoint_features['max_length']
        #     self.idx_write = checkpoint_features['idx_write']
        # else:
        #     self.idx_max_length = 130
        #     self.max_length = 0
        #     self.idx_write = 0
        #     save_checkpoint_feature(self.path_checkpoint_features, self.idx_max_length, self.max_length, self.idx_write)

            # self.max_length = 0
            # self.write_filtered_pad_feat_geo()
        # else:
        #     f, m, g = self._get_feat_geo_from_file(0)
        #     self.max_length = f.shape[0]

    def run_parallel_write(self):
        self.files_refined = os.listdir(self.init_refined)
        self.files_refined = [file for file in self.files_refined if file[0].isdigit()]
        self.files_refined.sort()
        self.idx_files_refined = list(range(0, len(self.files_refined)))
        with Pool(processes=8) as pool:
            pool.map(self.write_padd_feat_geo, self.idx_files_refined)
        self.write_checkpoint()

    def run_parallel_max_length(self):
        with Pool(processes=8) as pool:
            lengthes = pool.map(self._get_length, self.idx_files_refined) 
        # lengthes = []
        # with Pool(processes=8) as pool:
        #     with tqdm(total=len(self.idx_files_refined)) as pbar:
        #         for i, res in tqdm(enumerate(pool.imap_unordered(self._get_length, self.idx_files_refined))):
        #             lengthes.append(res)
        #             pbar.update()
            # lengthes = list(tqdm.tqdm(pool.imap(self._get_length, self.idx_files_refined), total=len(self.idx_files_refined)))
            # lengthes = pool.map(self._get_length, self.idx_files_refined) 
        self.max_length = max(lengthes)
        print("********max********* - ", self.max_length)
    
    def _get_length(self, pdb_id):
        features_filt, geo_filt = self._get_features_geo_filtered(pdb_id)
        length = features_filt.shape[0]
        return length

    def write_padd_feat_geo(self, id):
        feat_filt_padded, masks, geo_filt_padded = self._get_features_geo_padded(id, self.max_length)
        path_feature, path_mask, path_geo = self._get_name_save(id)
        torch.save(feat_filt_padded, path_feature)
        torch.save(masks, path_mask)
        torch.save(geo_filt_padded, path_geo)

    def _get_feat_geo_from_file(self, id):
        """reads torch tensors of feature/geo from files

        Args:
            id ([int]): [pdb id of a protein]

        Returns:
            [toch.array]: [feature/geo padded filtered tensors from saved files]
        """
        path_feature, path_mask, path_geo = self._get_name_save(id)
        feature_filt_padded = torch.load(path_feature, map_location=torch.device('cpu')).long()
        mask = torch.load(path_mask, map_location=torch.device('cpu'))
        geo_filt_padded = torch.load(path_geo, map_location=torch.device('cpu'))
        return feature_filt_padded, mask, geo_filt_padded

    def write_filtered_pad_feat_geo(self):
        """1. calculates max length of feat/gep tensors
           2. padds feat/geo tensors with zeros till the max length
           3. writes resulting tensor to the file
        """
        length_max = self._get_length_max()
        data_list = range(self.idx_write, len(self.files_refined))
        # length_max = 150
        progress = tqdm(data_list)
        for id in progress:
            progress.set_postfix({'pdb': self.files_refined[id]})
            feat_filt_padded, masks, geo_filt_padded = self._get_features_geo_padded(id, length_max)
            path_feature, path_mask, path_geo = self._get_name_save(id)
            torch.save(feat_filt_padded, path_feature)
            torch.save(masks, path_mask)
            torch.save(geo_filt_padded, path_geo)
            save_checkpoint_feature(self.path_checkpoint_features, len(self.files_refined), self.max_length, id)
        self.write_checkpoint()

    def _get_name_save(self, id: int):
        """creates a path name for feature/geo

        Example:
           1a1e_feature_r_5_hot_simple_all_no_h.pt
           1a1e_geo_r_5_hot_simple_all_no_h.pt

        Args:
            id ([int]): [pdb id of a protein]

        Returns:
            [str]: [path name for feature and geometry]
        """
        # print("id", id)
        name_protein = self.files_refined[id]
        array_feat_names = [name_protein, "feature", "r", str(self.radious), self.type_feature, self.type_filtering, self.h_filterig]
        array_mask_names = [name_protein, "mask", "r", str(self.radious), self.type_feature, self.type_filtering, self.h_filterig]
        array_geo_names = [name_protein, "geo", "r", str(self.radious), self.type_feature, self.type_filtering, self.h_filterig]
        name_feature = "_".join(array_feat_names) + ".pt"
        name_mask = "_".join(array_mask_names) + ".pt"
        name_geo = "_".join(array_geo_names) + ".pt"
        path_feat = os.path.join(self.init_refined, name_protein, name_feature)
        path_mask = os.path.join(self.init_refined, name_protein, name_mask)
        path_geo = os.path.join(self.init_refined, name_protein, name_geo)
        return path_feat, path_mask, path_geo

    def _get_features_geo_padded(self, id: int, length_max):
        """padds filtered feature/geometry tensors till the max length

        Args:
            id ([int]): [pdb id]

        Returns:
            [torch.tensor]: [padded tensors [1 * length_max * feat_length]]
        """
        features_filt, geo_filt = self._get_features_geo_filtered(id)
        length_padding = length_max - features_filt.shape[0]
        mask_binary = torch.cat([torch.ones(features_filt.shape[0]),torch.zeros(length_padding)]).squeeze()
        # feat_padd_vector = torch.zeros(features_filt.shape[2])
        feat_filt_padded = F.pad(
            input=features_filt,
            pad=(0, 0, 0, length_padding),
            mode="constant",
            value = 0,
        )
        geo_filt_padded = F.pad(
            input=geo_filt,
            pad=(0, 0, 0,  length_padding),
            mode="constant",
            value=99,
        )
        return feat_filt_padded, mask_binary, geo_filt_padded

    def _get_length_max(self):
        """get the max length of feature array among all pdbids

        Returns:
            [int]: [maximum length]
        """
        # data_list = list(range(len(self.files_refined)))
        data_list = range(self.idx_max_length, len(self.files_refined))
        progress = tqdm(data_list)
        for pdb_id in progress:
            features_filt, geo_filt = self._get_features_geo_filtered(pdb_id)
            length = features_filt.shape[1]
            if (length > self.max_length):
                self.max_length = length
            progress.set_postfix({'pdb': self.files_refined[pdb_id],
                                  'length': length,
                                  'max_langth': self.max_length})
            save_checkpoint_feature(self.path_checkpoint_features, pdb_id, self.max_length, id)
        return self.max_length

    def _get_features_geo_filtered(self, pdb_id):
        """calculates features and geometry with filteing

        Args:pdb id of a protein]

        Returns:
            [torch.tensor]: [Num_atoms * Feat_dim]
        """
        features, geometry = self._get_features_geo(pdb_id)
        mask = self._get_mask_selected_atoms_pocket(pdb_id)
        features_filtered, geometry_filtered = features[mask, :], geometry[mask, :]
        features_filtered = torch.from_numpy(features_filtered).squeeze()
        geometry_filtered = torch.from_numpy(geometry_filtered).squeeze()
        return features_filtered, geometry_filtered

    def _get_features_geo(self, id):
        """gets features depending on the type of featuring
           Implemented: hot_simple, mass_charges, bio_properties

        Args:
            id ([str]): [id of a protein]

        Returns:
            [np.asarray]: [arrays of feature, geometry for a given pdb id]
        """
        #creates featues/geo tensors for all atoms in protein
        if self.type_feature == "hot_simple":
            features = self.hot_enc(id)
        elif self.type_feature == "mass_charges":
            features = self.mass_charges(id)
        elif self.type_feature == "bio_properties":
            features = self.bio_prop(id)
        elif self.type_feature == "bio_all_properties":
            features_1 = self.mass_charges(id)
            features_2 = self.bio_prop(id)
            features = np.concatenate((features_1, features_2), axis=1)
        geometry = self._get_geometry_protein(id)
        return features, geometry

    def hot_enc(self, id):
        #creates hot vector encoding for all atoms!
        elems = self._get_all_elems(id)
        features = [self.atom_to_hot_vector(elem) for elem in elems]
        features = np.asarray(features)
        return features

    def atom_to_hot_vector(self, elem: str):
        """ creates a hot vector of an atom type

        Parameters
        ----------
        elem   : str atom element
        """
        hot_vector = np.zeros(22)
        idx = self.dict_atoms_simple[elem]
        hot_vector[idx] = 1
        return hot_vector

    def mass_charges(self, id):
        """calculates "smart" hot vectors for the whole protein (all atoms!)
            mass of atoms on the atomic number's position

        Args:
            id ([type]): [description]

        Returns:
            [np.asarray]: [array of features [Num_elems * 80]]
        """
        elems = self._get_all_elems(id)
        features = [self.atom_to_mass_charge_hot(elem) for elem in elems]
        features = np.asarray(features)
        return features

    def atom_to_mass_charge_hot(self, elem: str):
        atom_mass = self.dict_atoms_masses[elem]
        atom_charge_idx = self.dict_atoms_charges[elem]
        vector = np.zeros(80)
        vector[atom_charge_idx] = atom_mass
        return vector
 
    def bio_prop(self, id: int):
        """calculates pharmacophoric properties for the whole protein (all atoms!)

        Args:
            id ([int]): [pdb id of a protein]

        Returns:
            [np.array]: [array of pharmacophoric properties [N_atoms, dim_feature]]
        """
        #pocket
        try:
            path_protein, _ = self._get_path(id)
            protein_name = self.files_refined[id]
            print("processing...", protein_name)
            mol = Molecule(path_protein)
            mol.filter('protein')
            mol = prepareProteinForAtomtyping(mol, verbose = False)

            features = getChannels(mol, version=2)
            features = (features[0] > 0).astype(np.float32)
            features = np.asarray(features[:, :-1])
        # print("feat shape bio - ", features.shape)
        except RuntimeError:
            path_to_exceptions = os.path.join(self.path_data, "exceptions")
            path_protein_folder = os.path.join(self.init_refined, protein_name)
            os.makedirs(self.path_to_exceptions, exist_ok=True)
            copy_tree(path_protein_folder, path_to_exceptions)
            shutil.rmtree(path_protein_folder)
        return features
    
    def _get_mask_selected_atoms_pocket(
        self, pdb_id: int,
    ):
        """selects atoms of "id_pdb" protein within the distance "precision" around "center_lig"
        
        Parameters
        ----------
        id_pdb   : str id of a protein
                Protein to be processed
        center : array
            Geometrical center of a ligand
        radious : int
            Radius of atoms selections wrp center of ligand
        """

        path_protein, path_ligand = self._get_path(pdb_id)
        center_ligand = self._get_ligand_center(path_ligand)
        if self.type_filtering == "all" and self.h_filterig == 'h':
            sel="protein and noh and sqr(x-'{0}')+sqr(y-'{1}')+sqr(z-'{2}') <= sqr('{3}')".format(
                str(center_ligand[0][0]),
                str(center_ligand[0][1]),
                str(center_ligand[0][2]),
                str(self.radious),
            )
        elif self.type_filtering == "all" and self.h_filterig == '-h':
            sel="sqr(x-'{0}')+sqr(y-'{1}')+sqr(z-'{2}') <= sqr('{3}')".format(
                str(center_ligand[0][0]),
                str(center_ligand[0][1]),
                str(center_ligand[0][2]),
                str(self.radious),
            )
        mol_protein = Molecule(path_protein)
        mol_protein.filter('protein')
        if (self.type_feature == "bio_properties" or self.type_feature == "bio_all_properties"):
            mol_protein = prepareProteinForAtomtyping(mol_protein, verbose = False)
        mask = mol_protein.atomselect(sel)
        return mask

    def _get_ligand_center(self, path_ligand):
        """get the geometrical center of a ligand

        Args:
            path_ligand ([str]): [path to the mol2 file]

        Returns:
            [np.asarray]: geo center of a ligand
        """
        mol_ligand = Molecule(path_ligand)
        coor_lig = mol_ligand.coords
        center = np.mean(coor_lig, axis=0)
        center = center.reshape(1, -1)
        return center

    def _get_all_elems(self, protein_id: int):
        """takes all elems in protein

        Args:
            protein_id (int): [id of a protein]

        Returns:
            [list]: [all elements]
        """
        path_protein, _ = self._get_path(protein_id)
        try:
            # mol_pocket = Molecule(path_protein)
            mol_protein = Molecule(path_protein)
            mol_protein.filter('protein')
            if (self.type_feature == "bio_properties" or self.type_feature == "bio_all_properties"):
                mol_protein = prepareProteinForAtomtyping(mol_protein, verbose = False)
            mol_pocket_element = mol_protein.element
        except FileNotFoundError:
            print(protein_id, "   exception")
            path_protein, path_lig = self._get_path(2)
            mol_pocket = Molecule(path_protein)
            mol_pocket_element = mol_pocket.element
        return mol_pocket_element

    def _get_all_elem_general(self, protein_id: int):
        path_protein, _ = self._get_path(protein_id)
        try:
            # mol_pocket = Molecule(path_protein)
            mol_protein = Molecule(path_protein)
            mol_protein.filter('protein')
            mol_pocket_element = mol_protein.element
        except FileNotFoundError:
            print(protein_id, "   exception")
            path_protein, path_lig = self._get_path(2)
            mol_pocket = Molecule(path_protein)
            mol_pocket_element = mol_pocket.element
        return mol_pocket_element



 
    def _get_geometry_protein(self, protein_id: int):
        """ gives np.array of coordinates for a pocket and a ligand in one complex

        Parameters
        ----------
        protein_id   : str
                      id of a complex
        """
        path_protein, _ = self._get_path(protein_id)
        mol_protein = Molecule(path_protein)
        mol_protein.filter("protein")
        if (self.type_feature == "bio_properties" or self.type_feature == "bio_all_properties"):
            mol_protein = prepareProteinForAtomtyping(mol_protein, verbose = False)
        coords_protein = mol_protein.coords
        coords_protein = np.asarray(coords_protein)
        return coords_protein

    def _get_path(self, protein_id: int):
        """ get a full path to protein/ligand

        """
        protein_name = self.files_refined[protein_id]
        path_protein = os.path.join(
            self.init_refined, protein_name, protein_name + "_protein.pdb"
        )
        path_ligand = os.path.join(
            self.init_refined, protein_name, protein_name + "_ligand.mol2"
        )
        return path_protein, path_ligand

    def write_checkpoint(self):
        """writes inf about radious, type_feature, type_filtering, h_filterig used at extracting features/geometry of atoms
        """
        self.file_checkpoint_data = open(self.path_checkpoint, "a+")
        array_to_write = [str(self.radious), self.type_feature, self.type_filtering, self.h_filterig]
        self.file_checkpoint_data.write(','.join(array_to_write) + "\n")
        self.file_checkpoint_data.flush()

    def check_featuring(self):
        """check if feature generation was already done with params (mentioned in command line args)

        Returns:
            [bool]: [True if generation was done/ False if wasn't]
        """
        existing_featuring = pd.read_csv(self.path_checkpoint)
        array_to_check = [float(self.radious), self.type_feature, self.type_filtering, self.h_filterig]
        bool_answer = (existing_featuring == array_to_check).all(1).any()
        # self.file_checkpoint_data.close()
        return bool_answer


class Batch_prep(Featuring):
    def __init__(self, cfg, radious, type_feature, type_filtering, h_filterig, n_proc=2, mp_pool=None):
        super(Batch_prep, self).__init__(cfg, radious, type_feature, type_filtering, h_filterig)
        self.mp = multiprocessing.Pool(n_proc)

    def transform_data(self):
        inputs = self.mp.map(self._get_length, self.files_refined)
        # Sometimes representation generation fails
        inputs = list(filter(lambda x: x is not None, inputs))
        return max(inputs)

def main():
    parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model.')
    
    parser.add_argument('--config', type=str, help='Path to config file.')
    parser.add_argument('--radious', type=int , default=8, help='dimension of word embedding vectors')
    parser.add_argument('--type_feature', type=str , default='mass_charge', help='type_feature')
    parser.add_argument('--type_filtering', type=str , default = 'all', help='type_filtering')
    parser.add_argument('--h_filterig', type=str , default='without_h', help='h')
    parser.add_argument('--type_fold', type=str, help='type_fold')
    parser.add_argument('--idx_fold', type=str, help='idx fold')
    args = parser.parse_args()
                         
    cfg = config.load_config(args.config, 'configurations/config_local/default.yaml')
    type_fold = args.type_fold
    idx_fold = args.idx_fold
    savedir = cfg["output_parameters"]["savedir"]
    model_name = cfg["model_params"]["model_name"]
    num_epoches = cfg["model_params"]["num_epochs"]
    #features generation
    Feature_gen = Featuring(cfg, args.radious, args.type_feature, args.type_filtering, args.h_filterig)
    print("max length", Feature_gen.max_length)

if __name__ == "__main__":
    main()


     