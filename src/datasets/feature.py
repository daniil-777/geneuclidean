import os
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

# from dict 
class Featuring():
    def __init__(self, cfg, radious, type_feature, type_filtering, h_filterig):
        """uses cfg file which is given as arg in "python train_captioning.py"
        """
        self.path_root = cfg['preprocessing']['path_root']
        self.path_data = cfg['data']['path']
        self.path_checkpoint = os.path.join(self.path_data,  "preprocess_checkpoint.csv")
        self.file_checkpoint_data = open(self.path_checkpoint,  "a+")
  
        if (len(open(self.path_checkpoint).readlines()) == 0):
            self.file_checkpoint_data.write('radious,type_feature,type_filtering,h_filterig'+ "\n")
            self.file_checkpoint_data.flush()

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
        
        if not self.check_featuring():
            self.max_length = 0
            self.write_filtered_pad_feat_geo()
        else:
            f, m, g = self._get_feat_geo_from_file(0)
            self.max_length = f.shape[0]

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
        # length_max = self._get_length_max()
        length_max = 150
        progress = tqdm(range(len(self.files_refined)))
        for id in progress:
            progress.set_postfix({'pdb': self.files_refined[id]})
            feat_filt_padded, masks, geo_filt_padded = self._get_features_geo_padded(id, length_max)
            path_feature, path_mask, path_geo = self._get_name_save(id)
            torch.save(feat_filt_padded, path_feature)
            torch.save(masks, path_mask)
            torch.save(geo_filt_padded, path_geo)
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
        data_list = list(range(len(self.files_refined)))
        progress = tqdm(data_list)
    
        for pdb_id in progress:
            features_filt, geo_filt = self._get_features_geo_filtered(pdb_id)
            length = features_filt.shape[1]
            if (length > self.max_length):
                self.max_length = length
            progress.set_postfix({'pdb': self.files_refined[pdb_id],
                                  'length': length,
                                  'max_langth': self.max_length})
        return self.max_length

    def _get_features_geo_filtered(self, id):
        """calculates features and geometry with filteing

        Args:pdb id of a protein]

        Returns:
            [torch.tensor]: [Num_atoms * Feat_dim]
        """
        features, geometry = self._get_features_geo(id)
        mask = self._get_mask_selected_atoms_pocket(id)
        print("feat shape", features.shape)
        print("geo shape", geometry.shape)
        print("mask shape", len(mask))
        features_filtered, geometry_filtered = features[mask, :], geometry[mask, :]
        print("shape feat_filt", features_filtered.shape)
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
        #
        # selection_mask = self.mask_selected_atoms_pocket(id)
        # mol = Molecule(id, validateElements=False)
        # mol.filter('protein')
        #pocket
        path_protein, _ = self._get_path(id)
        protein_name = self.files_refined[id]
        # mol = Molecule(protein_name)
        # mol.filter('protein')
        mol = Molecule(path_protein)
        mol.filter('protein')
        mol = prepareProteinForAtomtyping(mol, verbose = False)

        features = getChannels(mol, version=2)
        features = (features[0] > 0).astype(np.float32)
        features = np.asarray(features[:, :-1])
        print("feat shape bio - ", features.shape)
        return features
    
    def _get_mask_selected_atoms_pocket(
        self, id_pdb: int,
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

        path_protein, path_ligand = self._get_path(id_pdb)
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
                str(center_lig[0][0]),
                str(center_lig[0][1]),
                str(center_lig[0][2]),
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
        # self.file_checkpoint_data = open(self.path_checkpoint, "a+")
        array_to_write = [str(radious), self.type_feature, self.type_filtering, self.h_filterig]
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
        return bool_answer



     