import os
from functools import partial

import numpy as np
import torch
from moleculekit.molecule import Molecule
from moleculekit.smallmol.smallmol import SmallMol
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# import dictionary of atoms' types and hot encoders
from data import dict_atoms

# number_atoms_unique = 22
LEN_PADDING = 286


class Pdb_Dataset(Dataset):
    """PdB binding dataset"""

    # def __init__(self, path_pocket: str, path_ligand: str):

    def __init__(self, path_root: str):
        # self.labels = self.read_labels(path_pocket + "labels.csv")
        # self.init_pocket = path_pocket
        # self.init_ligand = path_ligand
        print(path_root)
        self.init_pocket = path_root + "/new_dataset/"
        self.init_ligand = path_root + "/refined-set/"
        # self.init_core_ligand = path_root + "/CASF/ligand/docking/decoy_mol2/"
        self.init_core_ligand = path_root + "/CASF/ligand/ranking_scoring/crystal_mol2/"
        self.target_casf = path_root + "/core_processed_dataset/"
        self.labels = self.read_labels(path_root + "/data/labels.csv")
        
        self.labels_all = self._get_labels_refined_core(
            path_root + "/data/labels.csv", path_root + "/data/labels_core.csv")
        self.files_pdb = os.listdir(self.init_pocket)
        self.files_pdb.sort()
        self.len_files = len(self.files_pdb)
        self.files_core = os.listdir(self.target_casf)

        self.dict_atoms = dict_atoms
        self.set_atoms = []
        self.encoding = {}
        self.label_protein = np.array([1.0])  # identification of pocket
        self.label_ligand = np.array([-1.0])  # identification of ligand
        self.features_complexes = []  # tensors of euclidean features
        self.affinities_complexes = []  # targets


    def __len__(self):
        return len(self.files_pdb)


    def __getitem__(self, idx: int):
        # idx_str = self.index_int_to_str(idx)
        all_features = self._get_features_complex(idx)
        all_geometry = self._get_geometry_complex(idx)
        # target_pkd = np.asarray(self.labels[self.files_pdb.index(idx)])
        target_pkd = np.asarray(self.labels_all[idx])
        return idx, all_features, all_geometry, torch.from_numpy(target_pkd)
        # item = {
        #     "pdb_id": idx,
        #     "feature": all_features,
        #     "geometry": all_geometry,
        #     "target": self.labels[self.files_pdb.index(idx_str)],
        # }
        # return item

    def _get_path(self, protein_id: int):
        """ get a full path to pocket/ligand

        """
        if (protein_id >= self.len_files):
            new_id = protein_id-self.len_files
            protein_name = self.files_core[new_id]
            path_pocket = os.path.join(
                self.target_casf, protein_name, protein_name + "_pocket.pdb")
            path_ligand=os.path.join(
                self.init_core_ligand,  protein_name + "_ligand.mol2")
        else:
            protein_name = self.files_pdb[protein_id]
            path_pocket = os.path.join(
                self.init_pocket, protein_name, protein_name + "_pocket.pdb"
            )
            path_ligand = os.path.join(
                self.init_ligand, protein_name, protein_name + "_ligand.mol2"
            )
        return path_pocket, path_ligand

    def _get_elems(self, protein_id: int):
        """ gives np.array of elements for a pocket and a ligand in one complex

        Parameters
        ----------
        protein_id   : str
                      id of a complex
        """
        path_pocket, path_ligand = self._get_path(protein_id)
        mol_pocket = Molecule(path_pocket)
        mol_ligand = Molecule(path_ligand)
        return mol_pocket.element, mol_ligand.element

    def _get_coord(self, protein_id: int):
        """ gives np.array of coordinates for a pocket and a ligand in one complex

        Parameters
        ----------
        protein_id   : str
                      id of a complex
        """

        path_pocket, path_ligand = self._get_path(protein_id)
        mol_pocket = Molecule(path_pocket)
        mol_ligand = Molecule(path_ligand)
        return mol_pocket.coords, mol_ligand.coords
        # print(mol_ligand.element)

        # print(mol_ligand.coords.shape)
        # print(mol_ligand.element.shape)

    def index_int_to_str(self, index: int):
        """ creates a key name (string) of a protein-pdb

        Parameters
        ----------
        index   : int position of the pdb in he common list
        """
        if index > 5000:
            num = index - 5000
            name_pdb_id = self.files_core[index]
        else:
            name_pdb_id = self.files_pdb[index]
        return name_pdb_id

    def atom_to_vector(self, elem: str):
        """ creates a hot vector of an atom

        Parameters
        ----------
        elem   : str atom element
        """
        return self.dict_atoms[elem]

    def coords_to_tensor(self, coords: np.array):
        """ creates a tensor of coords 

        Parameters
        ----------
        coords   : array of coords of n atoms [n, 3]
        """
        return torch.tensor(coords)

    def _get_feature_vector_atom(self, elem: str, type_atom: str):
        """creates a tensor-feature vector concatenating label of protein/ligand and hot vector

        Parameters
        ----------
        elem   : str atom element
        """
        hot_vector_atom = self.atom_to_vector(elem)
        if type_atom == "pocket":
            feature_vector_atom = np.concatenate((self.label_protein, hot_vector_atom))
        elif type_atom == "ligand":
            feature_vector_atom = np.concatenate((self.label_ligand, hot_vector_atom))
        else:
            raise ValueError("type of atom should be pocket or ligand")
        # feature_tensor_atom = torch.from_numpy(feature_vector_atom)
        # return feature_tensor_atom
        return feature_vector_atom

    def _get_features_unit(self, elements: np.array, type_atom: str):
        """creates a union of tensors-features of an atoms' array at particlular biological unit: pocket/ligand 

        Parameters
        ----------
        elements   : np.array
                   elements of protein/ligand
        type_atom  : char
                   type of a biological unit: pocket/ligand

        Returns
        -------
        list_features_tensors : list
            The list of features-tensors
        """

        list_features_tensors = []
        for elem in elements:
            tensor_feature = self._get_feature_vector_atom(elem, type_atom)
            list_features_tensors.append(tensor_feature)
        # features = torch.cat(list_features_tensors, dim=-1)
        return list_features_tensors

    def _get_features_complex(self, id: int):
        """creates a tensor of all features in complex (pocket AND ligand)

        Parameters
        ----------
        id   : str
              id of a complex

        Returns
        -------
        tensor : torch.tensor [1, n, 23]
            The tensor of all n atoms' features:
            1 | 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 - pocket
            -1 | 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 - ligand

        """
        elem_pocket, elem_ligand = self._get_elems(id)
        coord_pocket, coord_ligand = self._get_coord(id)
        features_pocket_part = self._get_features_unit(elem_pocket, "pocket")
        features_ligand_part = self._get_features_unit(elem_ligand, "ligand")
        features_all = features_pocket_part + features_ligand_part
        tensor_all_features = (
            torch.tensor(features_all, dtype=torch.float32)
            .type("torch.FloatTensor")
            .unsqueeze(0)
        )
        length_padding = LEN_PADDING - tensor_all_features.shape[1]
        result = F.pad(
            input=tensor_all_features,
            pad=(0, 0, 0, length_padding),
            mode="constant",
            value=0,
        )

        return result

    def _get_geometry_complex(self, id: int):
        """creates a tensor of all geometries (coordinates) in complex (pocket AND ligand)

        Parameters
        ----------
        id   : str
            id of a complex

        Returns
        -------
        tensor_all_atoms_coords : torch.tensor [1, n, 3]
            The tensor of coords-tensors
        """
        coords_pocket, coords_ligand = self._get_coord(id)
        list_geom_tensors = []
        all_atoms_coords = np.concatenate((coords_pocket, coords_ligand))
        tensor_all_atoms_coords = (
            torch.from_numpy(all_atoms_coords).squeeze().unsqueeze(0)
        )
        length_padding = LEN_PADDING - tensor_all_atoms_coords.shape[1]
        result = F.pad(
            input=tensor_all_atoms_coords,
            pad=(0, 0, 0, length_padding),
            mode="constant",
            value=0,
        )
        return result

    def read_labels(self, path: str):
        # labels = np.loadtxt(path, delimiter='\n', unpack=True)
        file = open(path, "r")
        labels = [float(line.split(",")[1][:-1]) for line in file.readlines()]
        # labels = np.asarray(labels)
        file.close()
        return labels

    def _get_labels_refined_core(self, path_refined: str, path_core: str):
        """ gives list of labels of refined and core datasets

        Parameters
        ----------
        path_refined   : str
                      path to the refined pdbbind dataset
        path_core      : str
                      path to the core pdbbind (CASF) dataset
        """
        file_lb_refined = open(path_refined, "r")
        labels_refined = [float(line.split(",")[1][:-1])
                          for line in file_lb_refined.readlines()]
        # labels = np.asarray(labels)
        file_lb_refined.close()
        file_lb_core = open(path_core, "r")
        labels_core = [float(line.split(",")[1][:-1])
                       for line in file_lb_core.readlines()]
        # labels = np.asarray(labels)
        file_lb_core.close()
        return labels_refined + labels_core


    def _get_coord(self, protein_id: int):
        """ gives np.array of coordinates for a pocket and a ligand in one complex

        Parameters
        ----------
        protein_id   : str
                      id of a complex
        """

        path_pocket, path_ligand = self._get_path(protein_id)
        mol_pocket = Molecule(path_pocket)
        mol_ligand = Molecule(path_ligand)
        return mol_pocket.coords, mol_ligand.coords


class Loss(nn.Module):
    """
    MSELoss for rmsd_min / rmsd_ave and PoissonNLLLoss for n_rmsd
    """

    def __init__(self):
        super(Loss, self).__init__()
        self.loss_rmsd_pkd = nn.MSELoss()

    def forward(self, out1, pkd_mask):
        loss_rmsd_pkd = self.loss_rmsd_pkd(out1.double(), pkd_mask).double()
        return loss_rmsd_pkd
