import os
from functools import partial

import numpy as np
import torch
from moleculekit.molecule import Molecule
from moleculekit.smallmol.smallmol import SmallMol
from torch import nn
from torch.utils.data import DataLoader, Dataset

# import dictionary of atoms' types and hot encoders
from data import dict_atoms

# number_atoms_unique = 22


class Pdb_Dataset(Dataset):
    """PdB binding dataset"""

    def __init__(self, path_pocket: str, path_ligand: str):
        self.labels = self.read_labels(path_pocket + "labels.csv")
        self.init_pocket = path_pocket
        self.init_ligand = path_ligand
        self.files_pdb = os.listdir(self.init_pocket)
        self.files_pdb.sort()
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
        idx_str = self.index_int_to_str(idx)
        all_features = self._get_features_complex(idx_str)
        all_geometry = self._get_geometry_complex(idx_str)
        target_pkd = np.asarray(self.labels[self.files_pdb.index(idx_str)])
        return idx, all_features, all_geometry, torch.from_numpy(target_pkd)
        # item = {
        #     "pdb_id": idx,
        #     "feature": all_features,
        #     "geometry": all_geometry,
        #     "target": self.labels[self.files_pdb.index(idx_str)],
        # }
        # return item

    def _get_path(self, protein_id: str):
        """ get a full path to pocket/ligand

        """
        path_pocket = os.path.join(
            self.init_pocket, protein_id, protein_id + "_pocket.pdb"
        )
        path_ligand = os.path.join(
            self.init_ligand, protein_id, protein_id + "_ligand.mol2"
        )
        return path_pocket, path_ligand

    def _get_elems(self, protein_id: str):
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

    def _get_coord(self, protein_id: str):
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

    def _get_features_complex(self, id: str):
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

        return tensor_all_features

    def _get_geometry_complex(self, id: str):
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
        return tensor_all_atoms_coords

    def read_labels(self, path):
        # labels = np.loadtxt(path, delimiter='\n', unpack=True)
        file = open(path, "r")
        labels = [line.split(",")[1][:-1] for line in file.readlines()]
        # labels = np.asarray(labels)
        file.close()
        return labels

    def _get_coord(self, protein_id: str):
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
        loss_rmsd_pkd = self.loss_rmsd_pkd(out1, pkd_mask)
        return loss_rmsd_pkd
