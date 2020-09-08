import os
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from moleculekit.molecule import Molecule
# from moleculekit.smallmol.smallmol import SmallMol
from torch import nn
from torch.utils.data import DataLoader, Dataset

# import dictionary of atoms' types and hot encoders
from dictionaries import atom_most_common, dict_atoms_hot, dict_atoms_simple

# number_atoms_unique = 22
LEN_PADDING = 286


class Pdb_Dataset(Dataset):
    """PdB binding dataset"""

    def __init__(self, cfg, vocab):
        """uses cfg file which is given as arg in "python train_server.py"
        """

        self.path_root = cfg['preprocessing']['path_root']
        self.init_refined = self.path_root + "/data/new_refined/"
        # self.init_refined = path_root + "/data/refined_26.05/"
        self.init_casf = self.path_root + "/data/new_core_2016/"
        # self.init_casf = path_root + "/data/core_26.05/"
        # self.labels = self.read_labels(self.path_root + "/data/labels/labels.csv")
        self.vocab = vocab
        # self.labels_all = self._get_labels_refined_core(
        #     self.path_root + "/data/labels/new_labels_core_2016.csv",
        #     self.path_root + "/data/labels/new_labels_refined.csv",
        # )
        ##################refined files###################
        self.files_refined = os.listdir(self.init_refined)
        self.files_refined.sort()
        # self.files_refined.remove(".DS_Store")
        ##################################################
        self.len_files = len(self.files_refined)
        ###################core files#####################
        self.files_core = os.listdir(self.init_casf)
        self.files_core.sort()
        ##################################################
        self.dict_atoms = dict_atoms_hot
        self.dict_atoms_simple = dict_atoms_simple
        self.dict_words = atom_most_common
        self.set_atoms = []
        self.encoding = {}
        self.label_protein = np.array([5.0])  # identification of pocketd
        self.label_ligand = np.array([-5.0])  # identification of ligand
        self.features_complexes = []  # tensors of euclidean features
        self.affinities_complexes = []  # targets
        self.common_atoms = ["C", "H", "O", "N", "S"]
        # self.type_filtering = "filtered"
        self.type_filtering = cfg['preprocessing']['selection']  # "filtered"
        self.mask = cfg['preprocessing']['mask']
        print("filtering", self.type_filtering)

    def __len__(self):
        #!!!!!!!!!!!!!!!!
        return 20
        # return len(self.files_refined) # from the lab:

    def __getitem__(self, idx: int):
        vocab = self.vocab
        all_features = self._get_features_complex(idx)

        all_geometry = self._get_geometry_complex(idx)
        # print("shape all geom", all_geometry.shape)
        caption_raw = self._get_caption(idx)
        tokens = [token for token in caption_raw]
        caption = []
        caption.append(vocab("<start>"))
        # print("caption of start", vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab("<end>"))
        target = torch.Tensor(caption)
        if(self.mask == 'True'):
            return all_features, all_geometry, mask, target
        else:
            return all_features, all_geometry, target
    
    def _get_name_protein(self, idx: int):
        name_protein = self.files_refined[idx]
        return name_protein
        
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

    # def get_caption(self, idx: int):
    #     protein_name = self.files_refined[protein_id]
    #     path_ligand_caption = os.path.join(
    #             self.init_refined, protein_name, protein_name + "_ligand.txt"
    #         )
    #     ligand_caption = loadtxt(path_ligand_caption, delimiter=",", unpack=False)
    #     return ligand_caption

    def _get_path(self, protein_id: int):
        """ get a full path to pocket/ligand

        """
        protein_name = self.files_refined[protein_id]
        # print("current protein", protein_name)
        path_pocket = os.path.join(
            self.init_refined, protein_name, protein_name + "_pocket.pdb"
        )

        return path_pocket

    def _get_elems(self, protein_id: int, type_filtering: str):
        """ gives np.array of elements for a pocket and a ligand in one complex

        Parameters
        ----------
        protein_id   : str
                      id of a complex
        """
        path_pocket = self._get_path(protein_id)
        try:
            mol_pocket = Molecule(path_pocket)
            # mol_ligand = Molecule(path_ligand)
            if type_filtering == "filtered":
                mol_pocket_element = [
                    elem
                    for elem in mol_pocket.element
                    if elem in ["C", "H", "N", "O", "S"]
                ]
                # mol_ligand_element = [elem for elem in mol_ligand.element if elem in ["C", "H", "N", "O", "S"]]
            elif type_filtering == "all":
                mol_pocket_element = mol_pocket.element
                # mol_ligand_element = mol_ligand.element

        except FileNotFoundError:

            print(protein_id, "   exception")
            path_pocket = self._get_path(2)
            mol_pocket = Molecule(path_pocket)
            mol_pocket_element = mol_pocket.element

            # print("mol_ligand_element", mol_ligand.element)

        return mol_pocket_element

    def atom_to_vector(self, elem: str):
        """ creates a hot vector of an atom

        Parameters
        ----------
        elem   : str atom element
        """
        return self.dict_words[elem]
        # return self.dict_atoms[elem]
        # return self.dict_atoms[elem]

    def coords_to_tensor(self, coords: np.array):
        """ creates a tensor of coords 

        Parameters
        ----------
        coords   : array of coords of n atoms [n, 3]
        """
        return torch.tensor(coords)

    def _get_feature_vector_atom(self, elem: str, type_atom: str, type_filtering: str):
        """creates a tensor-feature vector concatenating label of protein/ligand and hot vector

        Parameters
        ----------
        elem   : str atom element
        """
        hot_vector_atom = self.atom_to_vector(elem)

        if type_atom == "pocket":
            if type_filtering == "filtered":
                feature_vector_atom = self.dict_words[elem]
                feature_vector_atom = np.array([feature_vector_atom])
            elif type_filtering == "all":
                feature_vector_atom = np.concatenate(
                    (self.label_protein, hot_vector_atom)
                )
                feature_vector_atom = np.array([hot_vector_atom])

            # print("feat_atom", feature_vector_atom)
            # feature_vector_atom = np.concatenate((self.label_protein, hot_vector_atom))
            # feature_vector_atom = np.array([hot_vector_atom])

            # print("feat vector", feature_vector_atom)

            # print("feat_atom_lig", feature_vector_atom)
            # print("feature_lig", feature_vector_atom)
            # print("feat vector", feature_vector_atom)
        else:
            raise ValueError("type of atom should be pocket or ligand")
        # feature_tensor_atom = torch.from_numpy(feature_vector_atom)
        # return feature_tensor_atom
        # print(type(feature_vector_atom))
        return feature_vector_atom
        # return hot_vector_atom

    def _get_features_unit(
        self, elements: np.array, type_atom: str, type_filtering: str
    ):
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
            tensor_feature = self._get_feature_vector_atom(
                elem, type_atom, type_filtering
            )
            list_features_tensors.append(tensor_feature)
        # features = torch.cat(list_features_tensors, dim=-1)
        return list_features_tensors

    def _get_features_dict(self, elements: np.array, type_atom: str):
        """creates a dictionary of atoms' features of a particular bio unit (protein/ligand)
        Parameters
        ----------
        id   : str
              id of a complex

        Returns
        -------
        dict   : 'O' : torch.tensor([2,2,2,2]) - tensor.size = number of 'O' in protein, 2 - positive encoding of atom 'O' in protein
                 'Na': torch.tensor([5,5]) - tensor.size = number of 'Na' in protein, 5 - positive encoding of atom 'Na' in protein
                 'Pb': torch.tensor([-3,-3,-3,-3]) - tensor.size = number of 'Pb' in ligand, -3 - negative encoding of atom 'Pb' in ligand
                 ..................................................................................
        """

        dict_atoms_feat = {}

        return dict_atoms_feat

    def _get_features_complex(self, id: int):
        """creates a tensor of all features in complex (pocket AND ligand)

        Parameters
        ----------
        id   : str
              id of a complex

        Returns
        -------
        type_filtering: all
        tensor : torch.tensor [1, n, 23]
            The tensor of all n atoms' features:
            1 | 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 - pocket
    
        type_filtering: filtered
            The tensor of all n atoms' features:
            (atoms are encoded from 0 to 4 - ["C", "H", "O", "N", "S"] for pocket)
            1 1 2 4 4 - pocket
    
        n atoms are padded then till max_length with 10 
        """
        elem_pocket = self._get_elems(id, self.type_filtering)
        # coord_pocket, coord_ligand = self._get_coord(id)
        features_pocket_part = self._get_features_unit(
            elem_pocket, "pocket", self.type_filtering
        )
        # features_ligand_part = self._get_features_unit(elem_ligand, "ligand", self.type_filtering)
        # features_all = features_pocket_part + features_ligand_part
        tensor_all_features = torch.tensor(
            features_pocket_part, dtype=torch.long
        ).unsqueeze(0)
        length_padding = LEN_PADDING - tensor_all_features.shape[1]
        result = F.pad(
            input=tensor_all_features,
            pad=(0, 0, 0, length_padding),
            mode="constant",
            value=5,
        )
        mask_binary = torch.cat([torch.ones(tensor_all_features.shape[1]),torch.zeros(length_padding)])
        # print("feature shape")
        # print(result.shape)
        # print(result)
        result = result.squeeze(0)
        
        return result, mask_binary
        # return result, elem_pocket, elem_ligand
        # return tensor_all_features

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
        coords_pocket = self._get_coord(id, self.type_filtering)

        list_geom_tensors = []
        all_atoms_coords = np.asarray(coords_pocket)
        tensor_all_atoms_coords = (
            torch.from_numpy(all_atoms_coords).squeeze().unsqueeze(0)
        )
        length_padding = LEN_PADDING - tensor_all_atoms_coords.shape[1]
        result = F.pad(
            input=tensor_all_atoms_coords,
            pad=(0, 0, 0, length_padding),
            mode="constant",
            value=99,
        )
        # print("goemetry shape")
        # print(result.shape)
        result = result.squeeze(0)
        return result
        # return result, tensor_all_atoms_coords.shape[1]
        # return tensor_all_atoms_coords

    def _get_coord(self, protein_id: int, type_filtering: str):
        """ gives np.array of coordinates for a pocket and a ligand in one complex

        Parameters
        ----------
        protein_id   : str
                      id of a complex
        """

        path_pocket = self._get_path(protein_id)
        mol_pocket = Molecule(path_pocket)
        # print("protein coords", mol_pocket.coords)

        if type_filtering == "all":
            coords_pocket = mol_pocket.coords

        elif type_filtering == "filtered":
            prot_idxs = [
                idx
                for idx, elem in enumerate(mol_pocket.element)
                if elem in self.common_atoms
            ]
            coords_pocket = [
                element for i, element in enumerate(mol_pocket.coords) if i in prot_idxs
            ]

            # lig_idxs = [idx for idx, elem in enumerate(mol_ligand.element) if elem in self.common_atoms]
            # coords_ligand = [element for i, element in enumerate(mol_ligand.coords) if i in lig_idxs]

        # lig_idxs = [idx for idx, elem in enumerate(mol_ligand.element) if elem in ["C", "H", "N", "O", "S"]]
        # lig_coords = [element for i, element in enumerate(mol_ligand.coords) if i in lig_idxs]
        # return mol_pocket.coords, mol_ligand.coords
        return coords_pocket


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        features: torch tensor of shape (batch_size, n_atoms, hidden_dim (1)).
        geometry: torch tensor of shape (batch_size, n_atoms, 3)
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[2]), reverse=True)
    features, geometry, captions = zip(*data)

    features = torch.stack(features, 0)
    geometry = torch.stack(geometry, 0)
    # Merge images (from tuple of 3D tensor to 4D tensor).

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    # we padd smaller targets till the max length with zeros. Therefore firstly create zero tensor
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return features, geometry, targets, lengths


def get_loader(cfg, vocab, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # Pdb caption dataset
    pdb_dataset = Pdb_Dataset(cfg=cfig, vocab=vocab)

    # Data loader for PDB refined dataset
    # This will return (features, geometry, captions, lengths) for each iteration.
    # features: a tensor of shape (batch_size, n_atoms, hidden_dim (1))
    # geometry: a tensor of shape (batch_size, n_atoms, 3)
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(
        dataset=pdb_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return data_loader


if __name__ == "__main__":
    # DATA_PATH = os.path.realpath(os.path.dirname(__file__))
    DATA_PATH = "/Volumes/Ubuntu"
    featuriser = Pdb_Dataset(DATA_PATH)
    lengthes = featuriser._get_length_padding("refined")
    # plt.title("Hist of features length")
    # plt.xlabel("Length")
    # plt.ylabel("Number of PDB")
    # plt.hist(lengthes)
    # plt.show()
    # plt.savefig("Hist_lenhth_refined", dpi=150)
