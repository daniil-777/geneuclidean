import os
from distutils.dir_util import copy_tree
from functools import partial

import numpy as np
import torch
from e3nn import SO3
from e3nn.kernel import Kernel
from e3nn.non_linearities import GatedBlock
from e3nn.point.operations import Convolution
from e3nn.radial import CosineBasisModel
from e3nn.util.plot import plot_sh_signal
from moleculekit.molecule import Molecule
from moleculekit.smallmol.smallmol import SmallMol


# import dictionary of atoms' types and hot encoders
from data import dict_atoms

# number_atoms = 22


class Encoding:
    """ applies S03 euclidean encoding to pocket/ligand

    """

    def __init__(self, path_pocket: str, path_ligand: str):
        self.init_pocket = path_pocket
        self.init_ligand = path_ligand
        self.files_pdb = os.listdir(self.init_pocket)
        self.files_pdb.sort()
        self.dict_atoms = dict_atoms
        self.set_atoms = []
        self.encoding = {}
        self.label_protein = np.array([1.0])
        self.label_ligand = np.array([-1.0])
        self.features_complexes = []  # tensors of euclidean features
        self.affinities_complexes = []  # labels

    def _get_labels(self):
        """ creates labels for every complex according to the order of complexes in the folder .../refined-set

            Returns
            -------
            labels : list
                The list of complexes' labels - affinity -logKd/Ki

        """
        file = open(
            os.path.join(self.init_ligand, "index/INDEX_refined_data.2019"), "r"
        )
        lines = file.readlines()
        array_id = [line.split()[0] for line in lines if line[0].isdigit()]
        array_pkd = [line.split()[3] for line in lines if line[0].isdigit()]
        # since the order of index document and complexes in folder differ we adjust the order here
        dict_id_pkd = dict(zip(array_id, array_pkd))
        labels = [dict_id_pkd[id] for id in self.files_pdb]
        return labels

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

    def hot_encoding(self, unique_elements):
        """ updates a dictionary of {atom: hotencoder}

        Parameters
        ----------
        unique_elements : list
              The list of unique atoms in pocket/ligand

        """

        for elem in unique_elements:
            if elem not in self.set_atoms:

                # if we want 0000100000 hot-vector
                self.encoding_hot[elem] = np.eye(number_atoms)[len(self.set_atoms), :]
                # if we want 5 as encoding instead of a hot-vector
                self.encoding_simple[elem] = len(self.set_atoms)
                self.set_atoms.append(elem)

    def complex_hot_encoding(
        self, elem_pocket, elem_ligand, coord_pocket, coord_ligand
    ):
        unique_elements_pocket = np.unique(elem_pocket)
        unique_elements_ligand = np.unique(elem_ligand)
        self.hot_encoding(unique_elements_pocket)
        self.hot_encoding(unique_elements_ligand)
        # print(unique_elems)

    def _get_dict_atoms(self):
        """ writes a dict {atom: hotencoder} to a text file
            there are two formates: encoding_hot with hotvectors and encoding_simple with numbers

        Parameters
        ----------
        unique_elements : list
              The list of unique atoms in pocket/ligand

        """
        for id in self.files_pdb:
            elem_pocket, elem_ligand = self._get_elems(id)

            coord_pocket, coord_ligand = self._get_coord(id)
            self.complex_compile(elem_pocket, elem_ligand, coord_pocket, coord_ligand)
        print(len(self.set_atoms))
        with open("encoding_hot.txt", "w") as f:
            print(self.encoding_hot, file=f)

        with open("encoding_simple.txt", "w") as f:
            print(self.encoding_simple, file=f)

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
        tensor : torch.tensor [1,n,23]
            The tensor of all n atoms' features:
            1 | 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 - pocket
            -1 | 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 - ligand

        """
        elem_pocket, elem_ligand = self._get_elems(id)
        coord_pocket, coord_ligand = self._get_coord(id)
        features_pocket_part = self._get_features_unit(elem_pocket, "pocket")
        features_ligand_part = self._get_features_unit(elem_ligand, "ligand")
        features_all = features_pocket_part + features_ligand_part
        # return torch.cat(features_ligand_part, features_pocket_part)
        return (
            torch.tensor(features_all, dtype=torch.float64)
            .type("torch.FloatTensor")
            .unsqueeze(0)
        )

    def _get_geometry_complex(self, id: str):
        """creates a tensor of all geometries (coordinates) in complex (pocket AND ligand)

        Parameters
        ----------
        id   : str
            id of a complex

        Returns
        -------
        tensor_all_atoms_coords : torch.tensor
            The tensor of coords-tensors
        """
        coords_pocket, coords_ligand = self._get_coord(id)
        list_geom_tensors = []
        all_atoms_coords = np.concatenate((coords_pocket, coords_ligand))
        tensor_all_atoms_coords = (
            torch.from_numpy(all_atoms_coords).squeeze().unsqueeze(0)
        )
        return tensor_all_atoms_coords

    def euclidean_vectorisation(self, id: str):
        """creates vvectorisation for every complex

        Parameters
        ----------
        id   : str
              id of a complex
        Returns
        -------
        features_new : torch.tensor
            The tensor of encoded features
        """
        # Radial model:  R -> R^d
        # Projection on cos^2 basis functions followed by a fully connected network
        RadialModel = partial(
            CosineBasisModel,
            max_radius=3.0,
            number_of_basis=3,
            h=100,
            L=1,
            act=torch.relu,
        )

        # kernel: composed on a radial part that contains the learned parameters
        #  and an angular part given by the spherical hamonics and the Clebsch-Gordan coefficients
        K = partial(Kernel, RadialModel=RadialModel)
        # 23 since we have 1 label and 22 items for hot encoder
        Rs_in = [(23, 0)]  # one scalar
        # in range(1) to narrow space of Rs_out
        Rs_out = [(1, l) for l in range(1)]

        convol = Convolution(K, Rs_in, Rs_out)

        all_features = self._get_features_complex(id)
        all_geometry = self._get_geometry_complex(id)
        features_new = convol(all_features, all_geometry)
        return features_new

    def main_process(self):
        """applies a procedure of pocket/ligand encoding for every complex id and generates labels
        """
        self.affinities_complexes = self._get_labels()
        # test version
        for id in ["1a1e"]:
            # for id in self.files_pdb:
            feature = self.euclidean_vectorisation(id)
            self.features_complexes.append(feature)
        print(self.features_complexes[0].shape)


if __name__ == "__main__":
    current_path = os.path.realpath(os.path.dirname(__file__))
    encoding = Encoding(
        os.path.join(current_path, "new_dataset"),
        os.path.join(current_path, "refined-set"),
    )
    # if you want to get dict of atoms beforehand
    # encoding._get_dict_atoms()
    encoding.main_process()
