import os
import numpy as np
import itertools as IT
import json
import os
import pandas as pd
import pickle
import time
from distutils.dir_util import copy_tree
from functools import partial
from multiprocessing import Pool
from shutil import copyfile

import _pickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as dist
import torch
from e3nn import SO3
from e3nn.kernel import Kernel
from e3nn.non_linearities import GatedBlock
from e3nn.non_linearities.rescaled_act import sigmoid, swish
from e3nn.point.operations import Convolution
from e3nn.radial import CosineBasisModel
from e3nn.util.plot import plot_sh_signal
from moleculekit.molecule import Molecule
from moleculekit.smallmol.smallmol import SmallMol
from openbabel import openbabel
from scipy import spatial as spatial

number_atoms = 22


class Utils:
    def __init__(self, path_pocket: str, path_ligand: str):
        self.init_pocket = path_pocket
        self.init_ligand = path_ligand
        self.files_pdb = os.listdir(self.init_pocket)
        self.files_pdb.sort()
        self.set_atoms = []
        self.encoding_hot = {}
        self.encoding_simple = {}

    def _get_path(self, protein_id):

        path_pocket = os.path.join(
            self.init_pocket, protein_id, protein_id + "_pocket.pdb"
        )
        path_ligand = os.path.join(
            self.init_ligand, protein_id, protein_id + "_ligand.mol2"
        )
        return path_pocket, path_ligand

    def _get_elems(self, protein_id):
        path_pocket, path_ligand = self._get_path(protein_id)
        mol_pocket = Molecule(path_pocket)
        mol_ligand = Molecule(path_ligand)
        return mol_pocket.element, mol_ligand.element

    def _get_coord(self, protein_id):
        path_pocket, path_ligand = self._get_path(protein_id)
        mol_pocket = Molecule(path_pocket)
        mol_ligand = Molecule(path_ligand)
        return mol_pocket.coords, mol_ligand.coords

    def hot_encoding(self, unique_elements):
        for elem in unique_elements:
            if elem not in self.set_atoms:

                # if we want 0000100000
                self.encoding_hot[elem] = np.eye(
                    number_atoms)[len(self.set_atoms), :]
                # if we want 5
                self.encoding_simple[elem] = len(self.set_atoms)
                self.set_atoms.append(elem)

    def complex_compile(self, elem_pocket, elem_ligand, coord_pocket, coord_ligand):
        unique_elements_pocket = np.unique(elem_pocket)
        unique_elements_ligand = np.unique(elem_ligand)
        self.hot_encoding(unique_elements_pocket)
        self.hot_encoding(unique_elements_ligand)
        # print(unique_elems)

    def main_process(self):
        # for id in ["1a1e", "1a4k", "2cn0"]:
        for id in self.files_pdb:
            elem_pocket, elem_ligand = self._get_elems(id)
            # print(elem_ligand)
            coord_pocket, coord_ligand = self._get_coord(id)
            self.complex_compile(elem_pocket, elem_ligand,
                                 coord_pocket, coord_ligand)
        print(len(self.set_atoms))
        with open("encoding_hot.txt", "w") as f:
            print(self.encoding_hot, file=f)

        with open("encoding_simple.txt", "w") as f:
            print(self.encoding_simple, file=f)


    def get_labels(self):
        """ creates labels for every complex according to the order of complexes in the folder .../refined-set

                Returns
                -------
                labels : list
                    The list of complexes' labels - affinity -logKd/Ki

            """
        file = open(
            os.path.join(self.init_ligand, 
                        "index/INDEX_refined_data.2019"), "r"
        )
        lines = file.readlines()
        array_id = [line.split()[0] for line in lines if line[0].isdigit()]
        array_pkd = [line.split()[3] for line in lines if line[0].isdigit()]
        # since the order of index document and complexes in folder differ we adjust the order here
        dict_id_pkd = dict(zip(array_id, array_pkd))
        labels = np.asarray([dict_id_pkd[id] for id in self.files_pdb])
        id_all = np.asarray(self.files_pdb)
        with open ("labels.csv", "w") as f:
            # f = open(".csv", "w")
            # f.write("{},{}\n".format("Name1", "Name2"))
            for x in zip(id_all, labels):
                f.write("{},{}\n".format(x[0], x[1]))
            f.close()
        print(labels[self.files_pdb.index('1a69')])
        # print(self.files_pdb)
        # print(self.files_pdb.index('1a69'))
        # print(id_all)
        # np.savetxt('labels.csv', zip(labels, id_all), delimiter=',', fmt='%f')
        # for id in id_all:

        # labels = np.asarray(labels)
        # np.savetxt('labels.txt', labels, delimiter=',', fmt='%s')
        # return labels


if __name__ == "__main__":
    current_path = os.path.realpath(os.path.dirname(__file__))
    path_to_pdb_protein = os.path.join(current_path, "new_dataset/")
    path_to_pdb_ligand = os.path.join(current_path, "refined-set/")
    utils = Utils(
        path_to_pdb_protein,
        path_to_pdb_ligand
    )
    utils.get_labels()
    path_labels = os.path.join(current_path, "labels.csv")
    # labels = np.load(os.path.join(current_path, "labels.txt"))
    # labels = np.loadtxt(path_labels, delimiter='\n', unpack=True)
    file = open(path_labels, 'r')
    labels = [line.split(',')[1][:-1] for line in file.readlines()]
    file.close()
    df = pd.read_csv(path_labels)
    df_temp = df.iloc[:, 1].tolist()
    # print(df_temp)
    # print(labels)
    
    # utils.main_process()
    

 
