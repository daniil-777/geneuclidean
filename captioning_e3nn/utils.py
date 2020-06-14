import itertools as IT
import json
import os
import pickle
import time
from distutils.dir_util import copy_tree
from functools import partial
from multiprocessing import Pool
from shutil import copyfile

import _pickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial.distance as dist
import torch

# import pyplot
# from e3nn import SO3
from e3nn.kernel import Kernel
from e3nn.non_linearities import GatedBlock
from e3nn.non_linearities.rescaled_act import sigmoid, swish
from e3nn.point.operations import Convolution
from e3nn.radial import CosineBasisModel
from e3nn.util.plot import plot_sh_signal
from matplotlib import pyplot as plt
from moleculekit.molecule import Molecule
from moleculekit.smallmol.smallmol import SmallMol
from numpy import mean, std
from openbabel import openbabel
from scipy import spatial as spatial
from scipy.stats import pearsonr

number_atoms = 22


class Utils:
    # def __init__(self, path_pocket: str, path_ligand: str):
    def __init__(self, config):

        self.path_root = config["preprocessing"]["path_root"]
        self.init_refined = self.path_root + "/data/new_refined/"

        # self.init_refined = path_root + "/data/refined_26.05/"
        # self.init_refined = path_root + "/data/refined_26.05/"
        # self.init_casf = path_root + "/core_processed_dataset"
        self.init_casf = self.path_root + "/data/new_core_2016"
        # self.init_casf = path_root + "/data/core_26.05"
        # self.init_test_data = path_root + "/CASF/PDBbind_core_set_v2007.2.lst"
        self.core_labels_data = (
            self.path_root + "/data/CASF-2016/power_scoring/CoreSet.dat"
        )
        self.refined_labels_data = (
            self.path_root + "/data/refined-set/index/INDEX_refined_data.2019"
        )

        # self.init_refined = path_pocket
        # self.init_refined = path_ligand
        self.files_refined = os.listdir(self.init_refined)
        self.files_refined.sort()
        self.files_core = os.listdir(self.init_casf)
        self.files_core.sort()
        self.set_atoms = []
        self.encoding_hot = {}
        self.encoding_simple = {}

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
        labels_refined = [
            float(line.split(",")[1][:-1]) for line in file_lb_refined.readlines()
        ]
        # labels = np.asarray(labels)
        file_lb_refined.close()
        file_lb_core = open(path_core, "r")
        labels_core = [
            float(line.split(",")[1][:-1]) for line in file_lb_core.readlines()
        ]
        # labels = np.asarray(labels)
        file_lb_core.close()
        return labels_refined + labels_core

    def plot_losses(self, path: str, path_dest: str, n_epoches: int, name: str):
        """plots losses at the train"

        """
        loss = np.loadtxt(path + "/losses_train_2016.out")
        epoches = np.arange(1, n_epoches + 1, 1)
        plt.title(name + " Train Loss of  pdb")
        plt.xlabel("epoches")
        plt.ylabel("loss")
        plt.plot(epoches, loss)
        plt.savefig(path_dest + name + "_loss", dpi=150)
        plt.clf()

    def plot_statistics(
        self,
        path: str,
        path_dest: str,
        n_epoches: int,
        name: str,
        flag: str,
        loss_train,
        loss_test,
    ):
        """plots the pred versus target pkd with Pearson correlation"

        """

        if flag == "train":
            target = np.load(path + "/target_pkd_all_train.npy")
            predict = np.load(path + "/pkd_pred_train_" + str(n_epoches - 1) + ".npy")
            corr, _ = pearsonr(target, predict)
        elif flag == "test":
            target = np.load(path + "/target_pkd_all_test.npy")
            predict = np.load(path + "/pkd_pred_test" + ".npy")
            # print(target.shape)
            # print(predict.shape)
            corr, _ = pearsonr(target, predict)
        else:
            raise NotImplementedError

        print("target_y: mean=%.3f stdv=%.3f" % (mean(target), std(target)))
        print("predicted_y: mean=%.3f stdv=%.3f" % (mean(predict), std(predict)))
        # plot
        print("Pearsons correlation: %.3f" % corr)
        plt.scatter(target, predict, label="predicted pkd & target")
        plt.xlabel("target_y")
        plt.ylabel("predicted_y")
        plt.legend(loc="upper left")
        plt.title(
            name
            + " "
            + flag
            + " Pearson correlation= "
            + r"$\bf{"
            + str(corr)[0:6]
            + "}$"
            + "\n"
            + " Loss train "
            + r"$\bf{"
            + str(loss_train)[0:6]
            + "}$"
            + " Loss test "
            + r"$\bf{"
            + str(loss_test)[0:6]
            + "}$"
            + "\n"
            + "target_y: mean="
            + r"$\bf{"
            + str(mean(target))[0:6]
            + "}$"
            + ",stdv="
            + r"$\bf{"
            + str(std(target))[0:6]
            + "}$"
            + "\n"
            + "predicted_y: mean="
            + r"$\bf{"
            + str(mean(predict))[0:6]
            + "}$"
            + ",stdv="
            + r"$\bf{"
            + str(std(predict))[0:6]
            + "}$",
            fontsize=7,
        )
        plt.savefig(path_dest + name + "_" + flag, dpi=150)
        plt.clf()

    def parse_configuration(self, config_file):
        """Loads config file if a string was passed
            and returns the input if a dictionary was passed.
        """
        print(config_file)
        with open(config_file) as json_file:
            return json.load(json_file)

    def _get_names_refined_core(self):
        return self.files_refined + self.files_core

    def _get_core_train_test(self):
        """ returns 1) indexes of pdb in refined for exception core dataset for training
            and indexes from the core dataset for the test, 2) all labels 3) all names
        """

        id_core_return = [
            self.files_refined.index(file)
            for file in self.files_refined
            if file in self.files_core
        ]

        # id_core_return = list(map(lambda x: x+id_refined[-1], id_core))
        id_refined_return = [
            self.files_refined.index(file)
            for file in self.files_refined
            if file not in self.files_core
        ]
        all_pdb_names = self.files_refined + self.files_core
        # labels_all = np.concatenate(labels_refined, labels_core)
        # print("casf_train", id_refined_return)

        return id_refined_return, id_core_return

    def _get_core_train_test_casf(self):
        """ returns 1) indexes of pdb in refined for exception core dataset for training
            and indexes from the core dataset for the test, 2) all labels 3) all names
        """
        id_refined, name_refined = self._get_refined_data()
        id_core, name_core = self._get_core_data()
        # print("id_core", id_core)
        print(len(self.files_refined))
        # id_core_return = [int(id + len(self.files_refined)) for id in id_core]
        # print("type core id", type(id_core_return[0]))
        # id_core_return = list(map(lambda x: x+id_refined[-1], id_core))
        # id_refined_return = [id for id in id_refined if id not in id_core]
        # id_refined_return = [self.files_refined.index(
        #     file) for file in self.files_refined if file not in self.files_core]
        # id_refined_in_core = [
        #     self.files_refined.index(file)
        #     for file in self.files_refined
        #     if file in self.files_core
        # ]
        list_indexes = [i for i in range(len(self.files_refined))]
        print("len ref", len(self.files_core))
        id_refined_return = [
            i for i in list_indexes if self.files_refined[i] not in self.files_core
        ]
        id_core_return = [
            i for i in list_indexes if self.files_refined[i] in self.files_core
        ]
        # len_ref = len(self.files_refined)
        # len_core = len(self.files_core)
        # id_core_return = [i for i in range(len_ref, len_ref + len_core)]
        # id_core_return = [i for i in range(4852, 5137)]

        return id_refined_return, id_core_return

    def _get_dataset_preparation(self):
        """ returns indexes of pdb in refined for exception core dataset for training
            and indexes from the core dataset for the test
        """
        id_pdb_without_core = [
            self.files_refined.index(file)
            for file in self.files_refined
            if file not in self.files_core
        ]
        id_pdb_core = [5000 + i for i in range(len(self.files_core))]
        # print(len(id_pdb_without_core))
        # print(id_pdb_core)
        return id_pdb_without_core, id_pdb_core

    def _get_core_data(self):
        """ 
        returns a list of PDB ids (list of ints) and list of pdb names (list of str)

        returns list of indexes and list of pdb-names of all complexes
        """
        list_indexes = [i for i in range(len(self.files_core))]
        return list_indexes, self.files_core

    def _get_refined_data(self):
        """ 
        returns a list of PDB ids (list of ints) and list of pdb names (list of str)

        returns list of indexes and list of pdb-names of all complexes
        """
        list_indexes = [i for i in range(len(self.files_refined))]
        return list_indexes, self.files_refined

    def _get_split(self, array_indexes, num_folds):
        """ splits a list "array_indexes" into "num_folders" chunks

        Parameters
        ----------
        array_indexes array of indexes  : list of int indexes 
        num_folders : int number of folders
        """
        num_folds += 1
        avg = len(array_indexes) / float(num_folds)
        out = []
        last = 0.0
        while last < len(array_indexes):
            out.append(array_indexes[int(last) : int(last + avg)])
            last += avg
        return out

    # split train : test as 80 : 20
    def _get_train_test_data(self, array_indexes):
        """ gives a list of indexes for a train and a test data' id from refined set in propotion 80 : 20

        Parameters
        ----------
        array_indexes array of indexes  : list
        """

        len_train_data = int(0.8 * len(array_indexes))
        indexes_train_data = array_indexes[0:len_train_data]
        indexes_test_data = array_indexes[len_train_data:]
        return indexes_train_data, indexes_test_data

    def _get_train_data(self):
        """ gives a list of first 3880 complexes' id from refined set

        Parameters
        ----------
        """
        # print(len(self.files_refined))
        array_train_id = self.files_refined[0:3880]  # array of str ids
        array_train_indexes = np.arange(0, 3881, 1)  # array of numbers-indexes
        array_train_indexes = [i for i in range(3880)]
        return array_train_indexes

    # "special" train and test getters with the use of CASF dataset
    def _get_test_data(self):
        """ gives a list of complexes' id from CASF dataser

        Parameters
        ----------
        """
        array_test_id = self.files_refined[3880:4852]  # array of str ids
        array_test_indexes = np.arange(3881, 4853, 1)  # array of numbers-indexes
        array_test_indexes = [i for i in range(3880, 4854)]
        return array_test_indexes

    def _get_train_data_special(self):
        files_refined = os.listdir(self.init_refined)
        # print(len(files_refined))

        array_test_id = self._get_test_data()
        array_train_id = [
            id for id in files_refined if id not in array_test_id
        ]  # array of str id
        # print(len(array_train_id))
        return array_train_id

    def _get_test_data_special(self):
        """ gives a list of complexes' id from CASF dataser

        Parameters
        ----------
        """
        file = open(self.init_test_data, "r")

        lines = file.readlines()
        array_test_id = [
            line.split(" ")[0] for line in lines if line[0].isdigit()
        ]  # array of str id
        file.close()
        return array_test_id

    def _get_path(self, protein_id):

        path_pocket = os.path.join(
            self.init_refined, protein_id, protein_id + "_pocket.pdb"
        )
        path_ligand = os.path.join(
            self.init_refined, protein_id, protein_id + "_ligand.mol2"
        )
        return path_pocket, path_ligand

    def _get_elems(self, protein_id):
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

    def _get_coord(self, protein_id):
        path_pocket, path_ligand = self._get_path(protein_id)
        mol_pocket = Molecule(path_pocket)
        mol_ligand = Molecule(path_ligand)
        return mol_pocket.coords, mol_ligand.coords

    def hot_encoding(self, unique_elements):
        """ updates a dictionary of {atom: hotencoder}

        Parameters
        ----------
        unique_elements : list
              The list of unique atoms in pocket/ligand

        """
        for elem in unique_elements:
            if elem not in self.set_atoms:

                # if we want 0000100000
                self.encoding_hot[elem] = np.eye(number_atoms)[len(self.set_atoms), :]
                # if we want 5
                self.encoding_simple[elem] = len(self.set_atoms)
                self.set_atoms.append(elem)

    def complex_compile(self, elem_pocket, elem_ligand, coord_pocket, coord_ligand):
        unique_elements_pocket = np.unique(elem_pocket)
        unique_elements_ligand = np.unique(elem_ligand)
        self.hot_encoding(unique_elements_pocket)
        self.hot_encoding(unique_elements_ligand)
        # print(unique_elems)

    def _get_dict(self):
        # for id in ["1a1e", "1a4k", "2cn0"]:
        for id in self.files_refined:
            elem_pocket, elem_ligand = self._get_elems(id)
            # print(elem_ligand)
            coord_pocket, coord_ligand = self._get_coord(id)
            self.complex_compile(elem_pocket, elem_ligand, coord_pocket, coord_ligand)
        # print(len(self.set_atoms))
        with open("refined_encoding_hot.txt", "w") as f:
            print(self.encoding_hot, file=f)

        with open("refined_encoding_simple.txt", "w") as f:
            print(self.encoding_simple, file=f)

    def _get_id_labels(self, path: str):
        """ read labels and id of complexes from the file

                Returns
                -------
                labels : list
                    The list of complexes' labels - affinity -logKd/Ki

        """
        file = open(os.path.join(self.init_refined, path), "r")
        lines = file.readlines()
        array_id = [line.split()[0] for line in lines if line[0].isdigit()]
        array_pkd = [line.split()[3] for line in lines if line[0].isdigit()]
        # since the order of index document and complexes in folder differ we adjust the order here
        dict_id_pkd = dict(zip(array_id, array_pkd))
        labels = np.asarray([dict_id_pkd[id] for id in self.files_refined])
        id_all = np.asarray(self.files_refined)
        file.close()
        return id_all, labels

    def write_core_labels(self):
        """ writes id - labels for every complex in the CASF dataset 

                Returns
                -------
                labels : list
                    The list of complexes' labels - affinity -logKd/Ki

            """
        file_core = open(self.core_labels_data, "r")
        lines = file_core.readlines()
        # array_id = self.files_core

        # find pdb-id in .Dat file which we have in new_core_2016 file
        array_id = [
            line.split()[0]
            for line in lines
            if line[0].isdigit() and line.split()[0] in self.files_core
        ]
        array_pkd = [
            line.split()[3]
            for line in lines
            if line[0].isdigit() and line.split()[0] in array_id
        ]
        # labels = np.asarray(array_pkd)

        # since the order of index document and complexes in folder differ we adjust the order here
        dict_id_pkd = dict(zip(array_id, array_pkd))
        labels = np.asarray([dict_id_pkd[id] for id in self.files_core])
        id_all = np.asarray(self.files_core)

        with open("data/labels/new_labels_core_2016.csv", "w") as f:
            # f = open(".csv", "w")
            # f.write("{},{}\n".format("Name1", "Name2"))
            for x in zip(id_all, labels):
                f.write("{},{}\n".format(x[0], x[1]))
            f.close()

    def write_labels(self):
        """ writes labels for every complex according to the order of complexes in the folder .../refined-set

                Returns
                -------
                labels : list
                    The list of complexes' labels - affinity -logKd/Ki

            """
        # print(len(self.files_core))
        # print(len(self.files_refined))
        file = open(self.refined_labels_data, "r")
        lines = file.readlines()
        # array_id = self.files_refined
        # array_id = [line.split()[0] for line in lines if line[0].isdigit()]
        # array_pkd = [line.split()[3] for line in lines if line[0].isdigit()
        #              and line.split()[0] in array_id]
        # labels = np.asarray(array_pkd)
        # id_all = np.asarray(array_id)
        # array_id = [line.split()[0] for line in lines if line[0].isdigit()]
        # array_pkd = [line.split()[3] for line in lines if line[0].isdigit()]
        # since the order of index document and complexes in folder differ we adjust the order here
        # dict_id_pkd = dict(zip(array_id, array_pkd))
        # labels = np.asarray([dict_id_pkd[id] for id in self.files_refined])
        # id_all = np.asarray(self.files_refined)

        array_id = [
            line.split()[0]
            for line in lines
            if line[0].isdigit() and line.split()[0] in self.files_refined
        ]
        array_pkd = [
            line.split()[3]
            for line in lines
            if line[0].isdigit() and line.split()[0] in array_id
        ]
        # labels = np.asarray(array_pkd)

        # since the order of index document and complexes in folder differ we adjust the order here
        dict_id_pkd = dict(zip(array_id, array_pkd))
        labels = np.asarray([dict_id_pkd[id] for id in self.files_refined])
        id_all = np.asarray(self.files_refined)

        with open("data/labels/new_labels_refined.csv", "w") as f:
            # f = open(".csv", "w")
            # f.write("{},{}\n".format("Name1", "Name2"))
            for x in zip(id_all, labels):
                f.write("{},{}\n".format(x[0], x[1]))
            f.close()
        # print(labels[self.files_refined.index("1a69")])
        # print(self.files_refined)
        # print(self.files_refined.index('1a69'))
        # print(id_all)
        # np.savetxt('labels.csv', zip(labels, id_all), delimiter=',', fmt='%f')
        # for id in id_all:

        # labels = np.asarray(labels)
        # np.savetxt('labels.txt', labels, delimiter=',', fmt='%s')
        # return labels


if __name__ == "__main__":
    path_root = os.path.realpath(os.path.dirname(__file__))
    current_path = os.path.realpath(os.path.dirname(__file__))
    path_to_pdb_protein = os.path.join(current_path, "new_dataset/")
    path_to_pdb_ligand = os.path.join(current_path, "refined-set/")
    utils = Utils(current_path)
    # utils._get_dict()

    utils.write_core_labels()
    utils.write_labels()
    # utils._get_train_data()
    # utils.write_core_labels()
    # labels_all = utils._get_labels_refined_core(
    #     path_root + "/data/labels.csv", path_root + "/data/labels_core2016.csv")
    # train, test = utils._get_core_train_test_casf()
    # print(labels_all[test[-1]])
    # utils._get_dataset_preparation()
    # path_labels = os.path.join(current_path, "labels.csv")
    # # labels = np.load(os.path.join(current_path, "labels.txt"))
    # # labels = np.loadtxt(path_labels, delimiter='\n', unpack=True)
    # file = open(path_labels, 'r')
    # labels = [line.split(',')[1][:-1] for line in file.readlines()]
    # file.close()
    # df = pd.read_csv(path_labels)
    # df_temp = df.iloc[:, 1].tolist()
    # print(df_temp)
    # print(labels)

    # utils.main_process()
