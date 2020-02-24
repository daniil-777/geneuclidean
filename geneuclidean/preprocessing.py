import itertools as IT
import os
import pickle
import time
from distutils.dir_util import copy_tree
from multiprocessing import Pool
from shutil import copyfile

import numpy as np
import scipy.spatial.distance as dist
from moleculekit.molecule import Molecule
from moleculekit.smallmol.smallmol import SmallMol
from openbabel import openbabel
from scipy import spatial as spatial


class Preprocessor:
    """
    Class for preprocessing
    """

    def __init__(self, init: str, target: str, presision: int, regime: str):
        self.init = init
        self.target = target
        self.files_pdb = os.listdir(self.init)
        self.files_pdb.sort()
        self.precision = presision
        self.regime = regime

    # parallel data processing

    def download_url(self):
        # Todo
        pass

    def dataset_parallel(self, agents, chunksize):
        """ Creates new dataset with protein.pdb, crystal.pdb and ligand.smile

        Parameters
        ----------
        agents   : int
                Number of Processes to parallize
        chunksize : int
                Number of items in one process
        """
        if not os.path.exists(self.target):
            os.makedirs(self.target)
        # files_pdb = os.listdir(self.init)
        # files_pdb.sort()
        with Pool(processes=agents) as pool:
            pool.map(self.refined_to_my_dataset, self.files_pdb, chunksize)

    def test_protein(self):
        for i in ["1a4k"]:
            self.pdb_to_pocket(i)

    def get_pockets_all_parallel(self, agents, chunksize):
        """ Creates new dataset with pocket.pdb

        Parameters
        ----------
        agents   : int
                Number of Processes to parallize
        chunksize : int
                Number of items in one process
        """
        with Pool(processes=agents) as pool:
            pool.map(self.pdb_to_pocket, self.files_pdb, chunksize)

    #
    def copy_all_folder(self, protein, name_folder_destination):
        path_to_exceptions = os.path.join(
            os.path.abspath(os.getcwd()), name_folder_destination
        )
        """copy folder of protein to the name_folder_destination
        """
        if not os.path.exists(path_to_exceptions):
            os.makedirs(path_to_exceptions)
        copy_tree(
            os.path.join(self.init, protein), os.path.join(path_to_exceptions, protein)
        )

    # one protein processing

    def refined_to_my_dataset(self, protein):
        if protein[0].isdigit():  # just proteins
            # create folder with pdb in my folder
            if not os.path.exists(os.path.join(self.target, protein)):
                os.makedirs(os.path.join(self.target, protein))

            try:
                # crystall = generateCrystalPacking(i) - why not?
                crystall = Molecule(protein)

                crystall.filter("protein")
                crystall.write(
                    os.path.join(self.target, protein, protein + "_crystall.pdb"),
                    type="pdb",
                )
                copyfile(
                    os.path.join(self.init, protein, protein + "_protein.pdb"),
                    os.path.join(self.target, protein, protein + "_protein.pdb"),
                )
            except RuntimeError:
                self.copy_all_folder(protein, "run_time_Molecule_new")

            try:
                smallmol = SmallMol(
                    os.path.join(self.init, protein, protein + "_ligand.mol2"),
                    removeHs=False,
                    fixHs=True,
                    force_reading=True,
                )

                sm = smallmol.toSMILES()
                with open(
                    os.path.join(self.target, protein, protein + "_ligand.smi"), "w"
                ) as txt:
                    txt.write(sm)
            except ValueError:
                self.copy_all_folder(protein, "exception_smiles_new")

    def using_cdist(self, points1, points2, cutoff):
        indices = np.where(dist.cdist(points1, points2) <= cutoff)[0]
        return indices

    def pdb_protein_to_pocket(
        self, id_pdb: str, coord_protein: np.array, center_lig: np.array, precision: int
    ):
        """Creates pocket.pdb file without exteranal library moleculekit.

        Parameters
        ----------
        id_pdb   : str id of a protein
                Protein to be processed
        coord_protein : array
                Coordinates of protein's atoms
        center_lig : array
            Geometrical center of a ligand
        precision : int
            Radios of atoms selections wrp center of ligand
        """
        path_pocket = os.path.join(self.target, id_pdb, +id_pdb + "_pocket.pdb")

        atoms_selected_indicies = self.using_cdist(coord_protein, center_lig, precision)
        file = open(path_pocket, "r")
        lines = file.readlines()
        tokens_atom = [
            l
            for l in lines
            if (
                (
                    l[0:3] in ["ATO", "HET"]
                    and int(l[7:11]) - 1 in atoms_selected_indicies
                )
                or l == "TER"
            )
        ]
        file.close()
        with open(path_pocket, "w") as f:
            f.write("MODEL        1" + "\n")
            number_atom = 1
            number_residue = 0
            for token in tokens_atom:
                if token == "TER":
                    number_residue += 1
                    f.write("TER" + "\n")
                else:
                    f.write(
                        token[0:7]
                        + "{:>4}".format(str(number_atom))
                        + token[11:73]
                        + str(number_residue)
                        + token[74:]
                    )
                    number_atom += 1
            f.write("ENDMDL" + "\n" + "END")
        return atoms_selected_indicies

    def mlkit_write_selected_atoms_to_pocket(
        self, id_pdb: str, center_lig: np.array, precision: int
    ):
        """selects atoms of "id_pdb" protein within the distance "precision" around "center_lig"

        Parameters
        ----------
        id_pdb   : str id of a protein
                Protein to be processed
        center : array
            Geometrical center of a ligand
        precision : int
            Radius of atoms selections wrp center of ligand
        """
        path_protein_source = os.path.join(
            "refined-set", id_pdb, id_pdb + "_protein.pdb"
        )
        path_pocket = os.path.join(self.target, id_pdb, id_pdb + "_pocket.pdb")
        print(path_pocket)
        mol_protein = Molecule(path_protein_source)
        mol_protein.write(
            path_pocket,
            sel="sqr(x-'{0}')+sqr(y-'{1}')+sqr(z-'{2}') <= sqr('{3}')".format(
                str(center_lig[0][0]),
                str(center_lig[0][1]),
                str(center_lig[0][2]),
                str(precision),
            ),
            type="pdb",
        )

    # does not work now
    def mlkit_2(self, id_pdb: str, name_lig: str, precision: int):
        path_pocket = os.path.join(self.target, id_pdb, id_pdb + "_pocket.pdb")
        mol_protein = Molecule(path_pocket)
        mol_protein.write(
            "pocket_mol_kit_2",
            sel="within '{0}' of name '{1}'".format(str(precision), name_lig),
        )

    def get_ligand_center(self, path_ligand):
        mol_ligand = Molecule(path_ligand)
        coor_lig = mol_ligand.coords
        center = np.mean(coor_lig, axis=0)
        center = center.reshape(1, -1)
        return center

    def get_protein_coord(self, path_pocket):
        mol_protein = Molecule(path_pocket)
        coord_protein = mol_protein.coords
        coord_protein = coord_protein[:, :, -1]
        return coord_protein

    def pdb_to_pocket(self, id_pdb: str):
        """
        Creates pocket.pdb files for every protein. Has three regimes
        """
        if id_pdb[0].isdigit():
            path_ligand = os.path.join("refined-set", id_pdb, id_pdb + "_ligand.mol2")
            path_protein = os.path.join("refined-set", id_pdb, id_pdb + "_protein.pdb")

            center_ligand = self.get_ligand_center(path_ligand)
            coord_protein = self.get_protein_coord(path_protein)

            if self.regime == "manual":
                self.pdb_protein_to_pocket(
                    id_pdb, coord_protein, center_ligand, self.precision
                )

            elif self.regime == "mlkit":
                print("start doing protein, '{0}'".format(id_pdb))
                self.mlkit_write_selected_atoms_to_pocket(
                    id_pdb, center_ligand, self.precision
                )
                print("end doing protein, '{0}'".format(id_pdb))

            elif self.regime == "mlkit2":

                self.mlkit_2(id_pdb, id_pdb, 8)
            else:
                raise ValueError("regime must be mlkit or my")


if __name__ == "__main__":
    current_path = os.path.realpath(os.path.dirname(__file__))
    process = Preprocessor(
        os.path.join(current_path, "refined-set"),
        os.path.join(current_path, "new_dataset"),
        8,
        "mlkit",
    )
    # process.dataset_parallel(5,5)
    process.get_pockets_all_parallel(5, 5)
    # process.test_protein()
