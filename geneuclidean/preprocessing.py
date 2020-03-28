import itertools as IT
import os
import pickle
import time
from distutils.dir_util import copy_tree
from shutil import copyfile
from multiprocessing import Pool
from shutil import copyfile

import numpy as np
import scipy.spatial.distance as dist
from moleculekit.molecule import Molecule
from moleculekit.smallmol.smallmol import SmallMol
from openbabel import openbabel
from scipy import spatial as spatial

# from e3nn import e3nn

class Preprocessor:
    """
    Class for preprocessing refined dataset or core dataset CASF
    """

    def __init__(self, init: str, target: str, presision: int, regime: str, flag: str):
        self.init = init
        self.target = target
        self.flag = flag  #refined with 4800 or core datasets with 200 complexes
        # self.files_pdb = os.listdir(self.init)
        # self.files_pdb = self.get_files_pdb
        self.precision = presision
        self.regime = regime
        

    # parallel data processing
    def get_files_pdb(self):
        """ Creates a list of pdb_id from pdbbind dataset (refined or core)
        """
        if(self.flag == "core"):
            files_proteins = os.listdir("CASF/protein/pdb")
            files = [file[0:4] for file in files_proteins]
            return files
        elif(self.flag =="refined"):
            files = os.listdir(self.init)
            # files = files.sort()
            return sorted(files)
        else:
            raise ValueError("flag must be refined or core")


    def _get_path_protein_init(self, pdb_id: str):
        """ Creates a path to initial protein.pdb depending on the type of a dataset (refined or core)

        Parameters
        ----------
        pdb_id : str
        """
        if(self.flag == 'core'):
            path_pdb = os.path.join(
                "CASF/protein/pdb/", pdb_id + "_protein.pdb")
            return path_pdb
        elif(self.flag == 'refined'):
            path_pdb = os.path.join(
                self.init, pdb_id, pdb_id + "_protein.pdb")
            return path_pdb
        else:
            raise ValueError("flag must be refined or core")

    def _get_path_ligand_init(self, pdb_id: str):
        """ Creates a path to initial ligand.mol2 depending on the type of a dataset (refined or core)

        Parameters
        ----------
        pdb_id : str
        """
        if(self.flag == 'core'):
            path_pdb = os.path.join(
                # "CASF/ligand/docking/decoy_mol2/", pdb_id + "_ligand.mol2")
            "CASF/ligand/ranking_scoring/crystal_mol2/", pdb_id + "_ligand.mol2")
            return path_pdb
        elif(self.flag == 'refined'):
            path_pdb = os.path.join(
                "refined-set", pdb_id, pdb_id + "_ligand.mol2")
            return path_pdb
        else:
            raise ValueError("flag must be refined or core")
    

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
        files_pdb = self.get_files_pdb()
        print(files_pdb)
        if not os.path.exists(self.target):
            os.makedirs(self.target)
        # files_pdb = os.listdir(self.init)
        # files_pdb.sort()
        for prot in files_pdb:
            self.refined_to_my_dataset(prot)
            self.pdb_to_pocket(prot)
        #bags with parallel version 
        # with Pool(processes=agents) as pool:
        #     pool.map(self.refined_to_my_dataset, files_pdb, chunksize)

    def test_protein(self):
        for i in ["1a4k"]:
            self.pdb_to_pocket(i)

    def _get_pockets_all_parallel(self, agents, chunksize):
        """ Creates new dataset with pocket.pdb

        Parameters
        ----------
        agents   : int
                Number of Processes to parallize
        chunksize : int
                Number of items in one process
        """
        files_pdb = self.get_files_pdb()

        for prot in files_pdb:
            self.pdb_to_pocket(prot)
        # with Pool(processes=agents) as pool:
        #     pool.map(self.pdb_to_pocket, self.files_pdb, chunksize)

    #
    def copy_all_folder(self, pdb_id: str, name_folder_destination):
        path_to_exceptions = os.path.join(
            os.path.abspath(os.getcwd()), name_folder_destination
        )
        """copy folder of protein to the name_folder_destination
        """
        if not os.path.exists(path_to_exceptions):
            os.makedirs(path_to_exceptions)
        init_path_protein = self._get_path_protein_init(pdb_id)
        copyfile(
            init_path_protein, os.path.join(
                path_to_exceptions, pdb_id)
        )

    # one protein processing

    def refined_to_my_dataset(self, pdb_id: str):
        if pdb_id[0].isdigit():  # just proteins
            # create folder with pdb in my folder
            if not os.path.exists(os.path.join(self.target, pdb_id)):
                os.makedirs(os.path.join(self.target, pdb_id))

            try:
                # crystall = generateCrystalPacking(i) - why not?
                crystall = Molecule(pdb_id)
                
                crystall.filter("protein")
                crystall.write(
                    os.path.join(self.target, pdb_id,
                                 pdb_id + "_crystall.pdb"),
                    type="pdb",
                )
                init_path_protein = self._get_path_protein_init(pdb_id)
                copyfile(
                    init_path_protein,
                    os.path.join(self.target, pdb_id,
                                 pdb_id + "_protein.pdb"),
                )
            except RuntimeError:
                self.copy_all_folder(pdb_id, "run_time_Molecule_new")

            try:
                smallmol = SmallMol(
                    self._get_path_ligand_init(pdb_id),
                    removeHs=False,
                    fixHs=True,
                    force_reading=True,
                )

                sm = smallmol.toSMILES()
                with open(
                    os.path.join(self.target, pdb_id,
                                 pdb_id + "_ligand.smi"), "w"
                ) as txt:
                    txt.write(sm)
            except ValueError:
                self.copy_all_folder(pdb_id, "exception_smiles_new")

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

        
       
        path_protein_source = self._get_path_protein_init(id_pdb)
        if not os.path.exists(os.path.join(self.target, id_pdb)):
            os.makedirs(os.path.join(self.target, id_pdb))
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

    def _get_ligand_center(self, path_ligand):
        mol_ligand = Molecule(path_ligand)
        coor_lig = mol_ligand.coords
        center = np.mean(coor_lig, axis=0)
        center = center.reshape(1, -1)
        return center

    def _get_protein_coord(self, path_pocket):
        mol_protein = Molecule(path_pocket)
        coord_protein = mol_protein.coords
        coord_protein = coord_protein[:, :, -1]
        return coord_protein

    def pdb_to_pocket(self, id_pdb: str):
        """
        Creates pocket.pdb files for every protein. Has three regimes
        """
        if id_pdb[0].isdigit():
            path_ligand = self._get_path_ligand_init(id_pdb)
            path_protein = self._get_path_protein_init(id_pdb)
        
            center_ligand = self._get_ligand_center(path_ligand)
            coord_protein = self._get_protein_coord(path_protein)

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
        os.path.join(current_path, "core_processed_dataset"),
        8,
        "mlkit", "core",
    )
    # process.dataset_parallel(5,5)
    process._get_pockets_all_parallel(5, 5)
    # process.test_protein()
