from moleculekit.molecule import Molecule
from moleculekit.smallmol.smallmol import SmallMol
from moleculekit.molecule import Molecule
import os
from shutil import copyfile
from distutils.dir_util import copy_tree


class Preprocessor:
    def __init__(self, init, target):
        self.init = init
        self.target = target
    #parallel data processing

    def download_url(self):
        #Todo
        pass

    def dataset_parallel(self):
        if not os.path.exists(self.target):
            os.makedirs(self.target)
        files_pdb = os.listdir(self.init)
        files_pdb.sort()
        agents = 5 #number of processes
        chunksize = 5
        with Pool(processes=agents) as pool:
            pool.map(self.refined_to_my_dataset, files_pdb, chunksize)

    #copy folder of protein to the name_folder_destination
    def copy_all_folder(self, protein, name_folder_destination):
        path_to_exceptions = os.path.join(os.path.abspath(os.getcwd()), name_folder_destination)
        if not os.path.exists(path_to_exceptions):
            os.makedirs(path_to_exceptions)
        copy_tree(os.path.join(self.init, protein), os.path.join(path_to_exceptions, protein))

    #one protein processing
    def refined_to_my_dataset(self, protein):
        if (protein[0].isdigit()): #just proteins
            # create folder with pdb in my folder
            if not os.path.exists(os.path.join(self.target, protein)):
                os.makedirs(os.path.join(self.target, protein))
            
            try:
                # crystall = generateCrystalPacking(i) - why not?
                crystall = Molecule(protein)
                crystall.filter('protein')
                crystall.write(os.path.join(self.target, protein, protein + '_crystall.pdb'), type="pdb")
                copyfile(os.path.join(self.init, protein, protein + '_protein.pdb'),
                        os.path.join(self.target, protein, protein + '_protein.pdb'))
            except RuntimeError:
                self.copy_all_folder(protein, 'run_time_Molecule_new')

            try:
                smallmol = SmallMol(os.path.join(self.init, protein, protein + '_ligand.mol2'), removeHs=False, fixHs=True,
                            force_reading=True)
                sm = smallmol.toSMILES()
                with open(os.path.join(self.target, protein, protein + '_ligand.smi'), "w") as txt:
                    txt.write(sm)
            except ValueError:
                self.copy_all_folder(protein, 'exception_smiles_new')


if __name__ == "__main__":
    current_path = os.path.abspath(os.path.dirname(__file__))
    process = Preprocessor(os.path.join(current_path,'refined-set'), os.path.join(current_path, 'new_dataset'))
    process.dataset_parallel()
