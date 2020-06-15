import numpy as np

# dictionary of unique atoms in PDBBINDING database of pockets/ligands
# {atom: hot_vecctor}

atom_type = {'c': 0, 
             'n': 1}

atom_most_common = {"C": 0,
                    "H": 1,
                    "N": 2,
                    "O": 3,
                    "S": 4
                      }


dict_atoms_simple = {"C": 1,"H": 2,"N": 3, "O": 4, "S": 5, "P": 6, "Zn": 7, "Cl": 8, "F": 9, "Mg": 10, "Ca": 11, "Na": 12, "Mn": 13, "I": 14,"Br": 15,"Fe": 16, "Cu": 17, "Cd": 18, "Ni": 19, "Co": 20, "Hg": 21, "K": 22}     

dict_atoms_mass = {}
dict_atoms_hot = {}


