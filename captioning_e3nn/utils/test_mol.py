from moleculekit.molecule import Molecule
from moleculekit.smallmol.smallmol import SmallMol
from moleculekit.tools.atomtyper import prepareProteinForAtomtyping, getFeatures
from moleculekit.tools.voxeldescriptors import getChannels
import numpy as np

mol = Molecule('1ATL')
mol.filter('protein')
mol = prepareProteinForAtomtyping(mol, verbose = False)
array = getChannels(mol, version=2)
print("array", array[0])
answer = (array[0] > 0).astype(np.float32)
print("res - ", answer)
print("shape answer", answer.shape)