from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import sys, math, gzip, pickle
import os.path
from collections import namedtuple


def readNPModel(filename=os.path.join(os.path.dirname(__file__), 'publicnp.model.gz')):
  """Reads and returns the scoring model,
  which has to be passed to the scoring functions."""
  print("reading NP model ...", file=sys.stderr)
  fscore = pickle.load(gzip.open(filename))
  print("model in", file=sys.stderr)
  return fscore


def scoreMolWConfidence(mol, fscore):
  """Next to the NP Likeness Score, this function outputs a confidence value
  between 0..1 that descibes how many fragments of the tested molecule
  were found in the model data set (1: all fragments were found).

  Returns namedtuple NPLikeness(nplikeness, confidence)"""

  if mol is None:
    raise ValueError('invalid molecule')
  fp = rdMolDescriptors.GetMorganFingerprint(mol, 2)
  bits = fp.GetNonzeroElements()

  # calculating the score
  score = 0.0
  bits_found = 0
  for bit in bits:
    if bit in fscore:
      bits_found += 1
      score += fscore[bit]

  score /= float(mol.GetNumAtoms())
  confidence = float(bits_found / len(bits))

  # preventing score explosion for exotic molecules
  if score > 4:
    score = 4. + math.log10(score - 4. + 1.)
  elif score < -4:
    score = -4. - math.log10(-4. - score + 1.)
  NPLikeness = namedtuple("NPLikeness", "nplikeness,confidence")
  return NPLikeness(score, confidence)


def scoreMol(mol, fscore):
  """Calculates the Natural Product Likeness of a molecule.

  Returns the score as float in the range -5..5."""
  return scoreMolWConfidence(mol, fscore).nplikeness


# def processMols(suppl):
#   fscore = readNPModel()
#   print("calculating ...", file=sys.stderr)
#   count = {}
#   scores = []
#   n = 0
#   for i, m in enumerate(suppl):
#     if m is None:
#       continue
#     n += 1
#     score = "%.3f" % scoreMol(m, fscore)
#     scores.append(float(score))

#   return scores


def processMols(suppl):
  fscore = readNPModel()
  # print("calculating ...", file=sys.stderr)
  count = {}
  scores = []
  n = 0
  for m in suppl:
    if m is None:
      continue
    n += 1
    score = "%.3f" % scoreMol(m, fscore)
    scores.append(float(score))

  return scores
