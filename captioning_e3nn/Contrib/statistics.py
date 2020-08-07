from rdkit import RDConfig
from rdkit import DataStructs
from rdkit import RDConfig
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem, QED
from rdkit.Chem.Descriptors import qed, ExactMolWt, MolLogP
from Contrib.SA_Score import sascorer
from Contrib.NP_Score import npscorer
from Contrib.NP_Score import npscorer_my
from Contrib.NP_Score.npscorer_my import processMols
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os
import numpy as np



def similarity(smile_true, smiles_others):
    m1 = Chem.MolFromSmiles(smile_true)
    fp1 = AllChem.GetMorganFingerprint(m1,2)
    similarities = []
    for smile in smiles_others:
        m2 = Chem.MolFromSmiles(smile)
        fp2 = AllChem.GetMorganFingerprint(m2,2)
        similarity = DataStructs.DiceSimilarity(fp1,fp2)
        similarities.append(similarity)
    return similarities


def analysis_to_csv(smiles,  name_protein, id_fold, type_fold):
 
    orig_smile = smiles[0] # original smile
    gen_smiles = smiles[1:] #list of generated smiles
    length = len(gen_smiles)

    ####################################diagrams##################################
    mol_orig = Chem.MolFromSmiles(orig_smile)
    mols_gen = [Chem.MolFromSmiles(smile) for smile in gen_smiles]


    orig_logP = MolLogP(mol_orig)
    orig_sa = sascorer.calculateScore(mol_orig)
    orig_qed = qed(mol_orig)
    orig_weight = ExactMolWt(mol_orig)
    orig_NP = processMols([mol_orig])
    

    gen_logP = [MolLogP(mol) for mol in mols_gen]
    gen_sa = [sascorer.calculateScore(mol) for mol in mols_gen]
    gen_qed = [qed(mol) for mol in mols_gen]
    gen_weight = [ExactMolWt(mol) for mol in mols_gen]
    gen_NP = processMols(mols_gen)

    # suppl = Chem.SmilesMolSupplier(file_smiles, smilesColumn=0, nameColumn=1, titleLine=False)
    # scores_NP = processMols(suppl) # 1: since first one is an initial smile
    # scores_NP = processMols(mols)
    # scores_NP_orig = processMols(mols)
    
    
    # print("scoresNP!!", scores_NP)
    #####################################similarity###############################

    # sim_random = similarity(orig_smile, smiles_all)
    gen_sim = similarity(orig_smile, gen_smiles)
    # print("name_protein ", name_protein, "gen_logP ", gen_logP)

    statistics = [length * [name_protein], length * [str(id_fold)], length * [type_fold], length * [orig_smile], gen_smiles, gen_NP, gen_logP, gen_sa, gen_qed, gen_weight, gen_sim,
                  length * [orig_NP], length * [orig_logP], length * [orig_sa], length * [orig_qed], length * [orig_weight]]

    return statistics

if __name__ == "__main__":
    analysis_to_csv("10gs")



# analysis = {'logP': gen_logP, 'sa': gen_sa, 'qed': gen_qed, 'gen_weight': gen_weight,
#             'similarity': gen_sim}
# df = pd.DataFrame(data=analysis)
# name_csv = os.path.join(save_dir, "analysis_" + name_protein + ".csv")
# df.to_csv(name_csv)

# statistics = np.vstack((np.asarray(length * [name_protein]), np.asarray(length * [str(id_fold)]),
#                        np.asarray(gen_logP), np.asarray(gen_sa),
#                        np.asarray(gen_qed), np.asarray(gen_weight), np.asarray(gen_sim)))
# return map(list, zip(*statistics))


# file_smiles = os.path.join("/Volumes/Ubuntu/research_drugs/data/gen_smiles_without_at/", name_protein, name_protein + ".txt")
# save_dir = os.path.join(save_dir_smiles, str(id_fold), name_protein)
# file_smiles = os.path.join(save_dir,  name_protein + ".txt")
# file_all_smiles = "/Volumes/Ubuntu/research_drugs/data/gen_smiles_without_at/all_smiles_lig.txt"
# with open(file_smiles) as fp:
#     smiles = fp.readlines()
# with open(file_all_smiles) as fp: 
#     smiles_all = fp.readlines()