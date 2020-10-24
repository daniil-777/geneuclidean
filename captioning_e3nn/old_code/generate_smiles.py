import argparse
import json
import csv
import os
import pickle
import sys
import time
from shutil import copyfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from rdkit import Chem
from build_vocab import Vocabulary
from data_loader import Pdb_Dataset
from models import DecoderRNN, Encoder_se3ACN, MyDecoderWithAttention
from utils import Utils
from Contrib.statistics import analysis_to_csv

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = os.path.realpath(os.path.dirname(__file__))


args = str(sys.argv[1])
# args = "configs/tetris_simple.json"
print(args)
# ags = "configs/tetris_simple.json"
# DATA_PATH = os.path.realpath(os.path.dirname(__file__))
DATA_PATH = "/Volumes/Ubuntu"


# utils = Utils(DATA_PATH)

# config = utils.parse_configuration(args)

with open(args) as json_file:
    configuration = json.load(json_file)


# configuration = utils.parse_configuration()

# model params
num_epochs = configuration["model_params"]["num_epochs"]
batch_size = configuration["model_params"]["batch_size"]
learning_rate = configuration["model_params"]["learning_rate"]
num_workers = configuration["model_params"]["num_workers"]
save_dir_folds = configuration["output_parameters"]["savedir"]


#sampling params
sampling = configuration["sampling_params"]["sampling"]
protein_dir = configuration["training_params"]["image_dir"]
save_dir_smiles = configuration["sampling_params"]["save_dir_smiles"]
number_smiles = configuration["sampling_params"]["number_smiles"]
time_waiting = configuration["sampling_params"]["time_waiting"]
type_fold = configuration["sampling_params"]["type_fold"]
file_folds = configuration["sampling_params"]["folds"]
name_file_stat = configuration["sampling_params"]["file_stat"]
id_fold = configuration["sampling_params"]["id_fold"]


# training params
image_dir = configuration["training_params"]["image_dir"]
caption_path = configuration["training_params"]["caption_path"]
log_step = configuration["training_params"]["log_step"]
save_step = configuration["training_params"]["save_step"]
encoder_path = configuration["training_params"]["encoder_path"]
decoder_path = configuration["training_params"]["decoder_path"]
# decoder params
embed_size = configuration["decoder_params"]["embed_size"]
hidden_size = configuration["decoder_params"]["hidden_size"]
num_layers = configuration["decoder_params"]["num_layers"]
vocab_path = configuration["preprocessing"]["vocab_path"]
sample_id = configuration["sampling_params"]["sample_id"]

# attention param
attention_dim = configuration["attention_parameters"]["attention_dim"]
emb_dim = configuration["attention_parameters"]["emb_dim"]
decoder_dim = configuration["attention_parameters"]["decoder_dim"]
encoder_dim = configuration["encoder_params"]["ffl1size"]
dropout = configuration["attention_parameters"]["dropout"]






if not os.path.exists(save_dir_smiles):
    os.makedirs(save_dir_smiles)

file_long_proteins = open(os.path.join(save_dir_smiles, "exceptions_long.txt"), "w")
file_all_stat = open(os.path.join(save_dir_smiles, "all_statistics.csv"), "w")

file_statistics = open(os.path.join(save_dir_smiles, name_file_stat), "w")
#the file of the whole stat
file_statistics.write("name,fold,type_fold, orig_smile, gen_smile, gen_NP, gen_logP,gen_sa,gen_qed,gen_weight,gen_similarity, orig_NP, orig_logP, orig_sa, orig_qed, orig_weight, frequency, sampling" + "\n")

file_statistics.flush()


with open(vocab_path, "rb") as f:
    vocab = pickle.load(f)

print("vocab", vocab)

dataset = Pdb_Dataset(configuration, vocab)



def load_pocket(id_protein, transform=None):
    print("loading data of a protein", dataset._get_name_protein(id_protein))
    features = dataset._get_features_complex(id_protein)
    geometry = dataset._get_geometry_complex(id_protein)
    return features, geometry

def printing_smiles(sampled_ids, list_smiles_all):
 
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == "<end>":
            break
    sentence = "".join(sampled_caption)
    sentence = sentence[7:-5]
    print(sentence)
    m = Chem.MolFromSmiles(sentence)
    if m is None or sentence == '' or sentence.isspace() == True:
        print('invalid')
        return 0
    else:
        print(sentence)
        # smiles.append(sentence)
        list_smiles_all.append(sentence)
        # smiles_file_all.write(sentence + "\n")
        # smiles_file_all.flush()
        # with open(smiles_file_write, 'w') as file:
        #     file.write(sentence)
        #     text_file_all.write(sentence + "\n")
        #     text_file_all.flush()
        return 1

def smiles_all_txt():
    save_dir_smiles = configuration["sampling_params"]["save_dir_smiles"]
    file_all_smiles = open(os.path.join(save_dir_smiles, "all_smiles_lig.txt"), "w")
    files_refined =  os.listdir(caption_path)
    files_refined.remove(".DS_Store")
    for protein_name in files_refined:
        init_path_smile =  os.path.join(
            caption_path, protein_name, protein_name + "_ligand.smi"
        )
        with open(init_path_smile) as fp: 
            initial_smile = fp.readlines()[0]
            file_all_smiles.write(initial_smile + "\n")
            file_all_smiles.flush()

def generate_smiles(id, id_fold, number_smiles, encoder_path, decoder_path):
    #original + gen smiles
    smiles = []
    protein_name =  dataset._get_name_protein(id)
    print("current protein ", protein_name)
    #path of the real smile
    encoder_protein_path = os.path.join(save_dir_folds, protein_name)
    if not os.path.exists(encoder_protein_path):
        os.makedirs(encoder_protein_path)

    init_path_smile =  os.path.join(
            caption_path, protein_name, protein_name + "_ligand.smi"
        )
    
    #file to write all generated smiles for a given protein
    #create a file where we will write generated smiles
    # file_smiles = open(os.path.join(save_dir_smiles, str(id_fold), protein_name, protein_name + ".txt"), "w")

    with open(init_path_smile) as fp: 
        initial_smile = fp.readlines()[0] #write a true initial smile
        # file_smiles.write(initial_smile + "\n")
        # file_smiles.flush()
    smiles.append(initial_smile)
    amount_val_smiles = 0
    
    iter = 0
    start = time.time()
    while (amount_val_smiles < number_smiles):
        end = time.time()
        print("time elapsed", end - start)
        if((end - start) > time_waiting):
            #stop generating if we wait for too long till 50 ligands
            file_long_proteins.write(protein_name + "\n") #write a protein with long time of generating
            file_long_proteins.flush()
            break
        iter += 1
    # Build models
        encoder = Encoder_se3ACN().eval()  # eval mode (batchnorm uses moving mean/variance)
        decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers)
        # decoder = MyDecoderWithAttention(
        #     attention_dim=attention_dim,
        #     embed_dim=emb_dim,
        #     decoder_dim=decoder_dim,
        #     vocab_size=len(vocab),
        #     encoder_dim=encoder_dim,
        #     dropout=dropout,
        # )
        encoder = encoder.to(device).double()
        decoder = decoder.to(device).double()

        # Load the trained model parameters
        encoder.load_state_dict(torch.load(encoder_path, map_location=torch.device('cpu')))
        decoder.load_state_dict(torch.load(decoder_path, map_location=torch.device('cpu')))

        # # Prepare features and geometry from pocket
        features, geometry = load_pocket(id)
        features_tensor = features.to(device).unsqueeze(0)
        geometry_tensor = geometry.to(device).unsqueeze(0)

        # Generate a caption from the image

        feature = encoder(geometry_tensor, features_tensor)
        print("encoded feature!!", feature)
        torch.save(feature, os.path.join(encoder_protein_path, "feature_encoder.pt"))
        if (sampling == "probabilistic"):
            sampled_ids = decoder.sample_prob(feature)
        elif ( samping == "max"):
            sampled_ids = decoder.sample(feature)
 
        # sampled_ids = decoder.sample(feature)

        sampled_ids = (
            sampled_ids[0].cpu().numpy()
        )  # (1, max_seq_length) -> (max_seq_length)
        # print("sampled_ids/ after processing", sampled_ids)
        # smiles_file_path = os.path.join(smiles, protein_name, protein_name + "_gen_" + "_ligand.smi")
        # Convert word_ids to words
        idx =  printing_smiles(sampled_ids, smiles)
        amount_val_smiles += idx
    

    
    if (amount_val_smiles > 0):
        # save_dir_analysis = os.path.join(save_dir_smiles, str(id_fold), protein_name)
        stat_protein = analysis_to_csv(smiles,  protein_name, id_fold, type_fold) #get the list of lists of statistics
        # stat_protein = np.transpose(np.vstack((stat_protein, np.asarray(amount_val_smiles * [amount_val_smiles /iter]))))
        stat_protein.append(amount_val_smiles * [amount_val_smiles /iter])
        stat_protein.append(amount_val_smiles * [sampling])
        # print("shape all_stat", all_stat.shape)

        # file_statistics.write(str(list(map(list, zip(*stat_protein)))) + "\n")
        wr = csv.writer(file_statistics)
        wr.writerows(list(map(list, zip(*stat_protein))))
        file_statistics.flush()
        # all_stat = np.vstack((all_stat, stat_protein))
        # all_stat += map(list, zip(*stat_protein))



def analysis_all():
    #for every fold takes indicies for the test, generates smiles and builds statistics
    num_folds = 3
    
  
    # all_stat = np.empty((1, 8))
    for id_fold in range(num_folds):
        file_freq = open(os.path.join(save_dir_smiles, str(id_fold), str(id_fold) + "_freq.txt"), "w")
        file_idx = os.path.join(save_dir_folds, "test_idx_" + str(id_fold))
        with (open(file_idx, "rb")) as openfile:
            idx_proteins = pickle.load(openfile)
        for id_protein in idx_proteins:
            generate_smiles(id_protein, id_fold, 
                            number_smiles, encoder_path, decoder_path)
           
    # df = pd.DataFrame(all_stat, columns = ['name', 'fold', 'logP','sa','qed','weight','similarity', 'frequency'])
    # df = pd.DataFrame(all_stat, columns = ['name', 'fold', 'logP','sa','qed','weight','similarity', 'orig_logP', 'orig_sa', 'orig_qed', 'orig_weight','frequency'])
    # df.to_csv(os.path.join(save_dir_smiles, "all_stat_new.csv"))
   
    
    
def test_analysis_all():
    #for every fold takes indicies for the test, generates smiles and builds statistics
    num_folds = 3
    encoder_path = configuration["training_params"]["encoder_path"]
    decoder_path = configuration["training_params"]["decoder_path"]
    all_stat = []
    # idx_array = [[11,12], [14, 15]]
    idx_array = [[11], [14]]
    for id_fold in range(2):
        file_freq = open(os.path.join(save_dir_smiles, str(id_fold), str(id_fold) + "_freq.txt"), "w")
        idx_proteins = idx_array[id_fold]
        for id_protein in idx_proteins:
            generate_smiles(id_protein, id_fold, number_smiles, all_stat)


    # all_stat = np.array(all_stat)
    print("shape all_stat", len(all_stat))
    print("all_stat", all_stat)
    df = pd.DataFrame(all_stat, columns = ['name', 'fold', 'logP','sa','qed','weight','similarity', 'orig_logP', 'orig_sa', 'orig_qed', 'orig_weight','frequency'])
    df.to_csv(os.path.join(save_dir_smiles, "all_stat_new.csv"))
  

def analysis_cluster():
    with (open(file_folds, "rb")) as openfile:
        idx_proteins = pickle.load(openfile)
    for id_protein in idx_proteins:
        generate_smiles(id_protein, id_fold, 
                        number_smiles, encoder_path, decoder_path)

def analysis_train_cluster():
    with (open(file_folds, "rb")) as openfile:
        idx_proteins = pickle.load(openfile)
    files_refined = os.listdir(protein_dir)
    idx_all = [i for i in range(len(files_refined) - 3)]
    #take indx of proteins in the training set
    idx_train =  np.setdiff1d(idx_all, idx_proteins)
    for id_protein in idx_train:
        generate_smiles(id_protein, id_fold, 
                        number_smiles, encoder_path, decoder_path)

def main():
    analysis_cluster()
    # analysis_all()
    # test_analysis_all()



if __name__ == "__main__":
    main()



#create a folder where we will write generated smiles
# if not os.path.exists(save_dir_smiles):
#     os.makedirs(save_dir_smiles)
# protein_name =  dataset._get_name_protein(id)
# #create a file for statistics for individual protein 
# if not os.path.exists(os.path.join(save_dir_smiles, str(id_fold), protein_name)):
#     os.makedirs(os.path.join(save_dir_smiles, str(id_fold), protein_name))
