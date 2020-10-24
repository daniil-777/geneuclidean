import argparse
import json
import os
import pickle
import sys
from shutil import copyfile
import matplotlib.pyplot as plt
import numpy as np
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
save_dir_smiles = configuration["sampling_params"]["save_dir_smiles"]

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

with open(vocab_path, "rb") as f:
    vocab = pickle.load(f)

print("vocab", vocab)

dataset = Pdb_Dataset(configuration, vocab)



def load_pocket(id_protein, transform=None):
    print("loading data of a protein", dataset._get_name_protein(id_protein))
    features = dataset._get_features_complex(id_protein)
    # print("features shape", features.shape)
    # print("features", features)
    geometry = dataset._get_geometry_complex(id_protein)
    # print("geomrery shape", geometry.shape)
    # print("geometry", geometry)
    return features, geometry

def printing_smiles(sampled_ids, smiles_file_write, text_file_all):
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == "<end>":
            break
    sentence = "".join(sampled_caption)
    sentence = sentence[7:-5]
    print(sentence)
    print("length sent", len(sentence))
    m = Chem.MolFromSmiles(sentence)
    print("m", m)
    # if m is None or sentence == '':
    if m is None or sentence == '' or sentence.isspace() == True:
        print('invalid')
        return 0
    else:
        print(sentence)
        text_file_all.write(sentence + "\n")
        text_file_all.flush()
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

def generate_smiles(id, id_fold):
    save_dir_smiles = configuration["sampling_params"]["save_dir_smiles"]
    if not os.path.exists(save_dir_smiles):
        os.makedirs(save_dir_smiles)
    protein_name =  dataset._get_name_protein(id)
    if not os.path.exists(os.path.join(save_dir_smiles, str(id_fold), protein_name)):
        os.makedirs(os.path.join(save_dir_smiles, str(id_fold), protein_name))
    
    print("current protein ", protein_name)
    init_path_smile =  os.path.join(
            caption_path, protein_name, protein_name + "_ligand.smi"
        )
    
    # path_initial_ligand_destination = os.path.join(save_dir_smiles, protein_name, protein_name + "_true_ligand.smi")
    # copyfile(init_path_smile,
    #         path_initial_ligand_destination,
    #         )
    #file to write all generated smiles for a given protein
    file_freq = open(os.path.join(save_dir_smiles, str(id_fold), str(fold) + "_freq.txt"), "w")
    file_smiles = open(os.path.join(save_dir_smiles, str(id_fold), protein_name, protein_name + ".txt"), "w")

    with open(init_path_smile) as fp: 
        initial_smile = fp.readlines()[0]
        file_smiles.write(initial_smile + "\n")
        file_smiles.flush()
    
    amount_val_smiles = 0
    iter = 0
    while (amount_val_smiles < 5):
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

        # Generate an caption from the image
        # print("geomrery shape", geometry_tensor.shape)
        # print("feature shape", features_tensor.shape)
        feature = encoder(geometry_tensor, features_tensor)
        sampled_ids = decoder.sample_prob(feature)
        # sampled_ids = decoder.sample(feature)
        # print("sampled_ids", sampled_ids)
        sampled_ids = (
            sampled_ids[0].cpu().numpy()
        )  # (1, max_seq_length) -> (max_seq_length)
        # print("sampled_ids/ after processing", sampled_ids)
        smiles_file_path = os.path.join(save_dir_smiles, protein_name, protein_name + "_gen_" + "_ligand.smi")
        # Convert word_ids to words
        idx =  printing_smiles(sampled_ids, smiles_file_path, file_smiles)
        amount_val_smiles += idx

    file_freq.write(protein_name + "," + 4/iter + "\n")
    file_freq.flush()

    save_dir_analysis = os.path.join(save_dir_smiles, str(id_fold), protein_name)
    analysis_to_csv(protein_name, save_dir_analysis)
        
    


def analysis_all():
    num_folds = 3
    for id_fold in range(num_folds):
        file_idx = os.path.join(save_dir_folds, "test_idx_" + str(id_fold))
        with (open(file_idx, "rb")) as openfile:
            idx_proteins = pickle.load(openfile)
        for id_protein in idx_proteins:
            generate_smiles(id_protein, id_fold)

        

def main():
    # analysis_all()
    generate_smiles(3, 3)
    # files_refined =  os.listdir(caption_path)
    # files_refined.remove(".DS_Store")
    # for id_protein in range(len(files_refined)):
    #     # for id_sample in range(50):
    #     generate_smiles(id_protein)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--image', type=str, required=True, help='input image for generating caption')
    # parser.add_argument('--encoder_path', type=str, default='models/encoder-5-3000.pkl', help='path for trained encoder')
    # parser.add_argument('--decoder_path', type=str, default='models/decoder-5-3000.pkl', help='path for trained decoder')
    # parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')

    # # Model parameters (should be same as paramters in train.py)
    # parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    # parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    # parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    # args = parser.parse_args()
    main()
    # smiles_all_txt()
    # main()
