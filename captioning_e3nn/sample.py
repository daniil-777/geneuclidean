import argparse
import json
import os
import pickle
import sys
import rdkit
from rdkit import Chem
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from build_vocab import Vocabulary
from data_loader import Pdb_Dataset
from models import DecoderRNN, Encoder_se3ACN, MyDecoderWithAttention
from utils import Utils

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


with open(vocab_path, "rb") as f:
    vocab = pickle.load(f)

print("vocab", vocab)

dataset = Pdb_Dataset(configuration, vocab)


def load_pocket(id_protein, transform=None):
    features = dataset._get_features_complex(id_protein)
    # print("features shape", features.shape)
    # print("features", features)
    geometry = dataset._get_geometry_complex(id_protein)
    # print("geomrery shape", geometry.shape)
    # print("geometry", geometry)
    return features, geometry


def main(args):
    # Image preprocessing
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406),
    #                          (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    # with open(vocab_path, 'rb') as f:
    #     vocab = pickle.load(f)
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    # Build models
    encoder = Encoder_se3ACN().eval()  # eval mode (batchnorm uses moving mean/variance)
    # decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers)
    decoder = MyDecoderWithAttention(
        attention_dim=attention_dim,
        embed_dim=emb_dim,
        decoder_dim=decoder_dim,
        vocab_size=len(vocab),
        encoder_dim=encoder_dim,
        dropout=dropout,
    )
    encoder = encoder.to(device).double()
    decoder = decoder.to(device).double()

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))

    # # Prepare features and geometry from pocket
    features, geometry = load_pocket(sample_id)
    features_tensor = features.to(device).unsqueeze(0)
    geometry_tensor = geometry.to(device).unsqueeze(0)

    # Generate an caption from the image
    print("geomrery shape", geometry_tensor.shape)
    print("feature shape", features_tensor.shape)
    feature = encoder(geometry_tensor, features_tensor)
    sampled_ids = decoder.sample(feature, vocab)
    sampled_ids = (
        sampled_ids[0].cpu().numpy()
    )  # (1, max_seq_length) -> (max_seq_length)
    print("sampled_ids", sampled_ids)

    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == "<end>":
            break
    sentence = " ".join(sampled_caption)

    # Print out the image and the generated caption
    print(sentence)
    # image = Image.open(image)
    # plt.imshow(np.asarray(image))


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
    main(args)
