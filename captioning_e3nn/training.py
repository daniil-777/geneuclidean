import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from build_vocab import Vocabulary
from data_loader import get_loader
from models import DecoderRNN, Encoder_se3ACN, MyDecoderWithAttention
from utils import Utils

DATA_PATH = os.path.realpath(os.path.dirname(__file__))

args = str(sys.argv[1])
print(args)
# ags = "configs/tetris_simple.json"
# DATA_PATH = os.path.realpath(os.path.dirname(__file__))
# DATA_PATH = '/Volumes/Ubuntu'




with open(args) as json_file:
    configuration = json.load(json_file)

utils = Utils(configuration)
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
model_path = configuration["training_params"]["model_path"]

# decoder params
embed_size = configuration["decoder_params"]["embed_size"]
hidden_size = configuration["decoder_params"]["hidden_size"]
num_layers = configuration["decoder_params"]["num_layers"]
vocab_path = configuration["preprocessing"]["vocab_path"]

# attention param
attention_dim = configuration["attention_parameters"]["attention_dim"]
emb_dim = configuration["attention_parameters"]["emb_dim"]
decoder_dim = configuration["attention_parameters"]["decoder_dim"]
encoder_dim = configuration["encoder_params"]["ffl1size"]
dropout = configuration["attention_parameters"]["dropout"]

#output files
savedir = configuration["output_parameters"]["savedir"]
tesnorboard_path = savedir


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log_file = open(os.path.join(savedir, "log.txt"), "w")
log_file_tensor = open(os.path.join(savedir, "log_tensor.txt"), "w")
writer = SummaryWriter(tesnorboard_path)

def main():
    # Create model directory
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Image preprocessing, normalization for the pretrained resnet
    # transform = transforms.Compose([
    #     transforms.RandomCrop(.crop_size),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406),
    #                          (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    # Build data loader
    data_loader = get_loader(
        configuration,
        vocab,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    # Build the models
    encoder = Encoder_se3ACN().to(device).double()
    decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers).to(device).double()
    # decoder = (
    #     MyDecoderWithAttention(
    #         attention_dim=attention_dim,
    #         embed_dim=emb_dim,
    #         decoder_dim=decoder_dim,
    #         vocab_size=len(vocab),
    #         encoder_dim=encoder_dim,
    #         dropout=dropout,
    #     )
    #     .to(device)
    #     .double()
    # )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # params_encoder = filter(lambda p: p.requires_grad, encoder.parameters())

    caption_params = list(decoder.parameters()) + list(encoder.parameters())
    caption_optimizer = torch.optim.Adam(caption_params, lr=learning_rate)
    # optimizer = torch.optim.Adam(params, lr=.learning_rate)

    # Train the models
    total_step = len(data_loader)
    print("len of datas", total_step)
    for epoch in range(num_epochs):
        for i, (features, geometry, captions, lengths) in enumerate(data_loader):

            # Set mini-batch dataset
            # features = torch.tensor(features)
            features = features.to(device)
            # features = torch.tensor(features)
            # print("type features", type(features))
            geometry = geometry.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            # print("targets", targets)
            # Forward, backward and optimize
            feature = encoder(geometry, features)
            # lengths = torch.tensor(lengths).view(-1, 1) uncomment for attention!!!
            # print("shape lengthes", lengths.shape)
            outputs = decoder(feature, captions, lengths)
            # print("outputs", outputs)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            caption_optimizer.step()
            writer.add_scalar("training_loss", loss.item(), epoch)
            log_file_tensor.write(str(loss.item()) + "\n")
            log_file_tensor.flush()
            # Print log info
            if i % log_step == 0:
                result = "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}".format(
                    epoch, num_epochs, i, total_step, loss.item(), np.exp(loss.item())
                )
                print(result)
                log_file.write(result + "\n")
                log_file.flush()

            # loss is a real crossentropy loss
            #
            # Save the model checkpoints
            if (i + 1) % save_step == 0:
                torch.save(
                    decoder.state_dict(),
                    os.path.join(
                        model_path, "decoder-{}-{}.ckpt".format(epoch + 1, i + 1)
                    ),
                )
                torch.save(
                    encoder.state_dict(),
                    os.path.join(
                        model_path, "encoder-{}-{}.ckpt".format(epoch + 1, i + 1)
                    ),
                )


if __name__ == "__main__":
    main()
