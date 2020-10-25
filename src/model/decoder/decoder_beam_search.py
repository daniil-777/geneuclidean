from functools import partial
import numpy as np
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
from se3cnn.non_linearities.rescaled_act import Softplus
from se3cnn.point.kernel import Kernel
from se3cnn.point.operations import NeighborsConvolution
from se3cnn.point.radial import CosineBasisModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_Length = 245


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, vocab_path, num_layers, beam_size):
        """Set the hyper-parameters and build the layers.
        """
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = MAX_Length
        self.init_weights()
        self.beam_size = beam_size
        self.vocab_path = vocab_path
        self.device = DEVICE
        with open(self.vocab_path, "rb") as f:
            self.vocab = pickle.load(f)


    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, features, captions, lengths):
        """Decodes shapes feature vectors and generates SMILES."""
        # print("captions shape initial", captions.shape)
        embeddings = self.embed(
            captions
        )  # shape [batch_size, padded_length, embed_size]
        # print("shape emb", embeddings.shape)
        # print("features emb", features.shape)
        embeddings = torch.cat(
            (features.unsqueeze(1), embeddings), 1
        )  # shape [batch_size, padded_length + 1, embed_size]
        # print("shape embeddings", embeddings.shape)
        packed = pack_padded_sequence(
            embeddings, lengths, batch_first=True
        )  # shape [packed_length, embed_size]
        # print("packed shape", packed.data.shape)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])  # shape [packed_length, vocab_size]
        # print("shape outputs", outputs.shape)
        return outputs

    def sample(self, features, states=None):
        """Samples SMILES tockens for given  features (Greedy search).
        """

        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids

    