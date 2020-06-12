import os
from functools import partial
import ast
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from e3nn import SO3
# from e3nn.non_linearities.rescaled_act import relu, sigmoid
# from e3nn.kernel import Kernel
from e3nn.non_linearities import GatedBlock
from e3nn.point.operations import Convolution
from e3nn.radial import CosineBasisModel
# from e3nn.util.plot import plot_sh_signal
from moleculekit.molecule import Molecule
from moleculekit.smallmol.smallmol import SmallMol
from torch.utils.data import DataLoader, Dataset

from network_utils import Pdb_Dataset

LEN_PADDING = 286

class EuclideanNet(nn.Module):
    def __init__(self):
        super(EuclideanNet, self).__init__()
        # Radial model:  R -> R^d
        # Projection on cos^2 basis functions followed by a fully connected network
        self.RadialModel = partial(
            CosineBasisModel,
            max_radius=3.0,
            number_of_basis=3,
            h=100,
            L=1,
            act=torch.relu,
        )
        # self.features = features
        # self.geometry = geometry

        # kernel: composed on a radial part that contains the learned parameters
        #  and an angular part given by the spherical hamonics and the Clebsch-Gordan coefficients
        self.K = partial(Kernel, RadialModel=self.RadialModel)
        # self.Rs_in = [(23, 0)]  # one scalar
        self.Rs_in = [(23, 0)] # here we give a vector of 23 length - [+-1 | 00000100000000] -  (+-1) depending on a ligand/protein
        # in range(1) to narrow space of Rs_out
        self.Rs_out = [(1, l) for l in range(1)]
        self.fc1 = nn.Linear(LEN_PADDING, 30)  # 
        self.fc2 = nn.Linear(30, 10)
        self.fc3 = nn.Linear(10, 1)  # prediction of Pkdb in regression
        
        # self.fc1 = nn.Linear(LEN_PADDING, 30)  # need to clarify 150 or not
        # self.fc2 = nn.Linear(30, 10)
        # self.fc3 = nn.Linear(10, 1)  # prediction of Pkdb in regression

        self.convol = Convolution(self.K, self.Rs_in, self.Rs_out)

    def forward(self, features, geometry, num_atoms):
        """
        outputs a concatenation of fully connected layers encodings
        """
        final_batch_features = []  # here we put outputs for every batch
        for feature_, geometry_ in zip(features, geometry):
            feature_one = feature_.squeeze(1)
            hot_to_index = feature_one[:,:num_atoms, 1:]
            print("shape feature", feature_one.shape)
            geometry_one = geometry_.squeeze(1)
            features_new = self.convol(feature_one, geometry_one)
            print("shape feat", features_new.shape)
            x = features_new.squeeze()
            # print("shape x", x.shape)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = x.view(x.shape[0], -1)
            final_batch_features.append(x)
        return torch.cat(final_batch_features).squeeze(1)


class SE3Net(nn.Module):
    def __init__(self, representations):
        super().__init__()
        # print(type(representations))
        # print(representations)
        # list_representations = ast.literal_eval(representations)
        if (representations == "simple"):
            list_representations = [(23,), (2,)]
        elif (representations == "middle"):
            list_representations = [(23,), (2, 2), (4, 4), (10,)]
        elif (representations == "middle_1"):
            list_representations = [(1,), (2, 2), (4, 4), (10,)]
        elif (representations == "hard_1"):
            list_representations = [(1,), (2, 2, 2, 1), (4, 4, 4, 4), (64,)]

        representations = [[(mul, l) for l, mul in enumerate(rs)] for rs in list_representations]

        R = partial(CosineBasisModel, max_radius=3.0,
                    number_of_basis=3, h=100, L=50, act=relu)
        K = partial(Kernel, RadialModel=R)

        def make_layer(Rs_in, Rs_out):
            act = GatedBlock(Rs_out, relu, sigmoid)
            conv = Convolution(K, Rs_in, act.Rs_in)
            return torch.nn.ModuleList([conv, act])

        self.firstlayers_simple = torch.nn.ModuleList([
            make_layer([(23,0)], [(2,0)])
            # for Rs_in, Rs_out in zip(representations, representations[1:])
        ])

        self.firstlayers = torch.nn.ModuleList([
            make_layer(Rs_in, Rs_out)
            for Rs_in, Rs_out in zip(representations, representations[1:])
        ])


        # self.firstlayers = torch.nn.ModuleList([
        #     make_layer([(23, 0)], [(24, 0), (2, 1), (2, 2), (1, 3)]),
        #     make_layer([(2, 0), (2, 1), (2, 2), (1, 3)],
        #                [(4, 0), (4, 1), (4, 2), (4, 3)]),
        #     make_layer([(4, 0), (4, 1), (4, 2), (4, 3)],
        #                [(6, 0), (4, 1), (4, 2), (0, 3)]),
        #     make_layer([(6, 0), (4, 1), (4, 2), (0, 3)], [(4, 0)])])
        #     for Rs_in, Rs_out in zip(representations, representations[1:])
        # ])

        # self.lastlayers = torch.nn.Sequential(
        #     AvgSpacial(), torch.nn.Linear(list_representations[-1][0], 1))

        self.lastlayers = torch.nn.Sequential(
            AvgSpacial(), torch.nn.Linear(list_representations[-1][0], 1))
        

    def forward(self, features, geometry, num_atoms):
        final_batch_features = []
        for feature_, geometry_ in zip(features, geometry):
            print("current feature ")
            feature__embed = nn.Embedding()
            length_padding = LEN_PADDING - tensor_all_atoms_coords.shape[1]
            feature_ = F.pad(
            input=feature_,
            pad=(0, 0, 0, length_padding),
            mode="constant",
            value=99,
        )
            print(feature_.shape)
            geometry_ = F.pad(
            input=geometry_,
            pad=(0, 0, 0, length_padding),
            mode="constant",
            value=99,
        )
            for conv, act in self.firstlayers:
                # feature_one = feature_
                
                # geometry_one = geometry_
                # print("shape geomtery", geometry_one.shape)
                # feature_ = feature_[:,1:20,:]
                # geometry_ = geometry_[:, 1:20, :]
                # print("shape feature", feature_[:, 1:20, :].shape)
                print("f", feature_.shape)
                feature_ = conv(feature_, geometry_, n_norm=4)
            # features = conv(features, geometry, n_norm=4)
                feature_ = act(feature_)
                # print("uuhh", feature_.shape)
            final_batch_features.append(self.lastlayers(feature_))
        return torch.cat(final_batch_features).squeeze(1)


class AvgSpacial(torch.nn.Module):
    #like a pooling layer
    def forward(self, features):
        return features.mean(1)
