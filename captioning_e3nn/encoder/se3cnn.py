
from functools import partial
import numpy as np
import torch
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

# from e3nn.rsh import spherical_harmonics_xyz
# from e3nn.non_linearities.rescaled_act import Softplus
# from e3nn.point.operations import NeighborsConvolution
# from e3nn.radial import CosineBasisModel
# from e3nn.kernel import Kernel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_Length = 245



class Encoder_se3ACN(nn.Module):
    """
    Architecture of molecular ACN model using se3 equivariant functions.
    """

    def __init__(
        self,
        device=DEVICE,
        nclouds=3, #1-3
        natoms=286,
        cloud_dim=8, # 4-96 !
        neighborradius=3,
        nffl=1,
        ffl1size=512,
        num_embeddings=6,
        emb_dim=4, #12-not so important
        cloudord=1,
        nradial=3,
        rad_neurons = 150,
        nbasis=3,
        Z=True,
    ):
        # emb_dim=4 - experimentals
        super(Encoder_se3ACN, self).__init__()
        self.num_embeddings = num_embeddings
        self.device = device
        self.natoms = natoms
        self.Z = Z  # Embedding if True, ONE-HOT if False

        self.emb_dim = emb_dim
        self.cloud_res = True
        
        self.leakyrelu = nn.LeakyReLU(0.2) # Relu
        self.relu = nn.ReLU()

        self.feature_collation = "pool"  # pool or 'sum'
        self.nffl = nffl
        self.ffl1size = ffl1size

        # Cloud specifications
        self.nclouds = nclouds
        self.cloud_order = cloudord
        self.cloud_dim = cloud_dim

        self.radial_layers = nradial
        self.sp = Softplus(beta=5)
        # self.sh = spherical_harmonics_xyz

        # Embedding
        self.emb = nn.Embedding(
            num_embeddings=self.num_embeddings, embedding_dim=self.emb_dim
        )

        # Radial Model
        self.number_of_basis = nbasis
        self.rad_neurons = rad_neurons
        self.neighbor_radius = neighborradius
        
        self.RadialModel = partial(
            CosineBasisModel,
            max_radius=self.neighbor_radius,  # radius
            number_of_basis=self.number_of_basis,  # basis
            h= self.rad_neurons,  # ff neurons
            L=self.radial_layers,  # ff layers
            act=self.sp,
        )  # activation
        # Kernel
        self.K = partial(
            Kernel,
            RadialModel=self.RadialModel,
            #  sh=self.sh,
            normalization="norm",
        )

        # Embedding
        self.clouds = nn.ModuleList()
        if self.Z:
            dim_in = self.emb_dim
        else:
            dim_in = 6  # ONE HOT VECTOR, 6 ATOMS HCONF AND PADDING = 6

        dim_out = self.cloud_dim
        Rs_in = [(dim_in, o) for o in range(1)]
        Rs_out = [(dim_out, o) for o in range(self.cloud_order)]

        for c in range(self.nclouds):
            # Cloud
            self.clouds.append(
                NeighborsConvolution(self.K, Rs_in, Rs_out, neighborradius)
            )
            Rs_in = Rs_out

        if self.cloud_res:
            cloud_out = self.cloud_dim * (self.cloud_order ** 2) * self.nclouds
        else:
            cloud_out = self.cloud_dim * (self.cloud_order ** 2)

        # Cloud residuals
        in_shape = cloud_out
        # passing molecular features after pooling through output layer
        self.e_out_1 = nn.Linear(cloud_out, cloud_out)
        self.bn_out_1 = nn.BatchNorm1d(cloud_out)

        self.e_out_2 = nn.Linear(cloud_out, 2 * cloud_out)
        self.bn_out_2 = nn.BatchNorm1d(2 * cloud_out)
        
        # Final output activation layer
        # self.layer_to_atoms = nn.Linear(
        #     ff_in_shape, natoms
        # )  # linear output layer from ff_in_shape hidden size to the number of atoms
        self.act = (
            nn.Sigmoid()
        )  # y is scaled between 0 and 1, better than ReLu of tanh for U0

    def forward(self, xyz, Z):
        # print("xyz input shape", xyz.shape)
        # print("Z input shape", Z.shape)
        # xyz -
        # Z -
        if self.Z:
            features = self.emb(Z).to(self.device)
        else:
            features = Z.to(self.device)

        xyz = xyz.to(torch.double)
        features = features.to(torch.double)
        features = features.squeeze(2)
        feature_list = []
        for _, op in enumerate(self.clouds):            
            features = op(features, xyz)
            feature_list.append(features)
        
            # self.res = nn.Linear(in_shape, in_shape)
            # features_linear = F.relu(self.res(features)) #features from linear layer operation
            # add all received features to common list
            # feature_list.append(features_linear)

        # Concatenate features from clouds

        features = (
            torch.cat(feature_list, dim=2).to(torch.double).to(self.device)
        )  # shape [batch, n_atoms, cloud_dim * nclouds] 
        #!! maybe use transformer, you have n_atoms with N features. You may define H "heads"
        # and then do Q, K, V as described in the article: https://arxiv.org/pdf/2004.08692.pdf

        # print("\nfeatures before pooling", features.shape)  # shape [batch, ]
        # Pooling: Sum/Average/pool2D
        if "sum" in self.feature_collation: #here attention!
            features = features.sum(1)
        elif "pool" in self.feature_collation:
            features = F.lp_pool2d(
                features,
                norm_type=2,
                kernel_size=(features.shape[1], 1),
                ceil_mode=False,
            )

        features = features.squeeze(1)  # shape [batch, cloud_dim * (self.cloud_order ** 2) * nclouds]

        
        features = self.leakyrelu(self.bn_out_1(self.e_out_1(features))) # shape [batch, 2 * cloud_dim * (self.cloud_order ** 2) * nclouds]
        features = self.leakyrelu(self.bn_out_2(self.e_out_2(features)))
        print("shape final features", features.shape)
        # for _, op in enumerate(self.collate):
        #     # features = F.leaky_relu(op(features))
        #     features = F.softplus(
        #         op(features)
        #     )  # shape [batch, ffl1size] running_mean should contain 1 elements not 512!!!
        # result = self.act(self.outputlayer(features)).squeeze(1)
        # features = self.layer_to_atoms(features) # if we want to mimic the image captioning with the number of pixels
        return features


class Encoder_se3ACN_Fast(nn.Module):
    """
    Architecture of molecular ACN model using se3 equivariant functions.
    """

    def __init__(
        self,
        device=DEVICE,
        nclouds=3, #1-3
        natoms=286,
        cloud_dim=8, # 4-96 !
        neighborradius=3,
        nffl=1,
        ffl1size=512,
        num_embeddings=6,
        emb_dim=4, #12-not so important
        cloudord=1,
        nradial=3,
        rad_neurons = 150,
        nbasis=3,
        Z=True,
    ):
        # emb_dim=4 - experimentals
        super(Encoder_se3ACN, self).__init__()
        self.num_embeddings = num_embeddings
        self.device = device
        self.natoms = natoms
        self.Z = Z  # Embedding if True, ONE-HOT if False

        self.emb_dim = emb_dim
        self.cloud_res = True
        
        self.leakyrelu = nn.LeakyReLU(0.2) # Relu
        self.relu = nn.ReLU()

        self.feature_collation = "pool"  # pool or 'sum'
        self.nffl = nffl
        self.ffl1size = ffl1size

        # Cloud specifications
        self.nclouds = nclouds
        self.cloud_order = cloudord
        self.cloud_dim = cloud_dim

        self.radial_layers = nradial
        self.sp = Softplus(beta=5)
        # self.sh = spherical_harmonics_xyz

        # Embedding
        self.emb = nn.Embedding(
            num_embeddings=self.num_embeddings, embedding_dim=self.emb_dim
        )

        # Radial Model
        self.number_of_basis = nbasis
        self.rad_neurons = rad_neurons
        self.neighbor_radius = neighborradius
        
        self.RadialModel = partial(
            CosineBasisModel,
            max_radius=self.neighbor_radius,  # radius
            number_of_basis=self.number_of_basis,  # basis
            h= self.rad_neurons,  # ff neurons
            L=self.radial_layers,  # ff layers
            act=self.sp,
        )  # activation
        # Kernel
        self.K = partial(
            Kernel,
            RadialModel=self.RadialModel,
            #  sh=self.sh,
            normalization="norm",
        )

        # Embedding
        self.clouds = nn.ModuleList()
        if self.Z:
            dim_in = self.emb_dim
        else:
            dim_in = 6  # ONE HOT VECTOR, 6 ATOMS HCONF AND PADDING = 6

        dim_out = self.cloud_dim
        Rs_in = [(dim_in, o) for o in range(1)]
        Rs_out = [(dim_out, o) for o in range(self.cloud_order)]

        for c in range(self.nclouds):
            # Cloud
            self.clouds.append(
                NeighborsConvolution(self.K, Rs_in, Rs_out, neighborradius)
            )
            Rs_in = Rs_out

        if self.cloud_res:
            cloud_out = self.cloud_dim * (self.cloud_order ** 2) * self.nclouds
        else:
            cloud_out = self.cloud_dim * (self.cloud_order ** 2)

        # Cloud residuals
        in_shape = cloud_out
        # passing molecular features after pooling through output layer
        self.e_out_1 = nn.Linear(cloud_out, cloud_out)
        self.bn_out_1 = nn.BatchNorm1d(cloud_out)

        self.e_out_2 = nn.Linear(cloud_out, 2 * cloud_out)
        self.bn_out_2 = nn.BatchNorm1d(2 * cloud_out)
        
        # Final output activation layer
        # self.layer_to_atoms = nn.Linear(
        #     ff_in_shape, natoms
        # )  # linear output layer from ff_in_shape hidden size to the number of atoms
        self.act = (
            nn.Sigmoid()
        )  # y is scaled between 0 and 1, better than ReLu of tanh for U0

    def forward(self, xyz, Z):
        # print("xyz input shape", xyz.shape)
        # print("Z input shape", Z.shape)
        # xyz -
        # Z -
        if self.Z:
            features = self.emb(Z).to(self.device)
        else:
            features = Z.to(self.device)

        xyz = xyz.to(torch.double)
        features = features.to(torch.double)
        features = features.squeeze(2)
        for _, op in enumerate(self.clouds):            
            new_features = op(new_features, xyz)
            
        features = features + new_features

        # Concatenate features from clouds

        # features = (
        #     torch.cat(feature_list, dim=2).to(torch.double).to(self.device)
        # )  # shape [batch, n_atoms, cloud_dim * nclouds] 
      
        # print("\nfeatures before pooling", features.shape)  # shape [batch, ]
        # Pooling: Sum/Average/pool2D
        if "sum" in self.feature_collation: #here attention!
            features = features.sum(1)
        elif "pool" in self.feature_collation:
            features = F.lp_pool2d(
                features,
                norm_type=2,
                kernel_size=(features.shape[1], 1),
                ceil_mode=False,
            )

        features = features.squeeze(1)  # shape [batch, cloud_dim * (self.cloud_order ** 2) * nclouds]
        features = self.leakyrelu(self.bn_out_1(self.e_out_1(features))) # shape [batch, 2 * cloud_dim * (self.cloud_order ** 2) * nclouds]
        features = self.leakyrelu(self.bn_out_2(self.e_out_2(features)))
        print("shape final features", features.shape)
        return features


