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


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx



class ResnetPointnet(nn.Module):
   #  PointNet-based encoder network with ResNet blocks.

   # Args:
   #     c_dim (int): dimension of latent code c
   #     dim (int): input points dimension
   #     hidden_dim (int): hidden dimension of the network
   #     n_channels (int): number of planes for projection
    

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, n_channels = 4):
        super().__init__()
        self.c_dim = c_dim
        self.hidden_dim = hidden_dim   
        self.n_channels = n_channels
        
        # For grid features
        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.block_0 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_2 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_3 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)
        self.unet = Unet(hidden_dim)

        # For plane prediction
        self.fc_plane_net = FCPlanenet(n_dim=dim, n_channels=n_channels, hidden_dim=hidden_dim)
        self.fc_plane_hdim = nn.Linear(n_channels*3, hidden_dim)

        # Activation & pooling
        self.actvn = nn.ReLU()
        self.pool = maxpool
        
        is_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if is_cuda else "cpu")

    def forward(self, p):
        batch_size, T, D = p.size()
        # Grid features
        net = self.fc_pos(p)
        net = self.block_0(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)
        net = self.block_1(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)
        net = self.block_2(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)
        net = self.block_3(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)
        net = self.block_4(net) # batch_size x T x hidden_dim (T: number of sampled input points)
        return net



class Encoder_Resnet_after_se3ACN(nn.Module):
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
        nbasis=3,
        Z=True,
        lat_out = 128
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
        self.neighbor_radius = neighborradius
        
        self.RadialModel = partial(
            CosineBasisModel,
            max_radius=self.neighbor_radius,  # radius
            number_of_basis=self.number_of_basis,  # basis
            h=150,  # ff neurons
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
        self.lat_out = lat_out
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

        self.resnet_block = ResnetPointnet(cloud_out, self.lat_out)

    def forward(self, xyz, Z):
        # print("xyz input shape", xyz.shape)
        # print("Z input shape", Z.shape)
        # xyz -
        # Z -
        if self.Z:
            features_emb = self.emb(Z).to(self.device)
        else:
            features_emb = Z.to(self.device)

        xyz = xyz.to(torch.double)
        features = features_emb.to(torch.double)
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
        )  # shape [batch, n_atoms, cloud_dim * cloud_order ** 2 * nclouds]

        features = self.resnet_block(features)
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

        features = features.squeeze(1)  # shape [batch, cloud_dim * (self.cloud_order ** 2) * nclouds
        
        # features = self.leakyrelu(self.bn_out_1(self.e_out_1(features))) # shape [batch, 2 * cloud_dim * (self.cloud_order ** 2) * nclouds]
        print("shape final features", features.shape)
        return features




class Encoder_Resnet_feat_geom_se3ACN(nn.Module):
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
        nbasis=3,
        Z=True,
        lat_out = 32
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
        self.lat_out = lat_out
        # Radial Model
        self.number_of_basis = nbasis
        self.neighbor_radius = neighborradius
        
        self.RadialModel = partial(
            CosineBasisModel,
            max_radius=self.neighbor_radius,  # radius
            number_of_basis=self.number_of_basis,  # basis
            h=150,  # ff neurons
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

        self.resnet_block = ResnetPointnet(cloud_out, self.lat_out)

    def forward(self, xyz, Z):
        # print("xyz input shape", xyz.shape)
        # print("Z input shape", Z.shape)
        # xyz -
        # Z -
        if self.Z:
            features_emb = self.emb(Z).to(self.device)
        else:
            features_emb = Z.to(self.device)

        xyz = xyz.to(torch.double)
        features = features_emb.to(torch.double)
        features = features.squeeze(2)

        features_all = torch.cat([xyz, features], dim=2)
        features_all = self.resnet_block(features_all)
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
        )  # shape [batch, n_atoms, cloud_dim * cloud_order ** 2 * nclouds]

        features_out = torch.cat([features, features_all], dim=2)
        #!! maybe use transformer, you have n_atoms with N features. You may define H "heads"
        # and then do Q, K, V as described in the article: https://arxiv.org/pdf/2004.08692.pdf

        # print("\nfeatures before pooling", features.shape)  # shape [batch, ]
        # Pooling: Sum/Average/pool2D
        if "sum" in self.feature_collation: #here attention!
            features_out = features_out.sum(1)
        elif "pool" in self.feature_collation:
            features_out = F.lp_pool2d(
                features_out,
                norm_type=2,
                kernel_size=(features_out.shape[1], 1),
                ceil_mode=False,
            )

        features_out = features_out.squeeze(1)  # shape [batch, cloud_dim * (self.cloud_order ** 2) * nclouds
        
        # features = self.leakyrelu(self.bn_out_1(self.e_out_1(features))) # shape [batch, 2 * cloud_dim * (self.cloud_order ** 2) * nclouds]
        print("shape final features", features_out.shape)
        return features_out


class Encoder_Resnet_geom_se3ACN(nn.Module):
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
        nbasis=3,
        Z=True,
        lat_out = 32
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
        self.neighbor_radius = neighborradius
        
        self.RadialModel = partial(
            CosineBasisModel,
            max_radius=self.neighbor_radius,  # radius
            number_of_basis=self.number_of_basis,  # basis
            h=150,  # ff neurons
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
        self.lat_out = lat_out
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

        self.resnet_block = ResnetPointnet(cloud_out, self.lat_out)

    def forward(self, xyz, Z):
        # print("xyz input shape", xyz.shape)
        # print("Z input shape", Z.shape)
        # xyz -
        # Z -
        if self.Z:
            features_emb = self.emb(Z).to(self.device)
        else:
            features_emb = Z.to(self.device)

        xyz = xyz.to(torch.double)
        features = features_emb.to(torch.double)
        features = features.squeeze(2)

        geom_resnet = self.resnet_block(xyz)

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
        )  # shape [batch, n_atoms, cloud_dim * cloud_order ** 2 * nclouds]

        features_out = torch.cat([features, geom_resnet], dim=2)
        #!! maybe use transformer, you have n_atoms with N features. You may define H "heads"
        # and then do Q, K, V as described in the article: https://arxiv.org/pdf/2004.08692.pdf

        # print("\nfeatures before pooling", features.shape)  # shape [batch, ]
        # Pooling: Sum/Average/pool2D
        if "sum" in self.feature_collation: #here attention!
            features_out = features_out.sum(1)
        elif "pool" in self.feature_collation:
            features_out = F.lp_pool2d(
                features_out,
                norm_type=2,
                kernel_size=(features_out.shape[1], 1),
                ceil_mode=False,
            )

        features_out = features_out.squeeze(1)  # shape [batch, cloud_dim * (self.cloud_order ** 2) * nclouds
        
        # features = self.leakyrelu(self.bn_out_1(self.e_out_1(features))) # shape [batch, 2 * cloud_dim * (self.cloud_order ** 2) * nclouds]
        print("shape final features", features_out.shape)
        return features_out


class Encoder_Resnet(nn.Module):
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
        nbasis=3,
        Z=True,
        lat_out = 256
    ):
        # emb_dim=4 - experimentals
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

        # self.sh = spherical_harmonics_xyz

        # Embedding
        self.emb = nn.Embedding(
            num_embeddings=self.num_embeddings, embedding_dim=self.emb_dim
        )

        # Radial Model
        self.number_of_basis = nbasis
        self.neighbor_radius = neighborradius
        

        # Embedding
        self.clouds = nn.ModuleList()
        if self.Z:
            dim_in = self.emb_dim
        else:
            dim_in = 6  # ONE HOT VECTOR, 6 ATOMS HCONF AND PADDING = 6

        dim_out = self.cloud_dim
        self.lat_out = lat_out

        # Cloud residuals

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

        self.resnet_block = ResnetPointnet(self.emb_dim + 3, self.lat_out)

    def forward(self, xyz, Z):
        # print("xyz input shape", xyz.shape)
        # print("Z input shape", Z.shape)
        # xyz -
        # Z -
        if self.Z:
            features_emb = self.emb(Z).to(self.device)
        else:
            features_emb = Z.to(self.device)

        xyz = xyz.to(torch.double)
        features = features_emb.to(torch.double)
        features = features.squeeze(2)
        features_all = torch.cat([xyz, features], dim=2)
        features_all = self.resnet_block(features_all)
   
        # Concatenate features from clouds

        #!! maybe use transformer, you have n_atoms with N features. You may define H "heads"
        # and then do Q, K, V as described in the article: https://arxiv.org/pdf/2004.08692.pdf

        # print("\nfeatures before pooling", features.shape)  # shape [batch, ]
        # Pooling: Sum/Average/pool2D
        if "sum" in self.feature_collation: #here attention!
            features_all = features_all.sum(1)
        elif "pool" in self.feature_collation:
            features_all = F.lp_pool2d(
                features_all,
                norm_type=2,
                kernel_size=(features_all.shape[1], 1),
                ceil_mode=False,
            )

        features_all = features_all.squeeze(1)  # shape [batch, cloud_dim * (self.cloud_order ** 2) * nclouds
        
        # features = self.leakyrelu(self.bn_out_1(self.e_out_1(features))) # shape [batch, 2 * cloud_dim * (self.cloud_order ** 2) * nclouds]
        print("shape final features", features_all.shape)
        return features_all #shape [batch, lat_out]
