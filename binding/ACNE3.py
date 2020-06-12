import torch
import torch.nn.functional as F
import torch.nn as nn
from functools import partial

# from e3nn.rsh import spherical_harmonics_xyz
# from e3nn.non_linearities.rescaled_act import Softplus
# from e3nn.point.operations import NeighborsConvolution
# from e3nn.radial import CosineBasisModel
# from e3nn.kernel import Kernel

from se3cnn.non_linearities.rescaled_act import Softplus
from se3cnn.point.operations import NeighborsConvolution
from se3cnn.point.kernel import Kernel
from se3cnn.point.radial import CosineBasisModel

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class se3ACN(nn.Module):
    """
    Architecture of molecular ACN model using se3 equivariant functions.
    """
    def __init__(self, device=DEVICE, nclouds=2, natoms=286, cloud_dim=4, neighborradius=3,
                 nffl=1, ffl1size=512, num_embeddings = 11, emb_dim=4, cloudord=1, nradial=3, nbasis=3, Z=True):
        #emb_dim=4 - experimentals
        super(se3ACN, self).__init__()
        self.num_embeddings = num_embeddings
        self.device = device
        self.natoms = natoms
        self.Z = Z  # Embedding if True, ONE-HOT if False

        self.emb_dim = emb_dim
        self.cloud_res = True

        self.feature_collation = 'pool'  # pool or 'sum'
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
        self.emb = nn.Embedding(num_embeddings =  self.num_embeddings, embedding_dim=self.emb_dim)

        # Radial Model
        self.number_of_basis = nbasis
        self.neighbor_radius = neighborradius

        self.RadialModel = partial(CosineBasisModel,
                                   max_radius=self.neighbor_radius,         # radius
                                   number_of_basis=self.number_of_basis,    # basis
                                   h=150,                                   # ff neurons
                                   L=self.radial_layers,                    # ff layers
                                   act=self.sp)                             # activation
        # Kernel
        self.K = partial(Kernel,
                         RadialModel=self.RadialModel,
                        #  sh=self.sh,
                         normalization='norm')

        # Embedding
        self.clouds = nn.ModuleList()
        if self.Z:
            dim_in = self.emb_dim
        else:
            dim_in = 6      # ONE HOT VECTOR, 6 ATOMS HCONF AND PADDING = 6

        dim_out = self.cloud_dim
        Rs_in = [(dim_in, o) for o in range(1)]
        Rs_out = [(dim_out, o) for o in range(self.cloud_order)]

        for c in range(self.nclouds):
            # Cloud
            self.clouds.append(NeighborsConvolution(self.K, Rs_in, Rs_out, neighborradius))
            Rs_in = Rs_out

        if self.cloud_res:
            cloud_out = self.cloud_dim * (self.cloud_order ** 2) * self.nclouds
        else:
            cloud_out = self.cloud_dim * (self.cloud_order ** 2)

        # Cloud residuals
        in_shape = cloud_out

        # passing molecular features after pooling through output layer
        self.collate = nn.ModuleList()
        ff_in_shape = in_shape
        for _ in range(self.nffl):
            out_shape = self.ffl1size // (_ + 1)
            self.collate.append(nn.Linear(ff_in_shape, out_shape))
            # print("batch_norm shape", out_shape)
            self.collate.append(nn.BatchNorm1d(out_shape))
            ff_in_shape = out_shape

        # Final output activation layer
        self.outputlayer = nn.Linear(ff_in_shape, 1)
        self.act = nn.Sigmoid()  # y is scaled between 0 and 1, better than ReLu of tanh for U0

    def forward(self, xyz, Z):
        # print("xyz input shape", xyz.shape)
        # print("Z input shape", Z.shape)
        #xyz - 
        #Z - 
        if self.Z:
            features_emb = self.emb(Z).to(self.device)
        else:
            features_emb = Z.to(self.device)

        # xyz = xyz.to(torch.float64).to(self.device)
        # features = features_emb.to(torch.float64)
        xyz = xyz.to(torch.double)
        features = features_emb.to(torch.double)
        features = features.squeeze(2)
        feature_list = []
        for _, op in enumerate(self.clouds):
            # print("xyz shape!!", xyz.shape)
            # print("feature shape!!", features.shape)
            features_e3nn = op(features, xyz) #features from e3nn operation
            #self.res = nn.Linear(in_shape, in_shape) 
            # features_linear = F.relu(self.res(features)) #features from linear layer operation
            #add all received features to common list
            feature_list.append(features_e3nn)
            # feature_list.append(features_linear)

        
        # Concatenate features from clouds
        features = torch.cat(feature_list, dim=2).to(torch.float64).to(self.device)

        
        # Pooling: Sum/Average/pool2D
        if 'sum' in self.feature_collation:
            features = features.sum(1)
        elif 'pool' in self.feature_collation:
            features = F.lp_pool2d(features, norm_type=2, kernel_size=(features.shape[1], 1), ceil_mode=False)
        
        # print("features_final_shape", features.shape)
        features = features.squeeze(1)
        # print("features_final_shape squeeze", features.shape)
        for _, op in enumerate(self.collate):
            # features = F.leaky_relu(op(features))
            # print("op_features", op(features).shape)
            features = F.softplus(op(features)) #running_mean should contain 1 elements not 512!!!
        # print("shape features flat out", features.shape)
        result = self.act(self.outputlayer(features)).squeeze(1)
        # print("result shape", result.shape)
        return result
        


