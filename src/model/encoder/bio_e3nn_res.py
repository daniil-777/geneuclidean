from functools import partial

import torch
from torch import nn as nn
from e3nn.point.kernelconv import KernelConv
from e3nn.radial import CosineBasisModel, GaussianRadialModel, BesselRadialModel
from e3nn.non_linearities import rescaled_act
from e3nn.non_linearities.gated_block import GatedBlock
from e3nn.rsh import spherical_harmonics_xyz
from src.model.encoder.base import Aggregate
import torch.nn.functional as F
import ast
from src.model.encoder.bio_e3nn import Bio_All_Network

CUSTOM_BACKWARD = False



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_kernel_conv(cutoff, n_bases, n_neurons, n_layers, act, radial_model):
    if radial_model == "cosine":
        RadialModel = partial(
            CosineBasisModel,
            max_radius=cutoff,
            number_of_basis=n_bases,
            h=n_neurons,
            L=n_layers,
            act=act
        )
    elif radial_model == "gaussian":
        RadialModel = partial(
            GaussianRadialModel,
            max_radius=cutoff,
            number_of_basis=n_bases,
            h=n_neurons,
            L=n_layers,
            act=act
        )
    elif radial_model == "bessel":
        RadialModel = partial(
            BesselRadialModel,
            max_radius=cutoff,
            number_of_basis=n_bases,
            h=n_neurons,
            L=n_layers,
            act=act
        )
    else:
        raise ValueError("radial_model must be either cosine or gaussian")
    K = partial(KernelConv, RadialModel=RadialModel)
    return K


def constants(geometry, mask):
    rb = geometry.unsqueeze(1)  # [batch, 1, b, xyz]
    ra = geometry.unsqueeze(2)  # [batch, a, 1, xyz]
    diff_geo = (rb - ra).double().detach()
    radii = diff_geo.norm(2, dim=-1).detach()
    return mask, diff_geo, radii


class ResNet_Out_Local_Network(Bio_All_Network):
    def __init__(self,  natoms, encoding, max_rad, num_basis, n_neurons, n_layers, beta, rad_model, num_embeddings,
                 embed,   scalar_act_name, gate_act_name,  list_harm, aggregation_mode, fc_sizes):
        super(ResNet_Out_Local_Network, self).__init__(natoms, encoding, max_rad, num_basis, n_neurons, n_layers, beta, rad_model, num_embeddings,
                 embed,   scalar_act_name, gate_act_name,  list_harm, aggregation_mode, fc_sizes)
        self.size_out_harm = self.Rs[-1][0][0]
        self.resnet_out_fc = nn.Linear(self.size_out_harm, self.size_out_harm)

    def resnet_out_block(self, features):
        features_out = self.resnet_out_fc(features)
        features = features + features_out
        return features
 
    def forward(self, features, geometry, mask):
        features = self.e3nn_block(features, geometry, mask)
        features = self.resnet_out_block(features)
        features = self.fc_output(features, mask)
        return features # shape ? 