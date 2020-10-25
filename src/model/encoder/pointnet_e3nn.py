from functools import partial

import torch
from torch import nn as nn
from e3nn.point.kernelconv import KernelConv
from e3nn.radial import CosineBasisModel, GaussianRadialModel, BesselRadialModel
from e3nn.non_linearities import rescaled_act
from e3nn.non_linearities.gated_block import GatedBlock
from e3nn.rsh import spherical_harmonics_xyz
from model.encoder.base import Aggregate
import torch.nn.functional as F
from model.encoder.loc_resnet import ResnetPointnet, ResnetPointnet_4, ResnetBlockFC, Aggregate_Pointnet

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


class PointNetAllNetwork(torch.nn.Module):
    def __init__(self,  max_rad, num_basis, n_neurons, n_layers, beta, rad_model, num_embeddings,
                 embed, l0,   L, scalar_act_name, gate_act_name, natoms, mlp_h, Out, aggregation_mode):
        super().__init__()
        self.natoms = natoms #286
        self.ssp = rescaled_act.ShiftedSoftplus(beta = beta)
        self.sp = rescaled_act.Softplus(beta=beta)
        self.l0 = l0
        if(scalar_act_name == "sp"):
            scalar_act = self.sp
        
        if(gate_act_name == "sigmoid"):
            gate_act = rescaled_act.sigmoid
        self.embed = embed

        Rs = [[(embed, 0)]]
        Rs_mid = [(mul, l) for l, mul in enumerate([l0])]
        Rs += [Rs_mid] * L
        Rs += [[(mlp_h, 0)]] * Out
        self.Rs = Rs
        self.device = DEVICE
        if aggregation_mode == "sum":
            self.atom_pool =  Aggregate(axis=1, mean=False)
        elif aggregation_mode == "avg":
            self.atom_pool =  Aggregate(axis=1, mean=True)
        self.num_embeddings = 6
        self.RadialModel = partial(
            CosineBasisModel,
            max_radius=max_rad,
            number_of_basis=num_basis,
            h=n_neurons,
            L=n_layers,
            act=self.ssp
        )

        # kernel_conv = create_kernel_conv(max_rad, num_basis, n_neurons, n_layers, self.ssp, rad_model)
        self.kernel_conv = partial(KernelConv, RadialModel=self.RadialModel)

        def make_layer(Rs_in, Rs_out):
            act = GatedBlock(Rs_out, scalar_act, gate_act)
            kc = self.kernel_conv(Rs_in, act.Rs_in)
            return torch.nn.ModuleList([kc, act])

        self.layers = torch.nn.ModuleList([torch.nn.Embedding(self.num_embeddings, embed, padding_idx=5)])
        self.layers += [make_layer(rs_in, rs_out) for rs_in, rs_out in zip(Rs, Rs[1:])]
        self.leakyrelu = nn.LeakyReLU(0.2) # Relu
        self.e_out_1 = nn.Linear(mlp_h, mlp_h)
        self.bn_out_1 = nn.BatchNorm1d(natoms)

        self.e_out_2 = nn.Linear(mlp_h, 2 * mlp_h)
        self.bn_out_2 = nn.BatchNorm1d(natoms)
        self.resnet_block = ResnetPointnet(self.embed, 2*self.embed)
        torch.autograd.set_detect_anomaly(True) 

    def forward(self, features, geometry, mask):
        mask = mask.to(torch.double)
        mask, diff_geo, radii = constants(geometry, mask)
        embedding = self.layers[0]
        features = torch.tensor(features).to(self.device).long()
        features = embedding(features).to(self.device)
        features = features.squeeze(2)
        set_of_l_filters = self.layers[1][0].set_of_l_filters
        y = spherical_harmonics_xyz(set_of_l_filters, diff_geo)
        for kc, act in self.layers[1:]:
            if kc.set_of_l_filters != set_of_l_filters:
                set_of_l_filters = kc.set_of_l_filters
                y = spherical_harmonics_xyz(set_of_l_filters, diff_geo)
            features = features.div(self.natoms ** 0.5).to(self.device)
            features = kc(
                features,
                diff_geo,
                mask,
                y=y,
                radii=radii,
                custom_backward=CUSTOM_BACKWARD
            )
            features = act(features)
            features = features * mask.unsqueeze(-1)
        print("features shape after enc", features.shape)
        
        # out_net = OutputMLPNetwork(kernel_conv=kernel_conv, previous_Rs = self.Rs[-1],
        #                          l0 = self.l0, l1 = 0, L = 1, scalar_act=sp, gate_act=rescaled_act.sigmoid,
        #                           mlp_h = 128, mlp_L = 1, natoms = 286)
        # features = out_net(features, geometry, mask)
        features = features.to(torch.double)

        features = self.resnet_block(features, mask)
        # features = self.leakyrelu(self.bn_out_1(self.e_out_1(features))) # shape [batch, 2 * cloud_dim * (self.cloud_order ** 2) * nclouds]
        # features = self.leakyrelu(self.bn_out_2(self.e_out_2(features)))

        # if self.atomref is not None:
        #     features_z = self.atomref(atomic_numbers)
        #     features = features_z + features
        features = self.atom_pool(features, mask)
        features = features.to(torch.double)
        # features = F.lp_pool2d(features,norm_type=2,
        #         kernel_size=(features.shape[1], 1),
        #         ceil_mode=False,)
        features = features.squeeze(1)
        features = features.squeeze(0)
        print("feat final shape", features.shape)
        return features # shape ? 

class PointNet_Geo_AllNetwork(torch.nn.Module):
    def __init__(self,  max_rad, num_basis, n_neurons, n_layers, beta, rad_model, num_embeddings,
                 embed, l0,   L, scalar_act_name, gate_act_name, natoms, mlp_h, Out, aggregation_mode, geo_out):
        super().__init__()
        self.natoms = natoms #286
        self.ssp = rescaled_act.ShiftedSoftplus(beta = beta)
        self.sp = rescaled_act.Softplus(beta=beta)
        self.l0 = l0
        if(scalar_act_name == "sp"):
            scalar_act = self.sp
        
        if(gate_act_name == "sigmoid"):
            gate_act = rescaled_act.sigmoid
        self.embed = embed

        Rs = [[(embed, 0)]]
        Rs_mid = [(mul, l) for l, mul in enumerate([l0])]
        Rs += [Rs_mid] * L
        Rs += [[(mlp_h, 0)]] * Out
        self.Rs = Rs
        self.device = DEVICE
        if aggregation_mode == "sum":
            self.atom_pool =  Aggregate(axis=1, mean=False)
        elif aggregation_mode == "avg":
            self.atom_pool =  Aggregate(axis=1, mean=True)
        self.num_embeddings = 6
        self.RadialModel = partial(
            CosineBasisModel,
            max_radius=max_rad,
            number_of_basis=num_basis,
            h=n_neurons,
            L=n_layers,
            act=self.ssp
        )
        self.geo_out = geo_out
        # kernel_conv = create_kernel_conv(max_rad, num_basis, n_neurons, n_layers, self.ssp, rad_model)
        self.kernel_conv = partial(KernelConv, RadialModel=self.RadialModel)

        def make_layer(Rs_in, Rs_out):
            act = GatedBlock(Rs_out, scalar_act, gate_act)
            kc = self.kernel_conv(Rs_in, act.Rs_in)
            return torch.nn.ModuleList([kc, act])

        self.layers = torch.nn.ModuleList([torch.nn.Embedding(self.num_embeddings, embed, padding_idx=5)])
        self.layers += [make_layer(rs_in, rs_out) for rs_in, rs_out in zip(Rs, Rs[1:])]
        self.leakyrelu = nn.LeakyReLU(0.2) # Relu
        self.e_out_1 = nn.Linear(mlp_h + self.geo_out, mlp_h + self.geo_out)
        self.bn_out_1 = nn.BatchNorm1d(natoms)

        self.e_out_2 = nn.Linear(mlp_h + self.geo_out, 2*(mlp_h + self.geo_out))
        self.bn_out_2 = nn.BatchNorm1d(natoms)
        self.resnet_block = ResnetPointnet(3, self.geo_out)
        torch.autograd.set_detect_anomaly(True) 

    def forward(self, features, geometry, mask):
        mask = mask.to(torch.double)
        mask, diff_geo, radii = constants(geometry, mask)
        embedding = self.layers[0]
        geometry = geometry.to(torch.double)
        geo_pointnet = self.resnet_block(geometry, mask)
        features = torch.tensor(features).to(self.device).long()
        features = embedding(features).to(self.device)
        features = features.squeeze(2)
        set_of_l_filters = self.layers[1][0].set_of_l_filters
        y = spherical_harmonics_xyz(set_of_l_filters, diff_geo)
        for kc, act in self.layers[1:]:
            if kc.set_of_l_filters != set_of_l_filters:
                set_of_l_filters = kc.set_of_l_filters
                y = spherical_harmonics_xyz(set_of_l_filters, diff_geo)
            features = features.div(self.natoms ** 0.5).to(self.device)
            features = kc(
                features,
                diff_geo,
                mask,
                y=y,
                radii=radii,
                custom_backward=CUSTOM_BACKWARD
            )
            features = act(features)
            features = features * mask.unsqueeze(-1)
        print("features shape after enc", features.shape)
        
        # out_net = OutputMLPNetwork(kernel_conv=kernel_conv, previous_Rs = self.Rs[-1],
        #                          l0 = self.l0, l1 = 0, L = 1, scalar_act=sp, gate_act=rescaled_act.sigmoid,
        #                           mlp_h = 128, mlp_L = 1, natoms = 286)
        # features = out_net(features, geometry, mask)
        features = features.to(torch.double)

        features = torch.cat([features, geo_pointnet], dim=2)

        features = self.leakyrelu(self.bn_out_1(self.e_out_1(features))) # shape [batch, 2 * cloud_dim * (self.cloud_order ** 2) * nclouds]
        features = self.leakyrelu(self.bn_out_2(self.e_out_2(features)))

        # if self.atomref is not None:
        #     features_z = self.atomref(atomic_numbers)
        #     features = features_z + features
        features = self.atom_pool(features, mask)
        features = features.to(torch.double)
        # features = F.lp_pool2d(features,norm_type=2,
        #         kernel_size=(features.shape[1], 1),
        #         ceil_mode=False,)
        features = features.squeeze(1)
        features = features.squeeze(0)
        print("feat final shape", features.shape)
        return features # shape ? 




