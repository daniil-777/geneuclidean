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


CUSTOM_BACKWARD = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_kernel_conv(cutoff, n_bases, n_neurons, n_layers, act, radial_model):
    #choice of radial model depending on the kind of basis functions
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


class Bio_All_Network(torch.nn.Module):
    """network to predict atom-wise features from pocket atoms.
       Takes pharmacophoric and atom type (charge) features
        Args:
            natoms (int): max number of atoms
            encoding (string): type of encoding
            max_rad (float): radious of protein pocket
            num_basis (int): number of basis functions in radial convolution model
            n_neurons (int): number of neurons in convolution model
            n_layers (int): number of layers in convolution model
            beta (int): normalisation coefficient in convolution model
            rad_model (string): type of radial convolution model: gaussian, laplassian or ...
            num_embeddings (int): number of embeddings in embedding layer
            embed (int): type of embedding
            scalar_act_name (string): name for the scalar activation function
            gate_act_name (string): name for the gated block
            list_harm (string): list of harmonics dim
            aggregation_mode (string): mode for pooling: avg or max
            fc_sizes (string): list of output fc layers dimension

        Returns:
            [type]: [description]
        """
    def __init__(self,  natoms, encoding, max_rad, num_basis, n_neurons, n_layers, beta, rad_model, num_embeddings,
                 embed, scalar_act_name, gate_act_name,  list_harm, aggregation_mode, fc_sizes):

        super().__init__()
        self.natoms = natoms
        self.encoding = encoding
        self.ssp = rescaled_act.ShiftedSoftplus(beta = beta)
        self.sp = rescaled_act.Softplus(beta=beta)
        self.embed = embed
        self.list_harm = list_harm
        if(scalar_act_name == "sp"):
            scalar_act = self.sp
        
        if(gate_act_name == "sigmoid"):
            gate_act = rescaled_act.sigmoid

        Rs = [[(embed, 0)]]
        Rs += ast.literal_eval(self.list_harm)
        self.Rs = Rs
        self.fc_sizes = ast.literal_eval(fc_sizes)
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
        self.layers = torch.nn.ModuleList([torch.nn.Embedding(self.num_embeddings, embed, padding_idx=0)])
        self.layers += [make_layer(rs_in, rs_out) for rs_in, rs_out in zip(Rs, Rs[1:])]
        self.leakyrelu = nn.LeakyReLU(0.2) # Relu
        torch.autograd.set_detect_anomaly(True) 
        def fc_out_block(in_f, out_f):
            return nn.Sequential(
                nn.Linear(in_f, out_f),
                nn.BatchNorm1d(self.natoms),
                self.leakyrelu
            )
        def fc_out_block_no_bn(in_f, out_f):
            return nn.Sequential(
                nn.Linear(in_f, out_f),
                self.leakyrelu
            )
        self.fc_blocks_out = [fc_out_block(block_size[0], block_size[1]) 
                       for block_size in self.fc_sizes]
        self.fc_out = nn.Sequential(*self.fc_blocks_out)


    def encoding_block(self, features):
        # mask, diff_geo, radii = constants(geometry, mask)
        if self.encoding == "embedding":
            embedding = self.layers[0]
            features = torch.tensor(features).to(self.device).long()
            features = embedding(features).to(self.device)
            features = features.squeeze(2)
        else:
            features = torch.tensor(features).to(self.device).float()
            # features = torch.tensor(features).to(self.device)
            linear = nn.Linear(features.shape[2], self.embed).to(self.device)
            # features = features.long()
            features = linear(features).to(self.device)
            features = features.squeeze(2)
            features = features.double()
        return features


    def e3nn_block(self, features, geometry, mask):
        mask, diff_geo, radii = constants(geometry, mask)
        if self.encoding == "embedding":
            embedding = self.layers[0]
            features = torch.tensor(features).to(self.device).long()
            features = embedding(features).to(self.device)
        else:
            features = torch.tensor(features).to(self.device).float()
            # features = torch.tensor(features).to(self.device)
            linear = nn.Linear(features.shape[2], self.embed).to(self.device)
            # features = features.long()
            features = linear(features).to(self.device)
        features = features.squeeze(2)
        features = features.double()
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

        return features

    def fc_output(self, features, mask):
        #chain of MLP and pooling in the end
        features = self.fc_out(features)
        features = self.atom_pool(features, mask)
        features = features.squeeze(1)
        features = features.double()
        return features

    def forward(self, features, geometry, mask):
        features_bio = features[:, :, :7] #take pharma features
        features_charge = features[:, :, 7:] #take charges (atom type)
        features_bio = self.e3nn_block(features_bio, geometry, mask) #output after neural net on pharma features
        features_charge = self.e3nn_block(features_charge, geometry, mask) #output after neural net on atom type features
        features = torch.cat([features_bio, features_charge], dim=2) #concat
        # features = features.float()
        features = self.fc_output(features, mask) #apply MLP and pooling in the end
        return features # shape ? 

class Bio_All_Network_no_batch(Bio_All_Network):
    """Bio_All_Network without batch normalisaation in output fully connected layers  
    """
    def __init__(self,  natoms, encoding, max_rad, num_basis, n_neurons, n_layers, beta, rad_model, num_embeddings,
                 embed,   scalar_act_name, gate_act_name,  list_harm, aggregation_mode, fc_sizes):
        super(Bio_All_Network_no_batch, self).__init__(natoms, encoding, max_rad, num_basis, n_neurons, n_layers, beta, rad_model, num_embeddings,
                 embed,   scalar_act_name, gate_act_name,  list_harm, aggregation_mode, fc_sizes)
        def fc_out_block_no_bn(in_f, out_f):
            return nn.Sequential(
                nn.Linear(in_f, out_f),
                self.leakyrelu
            )

        self.fc_blocks_out = [fc_out_block_no_bn(block_size[0], block_size[1]) 
                       for block_size in self.fc_sizes]
        self.fc_out = nn.Sequential(*self.fc_blocks_out)

    def fc_output_no_bn(self, features, mask):
        features = self.fc_out(features)
        features = self.atom_pool(features, mask)
        features = features.squeeze(1)
        features = features.double()
        return features

    def forward(self, features, geometry, mask):
        features_bio = features[:, :, :7]
        features_charge = features[:, :, 7:]
        features_bio = self.e3nn_block(features_bio, geometry, mask)
        features_charge = self.e3nn_block(features_charge, geometry, mask)
        features = torch.cat([features_bio, features_charge], dim=2)
        # features = features.float()
        features = self.fc_output_no_bn(features, mask)
        return features # shape ? 

class Bio_Vis_All_Network(Bio_All_Network):
    """Bio_All_Network without pooling in output fully connected layers  
    """
    def __init__(self,  natoms, encoding, max_rad, num_basis, n_neurons, n_layers, beta, rad_model, num_embeddings,
                 embed,   scalar_act_name, gate_act_name,  list_harm, aggregation_mode, fc_sizes):
        super(Bio_Vis_All_Network, self).__init__(natoms, encoding, max_rad, num_basis, n_neurons, n_layers, beta, rad_model, num_embeddings,
                 embed,   scalar_act_name, gate_act_name,  list_harm, aggregation_mode, fc_sizes)
    
    def forward(self, features, geometry, mask):
        features_bio = features[:, :, :7]
        features_charge = features[:, :, 7:]
        features_bio = self.e3nn_block(features_bio, geometry, mask)
        features_charge = self.e3nn_block(features_charge, geometry, mask)
        features = torch.cat([features_bio, features_charge], dim=2)
        features = self.fc_out(features)
        # features = features.squeeze(1)
        features = features.double()
        return features # shape ? 


class Bio_Local_Network(Bio_All_Network):
    """Takes one type of features (just atom types or pharmocophoric features) and calculates atom-wise features
    """
    def __init__(self,  natoms, encoding, max_rad, num_basis, n_neurons, n_layers, beta, rad_model, num_embeddings,
                 embed,   scalar_act_name, gate_act_name,  list_harm, aggregation_mode, fc_sizes):
        super(Bio_Local_Network, self).__init__(natoms, encoding, max_rad, num_basis, n_neurons, n_layers, beta, rad_model, num_embeddings,
                 embed,   scalar_act_name, gate_act_name,  list_harm, aggregation_mode, fc_sizes)
    
    def forward(self, features, geometry, mask):
        features = self.e3nn_block(features, geometry, mask)
        features = self.fc_output(features, mask)
        return features # shape ? 

class ResNet_Bio_ALL_Network(Bio_All_Network):
    """Bio_All_Network with residual connection between the first layer and features after all layers of e3nn convolution
    """
    def __init__(self,  natoms, encoding, max_rad, num_basis, n_neurons, n_layers, beta, rad_model, num_embeddings,
                 embed,   scalar_act_name, gate_act_name,  list_harm, aggregation_mode, fc_sizes):
        super(ResNet_Bio_ALL_Network, self).__init__(natoms, encoding, max_rad, num_basis, n_neurons, n_layers, beta, rad_model, num_embeddings,
                 embed,   scalar_act_name, gate_act_name,  list_harm, aggregation_mode, fc_sizes)

    def resnet_e3nn_block(self, features, geometry, mask):
        mask, diff_geo, radii = constants(geometry, mask)
        features = self.encoding_block(features)
        set_of_l_filters = self.layers[1][0].set_of_l_filters
        y = spherical_harmonics_xyz(set_of_l_filters, diff_geo)
        kc, act = self.layers[1]
        features = kc(
            features.div(self.natoms ** 0.5),
            diff_geo,
            mask,
            y=y,
            radii=radii,
            custom_backward=CUSTOM_BACKWARD
        )
        features = act(features)
        # print("shape feat before conv", features.shape)
        for kc, act in self.layers[2:]:
            if kc.set_of_l_filters != set_of_l_filters:
                set_of_l_filters = kc.set_of_l_filters
                y = spherical_harmonics_xyz(set_of_l_filters, diff_geo)
            new_features = kc(
                features.div(self.natoms ** 0.5),
                diff_geo,
                mask,
                y=y,
                radii=radii,
                custom_backward=CUSTOM_BACKWARD
            )
            new_features = act(new_features)
            new_features = new_features * mask.unsqueeze(-1)
            features = features + new_features
        return features

    def forward(self, features, geometry, mask):
        features_bio = features[:, :, :7]
        features_charge = features[:, :, 7:]
        features_bio = self.resnet_e3nn_block(features_bio, geometry, mask)
        features_charge = self.resnet_e3nn_block(features_charge, geometry, mask)
        features = torch.cat([features_bio, features_charge], dim=2)
        # features = features.float()
        features = self.fc_output(features, mask)
        
        return features


        
class ResNet_Bio_Local_Network(ResNet_Bio_ALL_Network):
    """Bio_Local_Network with residual connection between the first layer and features after all layers of e3nn convolution
    """
    def __init__(self,  natoms, encoding, max_rad, num_basis, n_neurons, n_layers, beta, rad_model, num_embeddings,
                 embed,   scalar_act_name, gate_act_name,  list_harm, aggregation_mode, fc_sizes):
        super(ResNet_Bio_Local_Network, self).__init__(natoms, encoding, max_rad, num_basis, n_neurons, n_layers, beta, rad_model, num_embeddings,
                 embed,   scalar_act_name, gate_act_name,  list_harm, aggregation_mode, fc_sizes)

    def forward(self, features, geometry, mask):
        features = self.resnet_e3nn_block(features, geometry, mask)
        features = self.fc_output(features, mask)
        return features # shape ?

class Concat_Bio_Local_Network(ResNet_Bio_ALL_Network):
    """Bio_Local_Network where all features in e3nn conv layers ae concatenated together
    """
    def __init__(self,  natoms, encoding, max_rad, num_basis, n_neurons, n_layers, beta, rad_model, num_embeddings,
                 embed,   scalar_act_name, gate_act_name,  list_harm, aggregation_mode, fc_sizes):
        super(Concat_Bio_Local_Network, self).__init__(natoms, encoding, max_rad, num_basis, n_neurons, n_layers, beta, rad_model, num_embeddings,
                 embed,   scalar_act_name, gate_act_name,  list_harm, aggregation_mode, fc_sizes)

    def concat_e3nn_block(self, features, geometry, mask):
        features_all = []
        mask, diff_geo, radii = constants(geometry, mask)
        features = self.encoding_block(features)
        set_of_l_filters = self.layers[1][0].set_of_l_filters
        y = spherical_harmonics_xyz(set_of_l_filters, diff_geo)
        kc, act = self.layers[1]
        features = kc(
            features.div(self.natoms ** 0.5),
            diff_geo,
            mask,
            y=y,
            radii=radii,
            custom_backward=CUSTOM_BACKWARD
        )
        features = act(features)
        features_all.append(features )
        for kc, act in self.layers[2:]:
            if kc.set_of_l_filters != set_of_l_filters:
                set_of_l_filters = kc.set_of_l_filters
                y = spherical_harmonics_xyz(set_of_l_filters, diff_geo)
            new_features = kc(
                features.div(self.natoms ** 0.5),
                diff_geo,
                mask,
                y=y,
                radii=radii,
                custom_backward=CUSTOM_BACKWARD
            )
            new_features = act(new_features)
            new_features = new_features * mask.unsqueeze(-1)
            features_all.append(new_features)
        features_all = torch.cat(features_all, 2) #concatenation of all features
        # features = features + new_features
        return features_all

    def forward(self, features, geometry, mask):
        features = self.concat_e3nn_block(features, geometry, mask)
        features = self.fc_output(features, mask)
        return features # shape ?