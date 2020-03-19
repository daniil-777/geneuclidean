import os
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from e3nn import SO3
from e3nn.kernel import Kernel
from e3nn.non_linearities import GatedBlock
from e3nn.point.operations import Convolution
from e3nn.radial import CosineBasisModel
from e3nn.util.plot import plot_sh_signal
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
        self.Rs_in = [(23, 0)]  # one scalar
        # in range(1) to narrow space of Rs_out
        self.Rs_out = [(1, l) for l in range(1)]
        self.fc1 = nn.Linear(LEN_PADDING, 30)  # need to clarify 150 or not
        self.fc2 = nn.Linear(30, 10)
        self.fc3 = nn.Linear(10, 1)  # prediction of Pkdb in regression
        self.convol = Convolution(self.K, self.Rs_in, self.Rs_out)

    def forward(self, x, features, geometry):
        features_new = self.convol(features, geometry).squeeze(2)
        length_padding = 286 - features_new.shape[1]
        # padding till the shape of the biggest feature size - 286
        result_feature = F.pad(
            input=features_new, pad=(0, length_padding, 0, 0), mode="constant", value=0
        )
        # x = ...(features_new)
        x = F.relu(self.fc1(features_new))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
