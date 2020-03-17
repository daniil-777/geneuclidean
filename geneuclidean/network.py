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


class Net(nn.Module):
    def __init__(self, features, geometry):
        super(Net, self).__init__()
        self.RadialModel = partial(
            CosineBasisModel,
            max_radius=3.0,
            number_of_basis=3,
            h=100,
            L=1,
            act=torch.relu,
        )
        self.features = features
        self.geometry = geometry

        self.Rs_in = [(23, 0)]  # one scalar
        # in range(1) to narrow space of Rs_out
        self.Rs_out = [(1, l) for l in range(1)]
        self.fc1 = nn.Linear(features_new.shape[0], 30)
        self.fc2 = nn.Linear(30, 10)
        self.fc3 = nn.Linear(10, 1)  # prediction of Pkdb in regression
        self.convol = self.Convolution(self.K, self.Rs_in, self.Rs_out)

    def forward(self, x):
        features_new = self.convol(self.features, self.geometry)
        # x = ...(features_new)
        x = F.relu(self.fc1(features_new))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "main":
    current_path = os.path.realpath(os.path.dirname(__file__))
    path_pdb = os.path.join(current_path, "new_dataset")
    path_ligand = os.path.join(current_path, "refined-set")
    dataset_pdb = Pdb_Dataset(path_pdb, path_ligand)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # trainset - ?
    # testset - ?

    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
    #                                           shuffle=True, num_workers=2)

    # testloader = torch.utils.data.DataLoader(testset, batch_size=4,
    #                                          shuffle=False, num_workers=2)

    # for epoch in range(2):  # loop over the dataset multiple times

    #     running_loss = 0.0
    #     for i, data in enumerate(trainloader, 0):
    #         # get the inputs; data is a list of [inputs, labels]
    #         inputs, labels = data

    #         # zero the parameter gradients
    #         optimizer.zero_grad()

    #         # forward + backward + optimize
    #         outputs = net(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()

    #         # print statistics
    #         running_loss += loss.item()
    #         if i % 2000 == 1999:    # print every 2000 mini-batches
    #             print('[%d, %5d] loss: %.3f' %
    #                 (epoch + 1, i + 1, running_loss / 2000))
    #             running_loss = 0.0
