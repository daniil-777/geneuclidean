from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable

def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out

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
    

    def __init__(self, dim=None, hidden_dim=None):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim   
        
        # For grid features
        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.block_0 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_2 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_3 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(2*hidden_dim, hidden_dim)
    
        # Activation & pooling
        self.actvn = nn.ReLU()
        self.pool = maxpool
        
        is_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if is_cuda else "cpu")

    def forward(self, p):
        batch_size, T, D = p.size()
        print("D", D)
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
        # pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        # net = torch.cat([net, pooled], dim=2)
        net = self.block_4(net) # batch_size x T x hidden_dim (T: number of sampled input points)
        return net
