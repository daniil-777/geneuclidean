import torch
from e3nn.o3 import rand_rot
from e3nn.networks import (
    GatedConvParityNetwork,
    GatedConvNetwork,
    ImageS2Network,
    S2ConvNetwork,
    S2ParityNetwork,
)


class SumNetwork(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.network = GatedConvNetwork(*args, **kwargs)

    def forward(self, *args, **kwargs):
        output = self.network(*args, **kwargs)
        return output.sum(-2)  # Sum over N


class MySumNetwork(torch.nn.Module):
    def __init__(self, final_out):
        super().__init__()
        self.final_out = final_out
        self.leakyrelu = nn.LeakyReLU(0.2) # Relu
        self.e_out_1 = nn.Linear(mlp_h, mlp_h)
        self.bn_out_1 = nn.BatchNorm1d(avg_n_atoms)

        self.e_out_2 = nn.Linear(mlp_h, 2 * mlp_h)
        self.bn_out_2 = nn.BatchNorm1d(avg_n_atoms)
   

    def forward(self, features, geometry):
       
        embedding = self.layers[0]
        features = embedding(features)
        Rs_in = [(1, 0)]
        Rs_hidden = [(middle, 0)]
        Rs_out = [(final_out, 0)]
        f = SumNetwork(Rs_in, Rs_hidden, Rs_out, lmax)
        f = f.to(device)
        features = torch.tensor(features).to(self.device).long()
        features = embedding(features).to(self.device)
        features = features.squeeze(2)
        features = f(features, geometry)
        features = F.lp_pool2d(features,norm_type=2,
                kernel_size=(features.shape[1], 1),
                ceil_mode=False,)
        features = self.leakyrelu(self.bn_out_1(self.e_out_1(features)))
        features = self.leakyrelu(self.bn_out_2(self.e_out_2(features)))

        features = F.lp_pool2d(features,norm_type=2,
                kernel_size=(features.shape[1], 1),
                ceil_mode=False,)
        features = features.squeeze(1)
        print("feat final shape", features.shape)
        return features # shape ? 

class MyS2convNetwork(torch.nn.Module):
    def __init__(self, final_out):
        super().__init__()
        self.final_out = final_out
        self.leakyrelu = nn.LeakyReLU(0.2) # Relu
        self.e_out_1 = nn.Linear(mlp_h, mlp_h)
        self.bn_out_1 = nn.BatchNorm1d(avg_n_atoms)

        self.e_out_2 = nn.Linear(mlp_h, 2 * mlp_h)
        self.bn_out_2 = nn.BatchNorm1d(avg_n_atoms)

    def forward(self, features, geometry):
       
        embedding = self.layers[0] #?
        features = embedding(features)
        lmax = 3
        Rs = [(1, l, 1) for l in range(lmax + 1)]
        model = S2ConvNetwork(Rs, 4, Rs, lmax)
        features = model(features, geometry)
        return features # shape ? 

def test_s2conv_network():
    torch.set_default_dtype(torch.float64)

    lmax = 3
    Rs = [(1, l, 1) for l in range(lmax + 1)]
    model = S2ConvNetwork(Rs, 4, Rs, lmax)

    features = rs.randn(1, 4, Rs)
    geometry = torch.randn(1, 4, 3)

    output = model(features, geometry)

    angles = o3.rand_angles()
    D = rs.rep(Rs, *angles, 1)
    R = -o3.rot(*angles)
    ein = torch.einsum
    output2 = ein('ij,zaj->zai', D.T, model(ein('ij,zaj->zai', D, features), ein('ij,zaj->zai', R, geometry)))

    assert (output - output2).abs().max() < 1e-10 * output.abs().max()


def main():
    torch.set_default_dtype(torch.float64)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tetris, labels = get_dataset()
    tetris = tetris.to(device)
    labels = labels.to(device)
    Rs_in = [(1, 0)]
    Rs_hidden = [(16, 0), (16, 1), (16, 2)]
    Rs_out = [(len(tetris), 0)]
    lmax = 3

    f = SumNetwork(Rs_in, Rs_hidden, Rs_out, lmax)
    f = f.to(device)

    optimizer = torch.optim.Adam(f.parameters(), lr=1e-2)

    feature = tetris.new_ones(tetris.size(0), tetris.size(1), 1)

    for step in range(50):
        out = f(feature, tetris)
        loss = torch.nn.functional.cross_entropy(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = out.argmax(1).eq(labels).double().mean().item()
        print("step={} loss={} accuracy={}".format(step, loss.item(), acc))

    out = f(feature, tetris)

    r_tetris, _ = get_dataset()
    r_tetris = r_tetris.to(device)
    r_out = f(feature, r_tetris)

    print('equivariance error={}'.format((out - r_out).pow(2).mean().sqrt().item()))


if __name__ == '__main__':
    main()
