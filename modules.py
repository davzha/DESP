import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import Sigmoid
import torchvision 


class FSPool(nn.Module):
    """
        Featurewise sort pooling.
        Adapted from: https://github.com/Cyanogenoid/fspool
    """

    def __init__(self, in_channels, n_pieces):
        """
        in_channels: Number of channels in input
        n_pieces: Number of pieces in piecewise linear
        """
        super().__init__()
        self.n_pieces = n_pieces
        self.weight = nn.Parameter(torch.zeros(n_pieces + 1, in_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def forward(self, x, n=None):
        index = torch.arange(end=x.size(1), device=x.device, dtype=torch.float32)
        index = index / (x.size(1) - 1)

        index = index.unsqueeze_(0).unsqueeze_(2)
        index = index.expand(x.size(0),-1, x.size(2))
        
        weight = self.weight.expand(index.size(0), -1, -1)
        perm = x.argsort(dim=1, descending=True)

        # linspace [0, 1] -> linspace [0, n_pieces]
        index = self.n_pieces * index
        idx = index.long()
        frac = index.frac()
        
        left = weight.gather(1, idx)
        right = weight.gather(1, (idx + 1).clamp(max=self.n_pieces))

        weight = (1-frac) * left + frac * right

        weight = weight.scatter(1, perm, weight)
        
        x = (x * weight)
        x = x.sum(1)
    
        return x


def get_mlp(d_in, d_hid, d_out, n_layers):
    layers = []
    for i in range(n_layers):
        layers.append(nn.Linear(
            d_in if i == 0 else d_hid, 
            d_hid if i < n_layers - 1 else d_out))
        if i < n_layers - 1:
            layers.append(nn.LeakyReLU(inplace=True))
    return nn.Sequential(*layers)


class DSEnergy(nn.Module):
    def __init__(self, d_x, d_y, d_hid, n_equiv, n_inv, normalize=False):
        super().__init__()
        self.normalize = normalize
        self.mlp_equiv = get_mlp(d_x + d_y, d_hid, d_hid, n_equiv)
        self.mlp_inv = get_mlp(d_hid, d_hid, d_hid, n_inv)
        self.L = nn.Parameter(torch.Tensor(d_hid, d_hid), requires_grad=True)
        self.pool = FSPool(d_hid, 20)
        self.reset_parameters()

    def reset_parameters(self):
        for l in self.parameters():
            if isinstance(l, nn.Linear):
                nn.init.kaiming_normal_(l.weight, a=self.activ.negative_slope)
                if l.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(l.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(l.bias, -bound, bound)
        nn.init.kaiming_normal_(self.L)

    def forward(self, y, x):
        if len(x.shape) < 3:
            x = x.unsqueeze(1).expand(-1, y.size(1), -1)
        z = torch.cat([x,y], dim=2)
        z = self.mlp_equiv(z)
        if self.normalize:
            z = z / z.size(1)
        z = self.pool(z)
        z = self.mlp_inv(z)
        energy = ((z @ torch.tril(self.L)) ** 2).sum(1)
        return energy



class FSEncoder(nn.Module):
    def __init__(self, d_in, d_hid, d_out, n_layers):
        super().__init__()
        self.d_in = d_in
        self.d_hid = d_hid
        self.d_out = d_out

        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(
                d_in if i == 0 else d_hid, 
                d_hid if i < n_layers - 1 else d_out))
            if i < n_layers - 1:
                layers.append(nn.ReLU(inplace=True))

        self.mlp = nn.Sequential(*layers)
        self.pool = FSPool(d_out, 20)

    def forward(self, x):
        x = self.mlp(x)
        x = self.pool(x)
        return x


class L1Energy(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, y, enc_x):
        enc_y = self.enc(y)
        return F.smooth_l1_loss(enc_y, enc_x, reduction='none').mean(1)


class ConvEncoder(nn.Module):
    """ Same as used in DSPN.
    ResNet34-based image encoder to turn an image into a feature vector 
    """

    def __init__(self, latent):
        super().__init__()
        resnet = torchvision.models.resnet34()
        self.layers = nn.Sequential(*list(resnet.children())[:-2])
        self.end = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            # now has 2x2 spatial size
            nn.Conv2d(512, latent, 2),
            # now has shape (n, latent, 1, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = self.end(x)
        return x.view(x.size(0), -1)


class PretrainedConvEncoder(nn.Module):
    """Resnet34 that was pre-trained for celebA classification.
    """

    def __init__(self, d_out, model_file, freeze_resnet=False):
        super().__init__()
        resnet = torchvision.models.resnet34()
        self.layers = nn.Sequential(*list(resnet.children())[:-1])
        state_dict = torch.load(model_file)
        self.layers.load_state_dict(state_dict,strict=False)

        if freeze_resnet:
            for param in self.layers.parameters():
                param.requires_grad = False

        self.end = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, d_out),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x)
        x = self.end(x.squeeze())
        return x.view(x.size(0), -1)