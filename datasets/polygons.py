import math
import random

import torch

TWO_PI = 2 * math.pi

class Polygons(torch.utils.data.Dataset):
    def __init__(self, n_points, n_poly, radius=0.35, noise=True, length=60000, mem_feat=False, mode=None):
        self.n_points = n_points
        self.length = length
        self.center = torch.tensor((0.5,0.5))
        self.radius = radius
        self.noise = noise
        self.n_poly = n_poly
        self.mem_feat = mem_feat
        self.mode = mode

    def _get_n_polygon(self, n):
        angles = torch.linspace(0., TWO_PI - (TWO_PI / n), n)
        radius = self.radius
        if self.noise:
            angles += torch.empty(1).uniform_(0., TWO_PI)
        # target = torch.randint(self.centers.shape[0], (1,))
        center = self.center

        x = torch.cos(angles) * radius
        y = torch.sin(angles) * radius

        coo = torch.stack((x,y), dim=1)
        coo = coo + center

        padding = torch.zeros(self.n_points - n, 2, dtype=coo.dtype)
        padding_len = padding.shape[0]
        nonpadding_len = coo.shape[0]
        coo = torch.cat([coo, padding], dim=0)
        if self.mem_feat:
            membership = torch.zeros(self.n_points, 1, dtype=coo.dtype)
            membership[:n].fill_(1.)
            coo = torch.cat([coo, membership], dim=-1)
        if self.n_points != coo.shape[0]:
            print(coo.shape, n, padding_len, nonpadding_len)
        return coo

    def __getitem__(self, item):
        # angles = torch.empty(self.n_points).uniform_(0., 2 * np.pi)
        n = random.choice(self.n_poly)
        coo = self._get_n_polygon(n)
        return n, coo

    def __len__(self):
        return self.length
