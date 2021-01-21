import math
import random

import numpy as np
import torch

class Digits(torch.utils.data.Dataset):
    def __init__(self, n_points, mem_feat=False, length=60000, mode=None):
        self.n_points = n_points
        self.mem_feat = mem_feat
        self.length = length

        self.mid_mm = [[0.48, 0.25],
                   [0.52, 0.75]]

        self.hook_mm = [[0.3, 0.00],
                       [0.5, 0.04]]
        self.hook_b = np.array([0.125, 0.37])

        self.hook_rotm = np.array([[math.cos(math.pi/4), -math.sin(math.pi/4)],
                          [math.sin(math.pi/4), math.cos(math.pi/4)]])
        
        self.mid_rotm = np.array([[math.cos(-math.pi/10), -math.sin(-math.pi/10)],
                          [math.sin(-math.pi/10), math.cos(-math.pi/10)]])
        self.mid_seven_b = np.array([-0.1,0.1])

        self.top_bar_mm = [[0.4,0.61],
                           [0.57,0.67]]

        self.bar_mm = [[0.415,0.42],
                       [0.595,0.48]]
        
        self.figure_fn = [self._build_one, self._build_seven]

    def pad(self, points, nonpad):
        points[:nonpad,1] = 1. - points[:nonpad,1]
        points = np.flip(points,axis=1).copy()

        membership = torch.zeros(self.n_points, 1, dtype=torch.float)
        membership[:nonpad].fill_(1.)
        
        points = torch.tensor(points, dtype=torch.float)

        if self.mem_feat:
            points = torch.cat([points, membership], dim=1)
        return points

    def _build_one(self, opt):
        n_mp = int(self.n_points / 1.5)
        n_hp = self.n_points - n_mp
        mid = np.random.uniform(self.mid_mm[0], self.mid_mm[1], size=(n_mp, 2))
        
        if opt:
            hook = np.random.uniform(self.hook_mm[0], self.hook_mm[1], size=(n_hp, 2))
            hook = hook @ self.hook_rotm.T + self.hook_b
            points = np.concatenate([mid, hook], axis=0)
            nonpad = self.n_points
        else:
            pad = np.zeros((n_hp,2))
            points = np.concatenate([mid, pad], axis=0)
            nonpad = n_mp

        points = self.pad(points, nonpad)

        return points

    def _build_seven(self, opt):
        n_mp = int(self.n_points / 2)
        n_tp = int(self.n_points / 4)
        n_bp = self.n_points - n_mp - n_tp

        mid = np.random.uniform(self.mid_mm[0], self.mid_mm[1], size=(n_mp, 2))
        mid = mid @ self.mid_rotm.T + self.mid_seven_b

        top = np.random.uniform(self.top_bar_mm[0], self.top_bar_mm[1], size=(n_tp, 2))

        if opt:
            bar = np.random.uniform(self.bar_mm[0], self.bar_mm[1], size=(n_bp,2))
            points = np.concatenate([mid, top, bar], axis=0)
            nonpad = self.n_points
        else:
            pad = np.zeros((n_bp, 2))
            points = np.concatenate([mid, top, pad], axis=0)
            nonpad = n_mp + n_tp

        points = self.pad(points, nonpad)

        return points

    def __getitem__(self, item):
        n, fn = random.choice(list(enumerate(self.figure_fn)))
        opt = random.choice([False, True])

        points = fn(opt=opt)

        return n, points

    def __len__(self):
        return self.length