import math

import scipy.optimize
import torch
import torch.nn as nn
import torch.nn.functional as F

TWO_PI = 2 * math.pi


def chamfer_distance(x, y):
    """
    Args:
        x: (batch, set, num_features)
        y: - // -
    """
    pdist = F.smooth_l1_loss(
        x.unsqueeze(1).expand(-1, y.size(1), -1, -1), 
        y.unsqueeze(2).expand(-1, -1, x.size(1), -1),
        reduction='none').mean(3)  # TODO mean?

    loss = pdist.min(1)[0] + pdist.min(2)[0]
    return loss.view(loss.size(0), -1).mean(1)


def LAP_loss(x, y, pool=None):
    pdist = F.smooth_l1_loss(
        x.unsqueeze(1).expand(-1, y.size(1), -1, -1), 
        y.unsqueeze(2).expand(-1, -1, x.size(1), -1),
        reduction='none').mean(3)

    pdist_ = pdist.detach().cpu().numpy()

    if pool is not None:
        indices = list(pool.map(scipy.optimize.linear_sum_assignment, pdist_))
    else:
        indices = [scipy.optimize.linear_sum_assignment(p) for p in pdist_]

    losses = [
        sample[row_idx, col_idx].mean()
        for sample, (row_idx, col_idx) in zip(pdist, indices)
    ]

    total_loss = torch.stack(losses)

    return total_loss


class ApproxLoss(nn.Module):
    """Approximate set loss between prediction and closest valid polygon.
    """
    def __call__(self, y, n):
        # batch , n_csets , n_points, d_feat
        n = torch.nonzero(self.labels.unsqueeze(0) == n.unsqueeze(1))[:,1]
        y_gt = self.examples[n]
        y_gt = y_gt.view(-1, *y.shape[1:])

        y_e = y.unsqueeze(1).expand(-1, self.gran, -1, -1)
        y_e = y_e.reshape(-1, *y.shape[1:])

        l = self.set_distance(y_gt, y_e)
        l = l.reshape(-1, self.gran).min(1)[0]

        return l

class RotatedPolygonLoss(ApproxLoss):
    def __init__(self, n_poly, set_distance, radius=0.35, center=0.5, max_n=None, gran=256):
        super().__init__()
        self.labels = nn.Parameter(torch.Tensor(n_poly), requires_grad=False)
        self.set_distance = set_distance
        self.radius = radius
        self.center = center
        self.max_n = max_n if max_n is not None else max(n_poly)
        self.gran = gran

        csets = []
        for n in n_poly:
            points = torch.linspace(0., TWO_PI-(TWO_PI / n),n)
            angles = torch.linspace(0, TWO_PI-(TWO_PI/self.gran),self.gran)
            theta = points.unsqueeze(0) + angles.unsqueeze(1)
            xy = self.theta2xy(theta)
            csets.append(xy)
        self.examples = nn.Parameter(torch.stack(csets), requires_grad=False)

    def theta2xy(self, theta):
        x = torch.cos(theta) * self.radius
        y = torch.sin(theta) * self.radius
        xy = torch.stack((x,y), dim=-1) + self.center
        if xy.size(1) < self.max_n:
            padding = torch.zeros_like(xy[:,:self.max_n - xy.size(1)])
            xy = torch.cat([xy, padding], dim=1)
        return xy


class DigitLoss(ApproxLoss):
    def __init__(self, digit_examples, labels,set_distance):
        super().__init__()
        self.examples = nn.Parameter(digit_examples, requires_grad=False)
        self.labels = nn.Parameter(labels, requires_grad=False)
        self.set_distance = set_distance

        self.gran = self.examples.size(0) // 2


