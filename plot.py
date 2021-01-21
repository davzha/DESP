import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import time_print


def plot_set(points, filepath=None, labels=None, n_sets=100, title=None, 
             thr=0.2, figsize=(3,3), point_size=1e2, fontsize=20, opacity=0.3,
             rows=None, cols=None):
    batch_size, n_points, n_feat = points.shape

    # tensor -> array
    if isinstance(points, torch.Tensor):
        points = points.data.cpu().numpy()

    if labels is not None and isinstance(labels, torch.Tensor):
        labels = labels.data.cpu().numpy()

    # add membership variable
    if n_feat == 2:
        points = np.concatenate([points, np.ones((batch_size, n_points, 1), dtype=points.dtype)], axis=2)

    if rows is not None and cols is not None:
        n_sets = rows * cols

    if batch_size < n_sets:
        time_print(f"Batch size {batch_size} is too small, require atleast {n_sets}")
        return False

    if rows is None:
        rows = int(n_sets ** 0.5)
    if cols is None:
        cols = int(n_sets ** 0.5)

    # point_color = colors.to_rgb("#969ad0")
    point_color = colors.to_rgb("#475468")
    # point_color = colors.to_rgb("#34495e")
    fig, axs = plt.subplots(rows, cols, squeeze=False)
    fig.set_size_inches(cols * figsize[0], rows * figsize[1])
    
    for i in range(rows*cols):
        color = np.zeros((n_points, 4))
        color[:, :3] = point_color

        color[:, 3] = np.clip(points[i, :, -1], 0, 1) * opacity

        keep = color[:, 3] > thr
        x = points[i,:, 1][keep]
        y = 1 - points[i,:, 0][keep]
        color = color[keep]

        axs[i // cols, i % cols].set_aspect('equal')
        
        if labels is not None:
            axs[i // cols, i % cols].set_title(labels[i], fontsize=fontsize)
        axs[i // cols, i % cols].scatter(x, y, marker=".", color=color, s=point_size, rasterized=True, lw=0)
        axs[i // cols, i % cols].axis(xmin=0, xmax=1, ymin=0, ymax=1)
        # axs[i // C, i % C].set_xticks([])
        # axs[i // C, i % C].set_yticks([])

    for a in axs.flat:
        a.set_xticks([])
        a.set_yticks([])

    if title is not None:
        axs.title(title)

    if filepath is not None:
        plt.savefig(filepath)
        plt.close()
    else:
        plt.show()

    return True
