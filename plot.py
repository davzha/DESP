import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

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



def plot_rect(points, img, filepath=None, n_sets=100, title=None, 
             thr=0.2, figsize=(3,3), point_size=1e2, fontsize=20, opacity=0.3,
             rows=None, cols=None):
    batch_size, n_points, n_feat = points.shape
    _, _, w, h = img.shape

    # tensor -> array
    if isinstance(points, torch.Tensor):
        points = points.data.cpu().numpy()

    if isinstance(img, torch.Tensor):
        img = img.data.cpu().numpy()

    points[:,:,:4] *= np.array([w,h,w,h])
    img = img.transpose(0, 2, 3, 1)

    if rows is not None and cols is not None:
        n_sets = rows * cols

    if batch_size < n_sets:
        time_print(f"Batch size {batch_size} is too small, require atleast {n_sets}")
        return False

    if rows is None:
        rows = int(n_sets ** 0.5)
    if cols is None:
        cols = int(n_sets ** 0.5)

    fig, axs = plt.subplots(rows, cols, squeeze=False)
    fig.set_size_inches(cols * figsize[0], rows * figsize[1])
    
    for i in range(rows*cols):
        axs[i // cols, i % cols].imshow(img[i])
        rect = gather_patch(points[i], thr)
        for r in rect:
            axs[i // cols, i % cols].add_patch(r)

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

def gather_patch(points, thr):
    rects = []
    mem = (points > thr).all(1)  # TODO
    for p in points[mem]:
        w = p[2] - p[0]
        l = p[3] - p[1]
        rects.append(Rectangle((p[0], p[1]), w, l, linewidth=1,edgecolor='r',facecolor='none'))

    return rects



def plot_imgs(imgs, filepath=None, labels=None, n_imgs=100, title=None, 
              figsize=(3,3), fontsize=20,
              rows=None, cols=None, row_labels=None, col_labels=None, rl_fs=24, cl_fs=24,
              sub_colors=None, sub_linewidth=None, sub_linestyle=None):
    batch_size, n_channels, width, height = imgs.shape

    imgs = imgs.permute(0,2,3,1)
    imgs = imgs.squeeze(3)
    # imgs = imgs * 0.5 + 0.5

    # tensor -> array
    if isinstance(imgs, torch.Tensor):
        imgs = imgs.data.cpu().numpy()

    if labels is not None and isinstance(labels, torch.Tensor):
        labels = labels.data.cpu().numpy()

    if rows is not None and cols is not None:
        n_imgs = rows * cols

    if batch_size < n_imgs:
        time_print(f"Batch size {batch_size} is too small, require atleast {n_imgs}")
        return False

    if rows is None:
        rows = int(n_imgs ** 0.5)
    if cols is None:
        cols = int(n_imgs ** 0.5)

    fig, axs = plt.subplots(rows, cols, squeeze=False)
    fig.set_size_inches(cols * figsize[0], rows * figsize[1])

    for i in range(rows*cols):
        axs[i // cols, i % cols].imshow(imgs[i])
        
        if sub_colors is not None:
            plt.setp(axs[i // cols, i % cols].spines.values(), color=sub_colors[i])
        if sub_linewidth is not None:
            plt.setp(axs[i // cols, i % cols].spines.values(), linewidth=sub_linewidth[i])
        if sub_linestyle is not None:
            plt.setp(axs[i // cols, i % cols].spines.values(), linestyle=sub_linestyle[i])
        if labels is not None:
            axs[i // cols, i % cols].set_title(labels[i], fontsize=fontsize)

    pad = 5
    if row_labels is not None:
        for a, l in zip(axs[:,0], row_labels):
            a.annotate(l, xy=(0, 0.5), xytext=(-a.yaxis.labelpad - pad, 0),
                    xycoords=a.yaxis.label, textcoords='offset points',
                    size=rl_fs, ha='right', va='center')

    if col_labels is not None:
        for a, l in zip(axs[0], col_labels):
            a.annotate(l, xy=(0.5, 1.), xytext=(0, 2 * pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size=cl_fs, ha='center', va='baseline')

    for a in axs.flat:
        a.set_xticks([])
        a.set_yticks([])

    if title is not None:
        axs.title(title)

    fig.tight_layout()

    if filepath is not None:
        plt.savefig(filepath)
        plt.close()
    else:
        plt.show()

    return True