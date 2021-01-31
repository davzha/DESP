from pathlib import Path
import numpy as np
from itertools import combinations

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader


class CelebASet(torchvision.datasets.CelebA):
    KEEP_ATTR = [
        4,  # Bald
        5,  # Bangs
        9,  # Blond_Hair
        14, # Double_Chin
        15, # Eyeglasses
        16, # Goatee
        17, # Gray_Hair
        20, # Male
        24, # No_Beard
        35, # Wearing_Hat
        38, # Wearing_Necktie
    ]
    def __init__(self, root, mode, n_attr, set_size, n_outliers, p_outliers=None, 
            min_n_target=1, length=500, transform=None, download=False):
        if mode == 'val':
            split = 'valid'
        elif mode == 'train':
            split = 'train'
        elif mode == 'test':
            split = 'test'
        else:
            raise ValueError(mode)
        super().__init__(Path(root), split, target_type="attr", transform=transform, download=download)
        self.n_attr = n_attr
        self.set_size = set_size
        self.n_outliers = n_outliers
        self.p_outliers = p_outliers
        self.min_n_target = min_n_target
        self.length = length
        self.min_in = set_size - max(n_outliers)

        if self.transform is None:
            self.transform = self.get_default_transform()

        self.init_attr_combinations()

    def init_attr_combinations(self, threshold=10):
        """ 
        Truncate attribute combinations by their number of images (having said attributes)
        """
        # all combinations
        attr_comb = np.array(list(combinations(self.KEEP_ATTR, self.n_attr)))
        n_img_per_attr_comb = self.attr[:,attr_comb].bool().all(2).int().sum(0)
        self.attr_comb = attr_comb[n_img_per_attr_comb > threshold]

    def load_imgs(self, sam):
        imgs = []
        for i in sam:
            img = default_loader(self.root / self.base_folder / "img_align_celeba" / self.filename[i])
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs)
        return imgs

    def get_default_transform(self):
        tr = [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            ]
        return transforms.Compose(tr)

    def __getitem__(self, index):
        # select random subset size of outlier samples
        n_out = np.random.choice(self.n_outliers, p=self.p_outliers)
        n_in = self.set_size - n_out
        
        targets = torch.zeros(self.set_size)
        targets[:n_in] = 1.

        while True:
            # select random attribute combinations used to sample inliers
            attr_in = self.attr_comb[np.random.randint(self.attr_comb.shape[0])]

            # sample
            inlier_indicator = (self.attr[:,attr_in] == 1).all(1)
            sample_in = np.random.choice(
                np.flatnonzero(inlier_indicator).squeeze(), size=(n_in,), replace=False)
            sample_out = np.random.choice(
                np.flatnonzero(~inlier_indicator).squeeze(), size=(n_out,), replace=False)
            sample = np.concatenate([sample_in, sample_out])

            # compute all valid targets, for example sample_out may include images
            # that share other attributes with sample_in, then were used for sampling
            all_targets = get_targets(self.attr[sample][:,self.KEEP_ATTR], min_in=self.min_in)
            # re-sample if there aren't enough targets
            if all_targets.size(0) >= self.min_n_target:
                break

        imgs = self.load_imgs(sample)

        return imgs, targets, all_targets

    def __len__(self):
        return self.length

def get_targets(attr, min_in):
    # list all possible combinations
    set_size = attr.size(0)
    x = torch.tensor([0,1])
    targets = torch.cartesian_prod(*[x]*set_size)

    # remove those that do not have enough inliers
    targets = targets[targets.sum(1) >= min_in]

    # check the remaining for if they are valid
    valid = []
    for i in range(targets.size(0)):
        valid.append(check(targets[i], attr))
    valid = torch.stack(valid)
    return targets[valid]

def check(target, attr, n_attr=2):
    inlier_indicator = target.squeeze().bool()
    inlier_attr = attr[inlier_indicator]  # attributes for each inlier
    attr_in = inlier_attr.bool().all(0)  # attributes shared by all inliers
    outlier_attr = attr[~inlier_indicator]  # attributes for each outlier
    shared_inlier_attr = (attr_in.sum() >= n_attr)  # predicate checking for sufficient number of shared attribtues
    not_shared_with_outliers = ((attr_in & outlier_attr.bool()).sum(1) < attr_in.sum()).all()
    return shared_inlier_attr & not_shared_with_outliers
