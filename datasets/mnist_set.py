from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as T


class MNISTSet(torch.utils.data.Dataset):
    """From https://github.com/Cyanogenoid/dspn with slight modifications.
    """
    def __init__(self, threshold=0.0, n_points=342, mode='train',
                 root=".mnist", full=False, random_padding=False,
                 overwrite_cache=False, filter_label=None,
                 mem_feat=False):
        self.mode = mode
        self.root = Path(root)
        self.threshold = threshold
        self.full = full
        self.random_padding = random_padding
        self.overwrite_cache = overwrite_cache
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        mnist = torchvision.datasets.MNIST(
            train=mode=='train', transform=transform, download=True, root=root
        )
        self.filter_label = filter_label
        self.mem_feat = mem_feat
        self.max = n_points
        self.data = self.cache(mnist)

    def cache(self, dataset):
        keep_label = "all" if self.filter_label is None else "".join(map(str,self.filter_label))
        cache_path = self.root / \
            (f"mnist_{self.mode}"
             f"_{self.threshold}"
             f"_{keep_label}.pth")
        if cache_path.exists() and not self.overwrite_cache:
            return torch.load(cache_path)

        print("Processing dataset...")
        data = []
        for datapoint in dataset:
            img, label = datapoint
            if self.filter_label is not None and label not in self.filter_label:
                continue
            point_set, cardinality = self.image_to_set(img)
            data.append((point_set, label, cardinality))
        torch.save(data, cache_path)
        print("Done!")
        return data

    def image_to_set(self, img):
        idx = (img.squeeze(0) > self.threshold).nonzero()
        cardinality = idx.size(0)
        return idx, cardinality

    def __getitem__(self, item):
        s, l, c = self.data[item]
        # make sure set is shuffled
        s = s[torch.randperm(c)]
        # pad to fixed size
        padding_size = self.max - s.size(0)
        # put in range [0, 1]
        s = s.float() / 27.
        if self.random_padding:
            s = torch.cat([s, torch.rand(padding_size, 2, dtype=torch.float)], dim=0)
        else:
            s = torch.cat([s, torch.zeros(padding_size, 2)], dim=0)

        # mask of which elements are valid,not padding
        if self.mem_feat:
            mask = torch.zeros(self.max, 1)
            mask[:c].fill_(1)
            s = torch.cat([s, mask], dim=-1)

        return l, s

    def __len__(self):
        if self.mode == 'train' or self.full:
            return len(self.data)
        else:
            return len(self.data) // 10