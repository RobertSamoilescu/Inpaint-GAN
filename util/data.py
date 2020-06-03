import numpy as np
import pandas as pd
import pickle as pkl
import os
import torch
from util import opt
from torch.utils.data import Dataset
from torch.utils import data
from PIL import Image


# dataloader sampler
class InfiniteSampler(data.sampler.Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        i = 0
        order = np.random.permutation(self.num_samples)
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                np.random.seed()
                order = np.random.permutation(self.num_samples)
                i = 0


# data set class
class NusceneDataset(Dataset):
    def __init__(self, root_dir: str,  train: bool=True, random_mask: bool=True):
        self.root_dir = root_dir
        self.files = pd.read_csv('%s/%s.csv' % (root_dir, "train" if train else "test")).name
        self.random_mask = random_mask

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # load ground-truth image
        filename = os.path.join(self.root_dir, "imgs", self.files[idx] + ".png")
        gt = np.array(Image.open(filename)) / 255.

        # load mask
        ridx = np.random.randint(0, len(self.files)) if self.random_mask else idx
        filename = os.path.join(self.root_dir, "masks", self.files[ridx] + ".png")
        mask = np.expand_dims(np.array(Image.open(filename)) / 255., 2)

        # image masked
        img = gt * mask

        # load depth map
        filename = os.path.join(self.root_dir, "depths", self.files[idx] + ".pkl")
        with open(filename, 'rb') as f:
            depth = pkl.load(f)

        # load intrinsic
        filename = os.path.join(self.root_dir, "intrinsics", self.files[idx] + ".pkl")
        with open(filename, 'rb') as f:
            intrinsic = pkl.load(f)

        # load extrinisc
        filename = os.path.join(self.root_dir, "extrinsics", self.files[idx] + ".pkl")
        with open(filename, 'rb') as f:
            extrinsic = pkl.load(f)

        # transform to tensors
        sample = {
            "img": torch.tensor(((img - opt.MEAN) / opt.STD).transpose(2, 0, 1)),
            "mask": torch.tensor(mask.transpose(2, 0, 1)).repeat(3, 1, 1),
            "gt": torch.tensor(((gt - opt.MEAN) / opt.STD).transpose(2, 0, 1)),
            "depth": torch.from_numpy(depth),
            "intrinsic": torch.from_numpy(intrinsic),
            "extrinsic": torch.from_numpy(extrinsic)
        }
        return sample
