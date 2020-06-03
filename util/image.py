import torch
from util import opt


def unnormalize(x):
    x = x.transpose(1, 3)
    x = x * torch.Tensor(opt.STD) + torch.Tensor(opt.MEAN)
    x = x.transpose(1, 3)
    return x


def normalize(x):
    x = x.transpose(1, 3)
    x = (x - torch.Tensor(opt.MEAN)) / torch.Tensor(opt.STD)
    x = x.transpose(1, 3)
    return x
