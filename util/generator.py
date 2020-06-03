import numpy as np
import random
from util.inverse_warp import *

action_list = [[0, 1], [0, -1], [1, 0], [-1, 0]]


def random_walk(canvas, ini_x, ini_y, length):
    x = ini_x
    y = ini_y
    height, width = canvas.shape
    x_list = []
    y_list = []
    for i in range(length):
        r = random.randint(0, len(action_list) - 1)
        x = np.clip(x + action_list[r][0], a_min=0, a_max=width - 1)
        y = np.clip(y + action_list[r][1], a_min=0, a_max=height - 1)
        x_list.append(x)
        y_list.append(y)
    canvas[np.array(y_list), np.array(x_list)] = 0
    return canvas


# define function that performs a random transformation to a batch of images
def random_transformation(imgs, intrinsics, extrinsics, depths):
    """
    Generate random transformation
    @param imgs:         [B, 3, H, W]
    @param depths:       [B, H, W]
    @param intrinsics:   [B, 3, 3]
    @param extrinsics:   [B, 3, 4]
    @param depths:       [B, 1, H, W]
    """
    # sample random transformation
    batch_size = imgs.shape[0]
    poses = torch.zeros(batch_size, 6).double()
    tx = 0.2 * 2 * (torch.rand(batch_size) - 0.5)
    ry = 0.2 * 2 * (torch.rand(batch_size) - 0.5)
    poses[:, 0], poses[:, 4] = tx, ry

    # apply transformation
    projected_imgs, valid_points = inverse_warp(
        img=imgs,
        depth=depths,
        pose=poses,
        intrinsics=intrinsics,
        extrinsics=extrinsics
    )

    # mask of valid points
    valid_points = (valid_points * (depths > 0)).double().unsqueeze(1)
    projected_imgs = projected_imgs * valid_points
    return projected_imgs, valid_points
