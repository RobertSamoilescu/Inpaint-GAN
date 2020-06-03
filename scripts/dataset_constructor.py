#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

# get_ipython().run_line_magic('matplotlib', 'inline')
from nuscenes.nuscenes import NuScenes
from util.depth_map_utils import *
from util.generator import random_walk
from util.inverse_warp import *
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle as pkl
from pyquaternion import Quaternion
from multiprocessing import Pool, cpu_count
from itertools import count


# load dataset
nusc = NuScenes(version="v1.0-test", dataroot='/mnt/storage/workspace/roberts/nuscene/v1.0-test', verbose=True)


def get_camera_extrinsic(cam):
    calib_sensor = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    translation = calib_sensor['translation']
    rotation = calib_sensor['rotation']

    q = Quaternion(rotation)
    R = q.rotation_matrix
    T = np.array(translation).reshape(-1, 1)

    # rotation to transform car's coordinate system to
    # camera coordinate system
    alphay = np.pi / 2
    alphaz = -np.pi / 2
    
    Ry = np.array([
        [np.cos(alphay), 0, -np.sin(alphay)],
        [0, 1, 0],
        [np.sin(alphay), 0, np.cos(alphay)]
    ])

    Rz = np.array([
        [np.cos(alphaz), np.sin(alphaz), 0],
        [-np.sin(alphaz), np.cos(alphaz), 0],
        [0, 0, 1],
    ])

    Rot = Rz @ Ry
    R = Rot @ R
    T = Rot @ T

    camera_extrinsic = np.hstack((R, T))
    return camera_extrinsic


def get_depth_map(sample, cam, img):
    # depth map
    lidar = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    points, coloring, im = nusc.explorer.map_pointcloud_to_image(
        pointsensor_token=lidar['token'],
        camera_token=cam['token']
    )

    depth_map = np.zeros((img.shape[0], img.shape[1])).astype(np.float32)
    for i, (x, y) in enumerate(zip(points[0], points[1])):
        int_x, int_y = np.floor([x, y]).astype(np.int)
        int_x = np.clip(int_x, 0, depth_map.shape[1] - 1)
        int_y = np.clip(int_y, 0, depth_map.shape[0] - 1)
        depth_map[int_y, int_x] = coloring[i]
    
    
    # Fast fill with Gaussian blur @90Hz (paper result)
    extrapolate = True
    blur_type = 'gaussian'
    
    depth_map = fill_in_fast(
        depth_map, 
        extrapolate=extrapolate, 
        blur_type=blur_type
    )    
    
    return depth_map


def random_transformation(imgs, intrinsics, extrinsics, depths):
    """
    Generate random transformation
    @param imgs:         [B, 3, H, W]
    @param depths:       [B, H, W]
    @param intrinsics:   [B, 3, 3]
    @param extriniscs:   [B, 3, 4]
    """
    # sample random transformation
    B = imgs.shape[0]
    poses = torch.zeros(B, 6).double()
    tx = 1.0 * 2 * (torch.rand(B) - 0.5)
    ry = 0.25 * 2 * (torch.rand(B) - 0.5)
    poses[:, 0], poses[:, 4] = tx, ry
    
    # apply transformation
    projected_imgs, valid_points = forward_warp(
        img=imgs, 
        depth=depths, 
        pose=poses, 
        intrinsics=intrinsics,
        extrinsics=None
    )
    
    # mask of valid points
    valid_points = (valid_points * (depths > 0).type(torch.long)).double()
    projected_imgs = projected_imgs * valid_points.unsqueeze(1)
    return projected_imgs, valid_points


def process_scene(scene, idx):
    current_sample_token = scene['first_sample_token']
    last_sample_token = scene['last_sample_token']
    name = scene['name']
    
    for i in count():
        # load current sample
        current_sample = nusc.get('sample', current_sample_token)
       
        # get cameras
        cam_front = nusc.get('sample_data', current_sample['data']['CAM_FRONT'])
        cam_front_right = nusc.get('sample_data', current_sample['data']['CAM_FRONT_RIGHT'])
        cam_front_left = nusc.get('sample_data', current_sample['data']['CAM_FRONT_LEFT'])
        cam_back = nusc.get('sample_data', current_sample['data']['CAM_BACK'])
        cam_back_right = nusc.get('sample_data', current_sample['data']['CAM_BACK_RIGHT'])
        cam_back_left = nusc.get('sample_data', current_sample['data']['CAM_BACK_LEFT'])
        
        # get images path and intrinsic matrix of central camera
        img_front_path, _, cam_front_intrinsic =             nusc.explorer.nusc.get_sample_data(cam_front['token'])
        img_front_right_path, _, cam_front_right_intrinsic =             nusc.explorer.nusc.get_sample_data(cam_front_right['token'])
        img_front_left_path, _, cam_front_left_intrinsic =             nusc.explorer.nusc.get_sample_data(cam_front_left['token'])
        img_back_path, _, cam_back_intrinsic =             nusc.explorer.nusc.get_sample_data(cam_back['token'])
        img_back_right_path, _, cam_back_right_intrinsic =             nusc.explorer.nusc.get_sample_data(cam_back_right['token'])
        img_back_left_path, _, cam_back_left_intrinsic =             nusc.explorer.nusc.get_sample_data(cam_back_left['token'])

        # get images
        img_front = np.array(Image.open(img_front_path))
        img_front_right = np.array(Image.open(img_front_right_path))
        img_front_left = np.array(Image.open(img_front_left_path))
        img_back = np.array(Image.open(img_back_path))
        img_back_right = np.array(Image.open(img_back_right_path))
        img_back_left = np.array(Image.open(img_back_left_path))
        
        # save original dimensions
        orig_height, orig_width, _ = img_front.shape
        
        # get front camera extrinsic
        cam_front_extrinsic = get_camera_extrinsic(cam_front)
        cam_front_right_extrinsic = get_camera_extrinsic(cam_front_right)
        cam_front_left_extrinisc = get_camera_extrinsic(cam_front_left)
        cam_back_extrinsic = get_camera_extrinsic(cam_back)
        cam_back_right_extrinsic = get_camera_extrinsic(cam_back_right)
        cam_back_left_extrinsic = get_camera_extrinsic(cam_back_left)

        # get depth map
        depth_map_front = get_depth_map(current_sample, cam_front, img_front)
        depth_map_front_right = get_depth_map(current_sample, cam_front_right, img_front_right)
        depth_map_front_left = get_depth_map(current_sample, cam_front_left, img_front_left)
        depth_map_back = get_depth_map(current_sample, cam_back, img_back)
        depth_map_back_right = get_depth_map(current_sample, cam_back_right, img_back_right)
        depth_map_back_left = get_depth_map(current_sample, cam_back_left, img_back_left)
            
        # resize image and depth_map
        height, width = 128, 256
        
        imgs = [
            cv2.resize(img_front, (width, height)),
            cv2.resize(img_front_right, (width, height)),
            cv2.resize(img_front_left, (width, height)),
            cv2.resize(img_back, (width, height)),
            cv2.resize(img_back_right, (width, height)),
            cv2.resize(img_back_left, (width, height)),            
        ]
        
        depths = [
            cv2.resize(depth_map_front, (width, height)),
            cv2.resize(depth_map_front_right, (width, height)),
            cv2.resize(depth_map_front_left, (width, height)),
            cv2.resize(depth_map_back, (width, height)),
            cv2.resize(depth_map_back_right, (width, height)),
            cv2.resize(depth_map_back_left, (width, height)),
        ]
        
        # update camera intrinsics according to the new size
        S = np.array([
            [width/orig_width, 0, 0],
            [0, height/orig_height, 0],
            [0, 0, 1]
        ])
        
        intrinsics = [
            S @ cam_front_intrinsic,
            S @ cam_front_right_intrinsic,
            S @ cam_front_left_intrinsic,
            S @ cam_back_intrinsic,
            S @ cam_back_right_intrinsic,
            S @ cam_back_left_intrinsic,
        ]
        
        extrinsics = [
            cam_front_extrinsic,
            cam_front_right_extrinsic,
            cam_front_left_extrinisc,
            cam_back_extrinsic,
            cam_back_right_extrinsic,
            cam_back_left_extrinsic,
        ]
        
        masks = []
        for j in range(len(imgs)):
            _, mask = random_transformation(
                imgs=torch.tensor(imgs[j].transpose(2, 0, 1)).unsqueeze(0).double(), 
                intrinsics=torch.tensor(intrinsics[j]).unsqueeze(0).double(), 
                extrinsics=torch.tensor(extrinsics[j]).unsqueeze(0).double(), 
                depths=torch.tensor(depths[j]).unsqueeze(0).double()
            )
            mask = 255 * mask.to(torch.uint8).squeeze(0).numpy()
            masks.append(mask)
            
            
        # cropp sky and lateral
        upper_limit, lower_limit = height // 2, height
        left_limit, right_limit = width // 4, -width // 4
        
        for j in range(len(imgs)):
            # crop imgs
            imgs[j] = imgs[j][upper_limit:lower_limit, left_limit:right_limit]
            depths[j] = depths[j][upper_limit:lower_limit, left_limit:right_limit]
            masks[j] = masks[j][upper_limit:lower_limit, left_limit:right_limit]
            
            # intrinsics update
            cropped_height, cropped_width, _ = imgs[j].shape
            scale = np.array([
                [cropped_width/width, 0, 0],
                [0, cropped_height/height, 0],
                [0, 0, 1]
            ])
            intrinsics[j] = scale @ intrinsics[j]
            
                
        # save data to files
        for j in range(len(imgs)):
            cv2.imwrite("/mnt/storage/workspace/roberts/nuscene/dataset/imgs/%s.%d.%d.%d.png" % (name, idx, i, j), imgs[j][:, :, ::-1])
            cv2.imwrite("/mnt/storage/workspace/roberts/nuscene/dataset/masks/%s.%d.%d.%d.png" % (name, idx, i, j), masks[j])
            
            with open("/mnt/storage/workspace/roberts/nuscene/dataset/depths/%s.%d.%d.%d.pkl" % (name, idx, i, j), "wb") as f:
                pkl.dump(depths[j], f)
            with open("/mnt/storage/workspace/roberts/nuscene/dataset/intrinsics/%s.%d.%d.%d.pkl" % (name, idx, i, j), "wb") as f:
                pkl.dump(intrinsics[j], f)
            with open("/mnt/storage/workspace/roberts/nuscene/dataset/extrinsics/%s.%d.%d.%d.pkl" % (name, idx, i, j), "wb") as f:
                pkl.dump(extrinsics[j], f)
                
        # check if last sample was processed
        if current_sample_token == last_sample_token:
            break
            
        # go to the next sample
        current_sample_token = current_sample['next']


pool = Pool(processes=cpu_count())

for idx, scene in enumerate(nusc.scene):
	pool.apply_async(process_scene, (scene, idx))

pool.close()
pool.join()
