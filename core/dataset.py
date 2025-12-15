# main.py
# opengl/blender -> colmap style
# use opengl for Plucker Embedding
# OpenGL (x=Right, y=Up, z=Backward (camera looks along −Z))
# Colmap (x=Right, y=Down, z=Forward (camera looks along +Z))


import os
import cv2
import random
import numpy as np
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from scipy.stats import gamma
from typing import Tuple, Literal, Dict, Optional



from kiui.cam import orbit_camera
from core.model_config import Options
from core.utils import get_rays, grid_distortion, orbit_camera_jitter

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class ObjaverseDataset(Dataset):
    def __init__(self, data_path, cfg: Options, type: Literal['train', 'test', 'val']='train'):
        
        self.data_path = data_path
        self.cfg = cfg
        self.type = type if type in ['train', 'test', 'val'] else 'train'

        
        self.subfolder = [os.path.join(data_path, sub) for sub in os.listdir(data_path) 
                          if os.path.isdir(os.path.join(data_path, sub))]
        
        self.items = []

        for sub in self.subfolder:
            for item in os.listdir(sub):
                item_path = os.path.join(sub, item)
                self.items.append(item_path)

        # naive split
        if self.type == 'val':
            self.items = self.items[-int(self.cfg.val_size * len(self.items)):]
        elif self.type == 'test':
            self.items = self.items[-int((self.cfg.val_size + self.cfg.test_size) * len(self.items)):-int(self.cfg.val_size * len(self.items) - 1)]
        else:
            self.items = self.items[:int(self.cfg.train_size * len(self.items))]

        # default camera intrinsics
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.cfg.fovy))
        self.projection_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.projection_matrix[0, 0] = 1 / self.tan_half_fov
        self.projection_matrix[1, 1] = 1 / self.tan_half_fov
        self.projection_matrix[2, 2] = (self.cfg.zfar + self.cfg.znear) / (self.cfg.zfar - self.cfg.znear)
        self.projection_matrix[3, 2] = - (self.cfg.zfar * self.cfg.znear) / (self.cfg.zfar - self.cfg.znear)
        self.projection_matrix[2, 3] = 1

        self.input_view_ids = [0, 2, 4, 6,         # L1
                                                   # L2
                                                   # L3
                               24,]                # L4
        
        self.test_view_ids = [i for i in range(cfg.num_views_total)]
        self.cam_config = {
            # this is the params to pass into kiui.orbit_camera() function
            # (elevation, azimuth)
            # elevation = 0
            0: [0, 0],
            1: [0, 45],
            2: [0, 90],
            3: [0, 135],
            4: [0, 180],
            5: [0, 225],
            6: [0, 270],
            7: [0, 315],

            # elevation = 30
            8: [30, 0],
            9: [30, 45],
            10: [30, 90],
            11: [30, 135],
            12: [30, 180],
            13: [30, 225],
            14: [30, 270],
            15: [30, 315],

            # elevation = 60
            16: [60, 0],
            17: [60, 45],
            18: [60, 90],
            19: [60, 135],
            20: [60, 180],
            21: [60, 225],
            22: [60, 270],
            23: [60, 315],

            # elevation = 90,
            24: [89.89, 180]
        }


    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        #  NEED TO PROCESS DATA IN .OBJ FORMAT TO (IMAGE-CAMERA POSE) PAIRS
        # your_dataset/
            # ├── uid/
            # │   ├── rgb/
            # │   │   ├── 000.png
            # │   │   ├── 001.png
            # │   ├── pose/
            # │   │   ├── 000.txt
            # │   │   ├── 001.txt

        assert len(self.input_view_ids) == self.cfg.num_views_input

        item_path = self.items[idx]
        input_0 = [(os.path.join(item_path, 'elev_0_azim_0_dist_1.5', 'rgb', f), 0) 
                   for f in os.listdir(os.path.join(item_path, 'elev_0_azim_0_dist_1.5', 'rgb')) 
                   if os.path.isfile(os.path.join(item_path, 'elev_0_azim_0_dist_1.5', 'rgb', f))]
        
        input_2 = [(os.path.join(item_path, 'elev_0_azim_90_dist_1.5', 'rgb', f), 2) 
                   for f in os.listdir(os.path.join(item_path, 'elev_0_azim_90_dist_1.5', 'rgb')) 
                   if os.path.isfile(os.path.join(item_path, 'elev_0_azim_90_dist_1.5', 'rgb', f))]
        
        input_4 = [(os.path.join(item_path, 'elev_0_azim_180_dist_1.5', 'rgb', f), 4) 
                   for f in os.listdir(os.path.join(item_path, 'elev_0_azim_180_dist_1.5', 'rgb')) 
                   if os.path.isfile(os.path.join(item_path, 'elev_0_azim_180_dist_1.5', 'rgb', f))]
        
        input_6 = [(os.path.join(item_path, 'elev_0_azim_270_dist_1.5', 'rgb', f), 6) 
                   for f in os.listdir(os.path.join(item_path, 'elev_0_azim_270_dist_1.5', 'rgb')) 
                   if os.path.isfile(os.path.join(item_path, 'elev_0_azim_270_dist_1.5', 'rgb', f))]
        
        input_24 = [(os.path.join(item_path, 'elev_89.89_azim_180_dist_1.5', 'rgb', f), 24) 
                    for f in os.listdir(os.path.join(item_path, 'elev_89.89_azim_180_dist_1.5', 'rgb')) 
                    if os.path.isfile(os.path.join(item_path, 'elev_89.89_azim_180_dist_1.5', 'rgb', f))]
        
        ref = [(os.path.join(item_path, 'reference', 'rgb', f), -1) 
               for f in os.listdir(os.path.join(item_path, 'reference', 'rgb')) 
               if os.path.isfile(os.path.join(item_path, 'reference', 'rgb', f))]

        results = {}

        # load num_views images
        images = []
        masks = []
        cam_poses = []
        
        image_paths = [random.choice(input_0), random.choice(input_2), random.choice(input_4), 
                    random.choice(input_6), random.choice(input_24)] + np.random.permutation(ref).tolist()
        image_paths = image_paths[:(self.cfg.num_views_input + self.cfg.num_views_output)]

        for (image_path, view_id) in image_paths:
            if view_id == -1:
                filename = image_path.split("/")[-1]

                pattern = r"elev_([-+]?[0-9]*\.?[0-9]+)_azim_([-+]?[0-9]*\.?[0-9]+)"
                match = re.search(pattern, filename)

                if match:
                    elev = int(match.group(1))
                    azim = int(match.group(2))

                    view_id = next(
                        (k for k, v in self.cam_config.items() if v == [elev, azim]),
                        None
                    )
                else:
                    raise ValueError(f"Cannot parse elev/azim from {image_path}")

            # try:
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # shape: [512, 512, 4]

            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image)  # shape: [H, W, C]
            
            c2w = torch.from_numpy(orbit_camera(-self.cam_config[view_id][0], self.cam_config[view_id][1], radius=self.cfg.cam_radius, opengl=True))

            # scale up radius to make model make scale predictions
            c2w[:3, 3] *= self.cfg.cam_radius / 1.5 # 1.5 is the default scale of the dataset
        
            # Background removing
            image = image.permute(2, 0, 1) # [4, 512, 512]
            mask = image[3:4] # [1, 512, 512]
            image = image[:3] * mask + (1 - mask) # [3, 512, 512], to white bg
            image = image[[2,1,0]].contiguous() # bgr to rgb

            images.append(image)
            masks.append(mask.squeeze(0))
            cam_poses.append(c2w)

        view_cnt = len(images)
        if view_cnt < (self.cfg.num_views_input + self.cfg.num_views_output):
            print(f'[WARN] dataset {item_path}: not enough valid views, only {view_cnt} views found!')
            # Padding to be enough views
            n = (self.cfg.num_views_input + self.cfg.num_views_output) - view_cnt
            images = images + [images[-1]] * n
            masks = masks + [masks[-1]] * n
            cam_poses = cam_poses + [cam_poses[-1]] * n

        images = torch.stack(images, dim=0)     # [V, C, H, W]
        masks = torch.stack(masks, dim=0)       # [V, H, W]
        cam_poses = torch.stack(cam_poses, dim=0)  # [V, 4, 4]

        # # normalized camera feats as in paper (transform the first pose to a fixed position)
        # transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.cfg.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[0])
        # cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]

        # resize input images
        images_input = F.interpolate(images[:len(self.input_view_ids)].clone(), size=(self.cfg.input_size, self.cfg.input_size), mode='bilinear', align_corners=False)   # [V, C, H, W]
        cam_poses_input = cam_poses[:len(self.input_view_ids)].clone()
        
        # # data augmentation
        # if self.type == 'train':
        #     # if random.random() < self.cfg.prob_grid_distortion:
        #     #     images_input[1:] = grid_distortion(images_input[1:])
        #     if random.random() < self.cfg.prob_cam_jitter:
        #         cam_poses_input[1:] = orbit_camera_jitter(cam_poses_input[1:])

        images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        # build rays for input views
        rays_embeddings = []
        for i in range(len(self.input_view_ids)):
            rays_o, rays_d = get_rays(cam_poses_input[i], self.cfg.input_size, self.cfg.input_size, self.cfg.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)

        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V=9, 6, h, w]
        final_input = torch.cat([images_input, rays_embeddings], dim=1) # [V=9, 9, H, W]

        results['input'] = final_input
        results['cam_poses_input'] = cam_poses_input

        # resize ground-truth images, still in range [0, 1]
        results['images_output'] = F.interpolate(images[len(self.input_view_ids):].clone(), (self.cfg.output_size, self.cfg.output_size), mode='bilinear', align_corners=False)
        results['masks_output'] = F.interpolate(masks[len(self.input_view_ids):].clone().unsqueeze(1), (self.cfg.output_size, self.cfg.output_size), mode='bilinear', align_corners=False)

        cam_poses = cam_poses[len(self.input_view_ids):].clone()
        # opengl to colmap camera for gaussian renderer
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction

        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2)     # World-to-camera matrix: [V, 4, 4] (row-vector)
        cam_view_proj = cam_view @ self.projection_matrix     # world-to-clip matrix: [V, 4, 4]
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]
        
        results['cam_view_output'] = cam_view
        results['cam_view_proj_output'] = cam_view_proj
        results['cam_pos_output'] = cam_pos

        # results = {
        #     [C, H, W]
        #     'input': ...,             (processed input images 5x9x256x256)
        #     'cam_poses_input': ...,   
        #     'images_output': ...,     (9x3x512x512)
        #     'masks_output': ...,      (.......)
        #     'cam_view_output': ...,          (colmap coordinate)
        #     'cam_view_proj_output': ...,     (colmap coordinate)
        #     'cam_pos_output': ...,           (colmap coordinate)
        # }
        return results