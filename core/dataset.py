# main.py
# opengl/blender -> colmap style
# use opengl for Plucker Embedding
# OpenGL (x=Right, y=Up, z=Backward (camera looks along −Z))
# Colmap (x=Right, y=Down, z=Forward (camera looks along +Z))


import os
import cv2
import random
import numpy as np

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
    def __init__(
        self, 
        data_path, 
        depth1_path, # /kaggle/input/10k-dataset-9-views-depth-and-normal
        depth2_path, # /kaggle/input/10k-dataset-9-views-depth-and-normal-2
        depth3_path, # /kaggle/input/10k-dataset-9-views-depth-and-normal-3
        depth4_path, # /kaggle/input/10k-dataset-9-views-depth-and-normal-4
        eval_path=None,
        cfg: Optional[Options] = None, 
        type: Literal['train', 'test', 'val']='train'
    ):
        
        self.data_path = data_path
        self.eval_path = eval_path
        self.cfg = cfg
        self.type = type if type in ['train', 'test', 'val'] else 'train'
        
        self.subfolder_depth = []
        if depth1_path is not None:
            self.subfolder_depth.extend([os.path.join(depth1_path, sub) for sub in os.listdir(depth1_path) 
                            if os.path.isdir(os.path.join(depth1_path, sub))])
        if depth2_path is not None:
            self.subfolder_depth.extend([os.path.join(depth2_path, sub) for sub in os.listdir(depth2_path) 
                            if os.path.isdir(os.path.join(depth2_path, sub))])
        if depth3_path is not None:
            self.subfolder_depth.extend([os.path.join(depth3_path, sub) for sub in os.listdir(depth3_path) 
                            if os.path.isdir(os.path.join(depth3_path, sub))])
        if depth4_path is not None:
            self.subfolder_depth.extend([os.path.join(depth4_path, sub) for sub in os.listdir(depth4_path) 
                          if os.path.isdir(os.path.join(depth4_path, sub))])
        
        self.items_depth = []

        for sub in self.subfolder_depth:
            for item in os.listdir(sub):
                item_path = os.path.join(sub, item)
                self.items_depth.append(item_path)

        # naive split
        if self.type == 'val':
            self.items_depth = self.items_depth[-int(self.cfg.val_size * len(self.items_depth)):]
        elif self.type == 'test':
            self.items_depth = self.items_depth[-int((self.cfg.val_size + self.cfg.test_size) * len(self.items_depth)):-int(self.cfg.val_size * len(self.items_depth) - 1)]
        else:
            self.items_depth = self.items_depth[:int(self.cfg.train_size * len(self.items_depth))]

        # default camera intrinsics
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.cfg.fovy))
        self.projection_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.projection_matrix[0, 0] = 1 / self.tan_half_fov
        self.projection_matrix[1, 1] = 1 / self.tan_half_fov
        self.projection_matrix[2, 2] = (self.cfg.zfar + self.cfg.znear) / (self.cfg.zfar - self.cfg.znear)
        self.projection_matrix[3, 2] = - (self.cfg.zfar * self.cfg.znear) / (self.cfg.zfar - self.cfg.znear)
        self.projection_matrix[2, 3] = 1

        self.certain_input_view_ids = [
            [i for i in range(0, 8)],       # (0 -> 45)
            [i for i in range(16, 24)],     # (90 -> 135)
            [i for i in range(32, 40)],     # (180 -> 225)
            [i for i in range(48, 56)],     # (270 -> 315)
            [64],                           # (top view)
        ]

        self.uncertain_input_view_ids = [
            [i for i in range(8, 16)],      # (45 -> 90)
            [i for i in range(24, 32)],     # (135 -> 180)
            [i for i in range(40, 48)],     # (225 -> 270)
            [i for i in range(56, 64)],     # (315 -> 360)
        ]

        self.test_view_ids = [i for i in range(cfg.num_views_total)]
        self.cam_config = {
            **{i: [0, 360 / (cfg.num_views_total - 1) * i] for i in range(cfg.num_views_total - 1)},
            64: [89.89, 180],
        }

        # --- Fixed eval view camera params (16 views) ---
        self.eval_camera_params = (
            [(30.0, 45.0 * i) for i in range(8)] +    # elev=30, azim=0,45,...,315
            [(60.0, 45.0 * i) for i in range(8)]      # elev=60, azim=0,45,...,315
        )


    def __len__(self):
        return len(self.items_depth)
    
    def find_nonzero_bbox(self, alpha_channel):
        """Find bounding box (ymin, ymax, xmin, xmax) where alpha > 0."""
        ys, xs = np.where(alpha_channel > 0.000001)
        if len(xs) == 0 or len(ys) == 0:  # Fully transparent
            return None
        return ys.min(), ys.max(), xs.min(), xs.max()

    def _resolve_eval_item_path(self, archive_name, item_name):
        if self.eval_path is None:
            return None
        # same archive name
        p = os.path.join(self.eval_path, archive_name, item_name)
        if os.path.isdir(os.path.join(p, "rgb")):
            return p
        # directly under eval_path (no archive)
        p = os.path.join(self.eval_path, item_name)
        if os.path.isdir(os.path.join(p, "rgb")):
            return p
        # search all sub archives in eval_path
        for a in sorted(os.listdir(self.eval_path)):
            cand = os.path.join(self.eval_path, a, item_name)
            if os.path.isdir(os.path.join(cand, "rgb")):
                return cand
        raise FileNotFoundError(f"Cannot find eval views for {item_name} inside {self.eval_path}")

    def __getitem__(self, idx):
        #  NEED TO PROCESS DATA IN .OBJ FORMAT TO (IMAGE-CAMERA POSE) PAIRS
        # your_dataset/
            # ├── uid/
            # │   ├── rgb/
            # │   │   ├── 000.png
            # │   │   ├── 001.png

        item_depth_path = self.items_depth[idx]
        item_name = item_depth_path.split('/')[-1]
        archive_name = item_depth_path.split('/')[-2]
        item_path = os.path.join(self.data_path, archive_name, item_name)
        results = {}

        # ------------------------------------------------------------------ #
        # adaptive input view_ids
        # ------------------------------------------------------------------ #
        num_bonus_views = random.randint(0, 4)
        bonus_views_list = random.choices(self.uncertain_input_view_ids, k=num_bonus_views)
        bonus_views = [random.choice(bonus_views_list[i]) for i in range(len(bonus_views_list))]

        view_ids = [random.choice(self.certain_input_view_ids[i]) for i in range(len(self.certain_input_view_ids))]
        view_ids += bonus_views
        if num_bonus_views < 4:
            view_ids += view_ids[-(self.cfg.num_views_input - len(self.certain_input_view_ids) - num_bonus_views):]
        input_view_ids = sorted(view_ids[:self.cfg.num_views_input])

        origin_elev = self.cam_config[input_view_ids[0]][0]
        origin_azim = self.cam_config[input_view_ids[0]][1]

        # ------------------------------------------------------------------ #
        # load input views from data_path
        # ------------------------------------------------------------------ #
        images = []
        masks = []
        depths = []
        cam_poses = []

        global_ymin, global_ymax = 1e9, -1
        global_xmin, global_xmax = 1e9, -1


        for view_id in input_view_ids:
            image_path = os.path.join(item_path, 'rgb', f'{view_id:03d}.png')
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # [H, W, 4]

            alpha = image[:, :, 3]
            bbox = self.find_nonzero_bbox(alpha)
            if bbox is None:
                bbox = (1e9, -1, 1e9, -1)
            ymin, ymax, xmin, xmax = bbox
            global_ymin = min(global_ymin, ymin)
            global_ymax = max(global_ymax, ymax)
            global_xmin = min(global_xmin, xmin)
            global_xmax = max(global_xmax, xmax)

            try:
                depth = torch.from_numpy(np.load(os.path.join(item_depth_path, 'depth', f'{view_id:03d}.npz'))['depth'])
            except:
                depth = torch.from_numpy(np.load(os.path.join(item_depth_path, 'depth', f'{view_id:03d}.npy')))
            depth = depth.unsqueeze(0)  # [1, H, W]

            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image)

            c2w = torch.from_numpy(orbit_camera(
                -(self.cam_config[view_id][0] - origin_elev),
                (self.cam_config[view_id][1] - origin_azim),
                radius=self.cfg.cam_radius,
                opengl=True
            ))
            c2w[:3, 3] *= self.cfg.cam_radius / 1.5

            image = image.permute(2, 0, 1)      # [4, H, W]
            mask = image[3:4]                   # [1, H, W]
            image = image[:3] * mask + (1 - mask)   # white bg
            image = image[[2, 1, 0]].contiguous()   # BGR -> RGB

            images.append(image)
            masks.append(mask.squeeze(0))
            depths.append(depth)
            cam_poses.append(c2w)

        # ------------------------------------------------------------------ #
        # load output views from eval_path
        # ------------------------------------------------------------------ #
        if self.eval_path is not None:
            eval_images = []
            eval_masks = []
            eval_cam_poses = []

            eval_item_path = self._resolve_eval_item_path(archive_name, item_name)

            for view_idx, (elev, azim) in enumerate(self.eval_camera_params):
                image_path = os.path.join(eval_item_path, 'rgb', f'{view_idx:03d}.png')
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # [H, W, 4]

                alpha = image[:, :, 3]
                bbox = self.find_nonzero_bbox(alpha)
                if bbox is not None:
                    ymin, ymax, xmin, xmax = bbox
                    global_ymin = min(global_ymin, ymin)
                    global_ymax = max(global_ymax, ymax)
                    global_xmin = min(global_xmin, xmin)
                    global_xmax = max(global_xmax, xmax)

                c2w = torch.from_numpy(orbit_camera(
                    -(elev - origin_elev),
                    (azim - origin_azim),
                    radius=self.cfg.cam_radius,
                    opengl=True
                ))
                c2w[:3, 3] *= self.cfg.cam_radius / 1.5

                image = image.astype(np.float32) / 255.0
                image = torch.from_numpy(image).permute(2, 0, 1)   # [4, H, W]
                mask = image[3:4]
                image = image[:3] * mask + (1 - mask)
                image = image[[2, 1, 0]].contiguous()

                eval_images.append(image)
                eval_masks.append(mask.squeeze(0))
                eval_cam_poses.append(c2w)
        else:
            # Fallback: random output views from data_path (for training)
            eval_images = []
            eval_masks = []
            eval_cam_poses = []
            output_view_ids = np.random.permutation(self.test_view_ids).tolist()[:self.cfg.num_views_output]
            for view_id in output_view_ids:
                image_path = os.path.join(item_path, 'rgb', f'{view_id:03d}.png')
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                alpha = image[:, :, 3]
                bbox = self.find_nonzero_bbox(alpha)
                if bbox is not None:
                    ymin, ymax, xmin, xmax = bbox
                    global_ymin = min(global_ymin, ymin)
                    global_ymax = max(global_ymax, ymax)
                    global_xmin = min(global_xmin, xmin)
                    global_xmax = max(global_xmax, xmax)
                c2w = torch.from_numpy(orbit_camera(
                    -(self.cam_config[view_id][0] - origin_elev),
                    (self.cam_config[view_id][1] - origin_azim),
                    radius=self.cfg.cam_radius, opengl=True
                ))
                c2w[:3, 3] *= self.cfg.cam_radius / 1.5
                image = image.astype(np.float32) / 255.0
                image = torch.from_numpy(image).permute(2, 0, 1)
                mask = image[3:4]
                image = image[:3] * mask + (1 - mask)
                image = image[[2, 1, 0]].contiguous()
                eval_images.append(image)
                eval_masks.append(mask.squeeze(0))
                eval_cam_poses.append(c2w)

        # ------------------------------------------------------------------ #
        # shared crop
        # ------------------------------------------------------------------ #
        origin_size = images[0].shape[1]
        if global_ymax < 0 or global_xmax < 0:
            min_res = 0
        else:
            res_ymax = origin_size - global_ymax
            res_ymin = global_ymin
            res_xmax = origin_size - global_xmax
            res_xmin = global_xmin
            min_res = int(min(res_ymax, res_ymin, res_xmax, res_xmin))
            min_res = max(min_res, 0)

        def crop_img(t):   # t: [C, H, W]
            if min_res == 0:
                return t
            s = t.shape[-1]
            return t[:, min_res:(s - min_res), min_res:(s - min_res)]

        def crop_mask(t):  # t: [H, W]
            if min_res == 0:
                return t
            s = t.shape[-1]
            return t[min_res:(s - min_res), min_res:(s - min_res)]

        def crop_depth(t): # t: [1, H, W]
            if min_res == 0:
                return t
            s = t.shape[-1]
            return t[:, min_res:(s - min_res), min_res:(s - min_res)]

        images   = [crop_img(x)   for x in images]
        masks    = [crop_mask(x)  for x in masks]
        depths   = [crop_depth(x) for x in depths]
        eval_images = [crop_img(x)  for x in eval_images]
        eval_masks  = [crop_mask(x) for x in eval_masks]

        # Padding input if not enough views
        view_cnt = len(images)
        if view_cnt < self.cfg.num_views_input:
            print(f'[WARN] dataset {item_path}: not enough valid views, only {view_cnt} views found!')
            n = self.cfg.num_views_input - view_cnt
            images    = images    + [images[-1]]    * n
            masks     = masks     + [masks[-1]]     * n
            depths    = depths    + [depths[-1]]    * n
            cam_poses = cam_poses + [cam_poses[-1]] * n

        images    = torch.stack(images, dim=0)      # [V_in, 3, H, W]
        masks     = torch.stack(masks, dim=0)       # [V_in, H, W]
        depths    = torch.stack(depths, dim=0)      # [V_in, 1, H, W]
        cam_poses = torch.stack(cam_poses, dim=0)   # [V_in, 4, 4]

        eval_images    = torch.stack(eval_images, dim=0)     # [V_out, 3, H, W]
        eval_masks     = torch.stack(eval_masks, dim=0)      # [V_out, H, W]
        eval_cam_poses = torch.stack(eval_cam_poses, dim=0)  # [V_out, 4, 4]

        # ------------------------------------------------------------------ #
        # resize input
        # ------------------------------------------------------------------ #
        images_input = F.interpolate(images.clone(), size=(self.cfg.input_size, self.cfg.input_size), mode='bilinear', align_corners=False)
        cam_poses_input = cam_poses.clone()
        depths_input = F.interpolate(depths.clone(), size=(self.cfg.splat_size, self.cfg.splat_size), mode='nearest')
        masks_input  = F.interpolate(masks.clone().unsqueeze(1), size=(self.cfg.splat_size, self.cfg.splat_size), mode='bilinear', align_corners=False)

        # data augmentation
        # if self.type == 'train':
        #     # if random.random() < self.cfg.prob_grid_distortion:
        #     #     images_input[1:] = grid_distortion(images_input[1:])
        #     if random.random() < self.cfg.prob_cam_jitter:
        #         cam_poses_input[1:] = orbit_camera_jitter(cam_poses_input[1:])

        images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        # build Plucker rays for input views
        rays_embeddings = []
        for i in range(self.cfg.num_views_input):
            rays_o, rays_d = get_rays(cam_poses_input[i], self.cfg.input_size, self.cfg.input_size, self.cfg.fovy)
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1)  # [h, w, 6]
            rays_embeddings.append(rays_plucker)

        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous()  # [V_in, 6, h, w]
        final_input = torch.cat([images_input, rays_embeddings], dim=1)  # [V_in, 9, H, W]

        results['input'] = final_input
        results['cam_poses_input'] = cam_poses_input
        results['depths_input'] = depths_input
        results['masks_input'] = masks_input

        # ------------------------------------------------------------------ #
        # output (eval views) processing
        # ------------------------------------------------------------------ #
        if not self.cfg.self_supervised:
            results['images_output'] = F.interpolate(eval_images.clone(), (self.cfg.output_size, self.cfg.output_size), mode='bilinear', align_corners=False)
            results['masks_output']  = F.interpolate(eval_masks.clone().unsqueeze(1), (self.cfg.output_size, self.cfg.output_size), mode='bilinear', align_corners=False)
            cam_poses_out = eval_cam_poses.clone()
        else:
            # self_supervised: supervise trên chính input views
            results['images_output'] = F.interpolate(images.clone(), (self.cfg.output_size, self.cfg.output_size), mode='bilinear', align_corners=False)
            results['masks_output']  = F.interpolate(masks.clone().unsqueeze(1), (self.cfg.output_size, self.cfg.output_size), mode='bilinear', align_corners=False)
            cam_poses_out = cam_poses.clone()

        # OpenGL -> COLMAP for gaussian renderer
        cam_poses_out[:, :3, 1:3] *= -1

        cam_view      = torch.inverse(cam_poses_out).transpose(1, 2)   # [V_out, 4, 4]
        cam_view_proj = cam_view @ self.projection_matrix               # [V_out, 4, 4]
        cam_pos       = -cam_poses_out[:, :3, 3]                        # [V_out, 3]

        results['cam_view_output']      = cam_view
        results['cam_view_proj_output'] = cam_view_proj
        results['cam_pos_output']       = cam_pos
        results['object_id']            = f"{archive_name}/{item_name}"

        # results = {
        #     'input':                [V_in, 9, input_size, input_size]
        #     'cam_poses_input':      [V_in, 4, 4]
        #     'depths_input':         [V_in, 1, splat_size, splat_size]
        #     'masks_input':          [V_in, 1, splat_size, splat_size]
        #     'images_output':        [V_out, 3, output_size, output_size]
        #     'masks_output':         [V_out, 1, output_size, output_size]
        #     'cam_view_output':      [V_out, 4, 4]  (colmap)
        #     'cam_view_proj_output': [V_out, 4, 4]  (colmap)
        #     'cam_pos_output':       [V_out, 3]     (colmap)
        #     'object_id':            str
        # }
        return results