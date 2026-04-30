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
        eval_path,
        cfg: Options, 
        type: Literal['train', 'test', 'val']='train'
    ):
        
        self.data_path = data_path
        self.eval_path = eval_path
        self.cfg = cfg
        self.type = type if type in ['train', 'test', 'val'] else 'train'
        
        # depth roots
        self.subfolder_depth = []
        for root in [depth1_path, depth2_path, depth3_path, depth4_path]:
            if root is not None and os.path.isdir(root):
                self.subfolder_depth.extend(
                    [
                        os.path.join(root, sub)
                        for sub in sorted(os.listdir(root))
                        if os.path.isdir(os.path.join(root, sub))
                    ]
                )

        self.items_depth = []
        for sub in sorted(self.subfolder_depth):
            for item in sorted(os.listdir(sub)):
                item_path = os.path.join(sub, item)
                if os.path.isdir(os.path.join(item_path, "depth")):
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

        # 9 fixed input views from the 65-view dataset:
        # - 8 equally spaced equator views: 000, 008, ..., 056
        # - 1 top view: 064
        self.input_view_ids = [0, 8, 16, 24, 32, 40, 48, 56, 64]
        self.input_camera_params = (
            [(0.0, 45.0 * i) for i in range(8)] +
            [(89.89, 180.0)]
        )

        # 16 views eval 
        self.eval_camera_params = (
            [(30.0, 45.0 * i) for i in range(8)] +
            [(60.0, 45.0 * i) for i in range(8)]
        )

        if self.cfg.num_views_input != len(self.input_view_ids):
            raise ValueError(
                f"This 9-view dataset expects cfg.num_views_input={len(self.input_view_ids)}, "
                f"but got {self.cfg.num_views_input}. Please run with --num_views_input 9."
            )
        if self.cfg.num_views_output != len(self.eval_camera_params):
            raise ValueError(
                f"This eval setup expects cfg.num_views_output={len(self.eval_camera_params)}, "
                f"but got {self.cfg.num_views_output}. Please run with --num_views_output 16."
            )


    def __len__(self):
        return len(self.items_depth)
    
    @staticmethod
    def find_nonzero_bbox(alpha_channel):
        """Find bounding box (ymin, ymax, xmin, xmax) where alpha > 0."""
        ys, xs = np.where(alpha_channel > 0.000001)
        if len(xs) == 0 or len(ys) == 0:  # Fully transparent
            return None
        return ys.min(), ys.max(), xs.min(), xs.max()

    @staticmethod
    def _load_rgba(path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(path)

        # if RGB only, append alpha=1
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.shape[-1] == 3:
            alpha = np.ones(img.shape[:2], dtype=np.uint8) * 255
            img = np.concatenate([img, alpha[..., None]], axis=-1)

        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # [4, H, W], BGRA
        return img

    @staticmethod
    def _load_depth(depth_dir, view_id):
        npz_path = os.path.join(depth_dir, f"{view_id:03d}.npz")
        npy_path = os.path.join(depth_dir, f"{view_id:03d}.npy")

        if os.path.exists(npz_path):
            f = np.load(npz_path)
            if "depth" in f:
                arr = f["depth"]
            elif "data" in f:
                arr = f["data"]
            else:
                raise KeyError(f"No 'depth' or 'data' key in {npz_path}")
        elif os.path.exists(npy_path):
            arr = np.load(npy_path)
        else:
            raise FileNotFoundError(f"Depth not found for view {view_id:03d} in {depth_dir}")

        arr = arr.astype(np.float32)
        if arr.ndim == 1:
            side = int(np.sqrt(arr.shape[0]))
            arr = arr.reshape(side, side)
        elif arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = arr[0]

        return torch.from_numpy(arr).unsqueeze(0)  # [1, H, W]

    def _make_pose(self, elev, azim, origin_elev=0.0, origin_azim=0.0):
        c2w = torch.from_numpy(
            orbit_camera(
                -(elev - origin_elev),
                (azim - origin_azim),
                radius=self.cfg.cam_radius,
                opengl=True,
            )
        )
        c2w[:3, 3] *= self.cfg.cam_radius / 1.5  # keep original scale logic
        return c2w

    def _resolve_eval_item_path(self, archive_name, item_name):
        # same archive name
        p = os.path.join(self.eval_path, archive_name, item_name)
        if os.path.isdir(os.path.join(p, "rgb")):
            return p
        # directly under eval_path
        p = os.path.join(self.eval_path, item_name)
        if os.path.isdir(os.path.join(p, "rgb")):
            return p
        # search all archives under eval_path
        for a in sorted(os.listdir(self.eval_path)):
            cand = os.path.join(self.eval_path, a, item_name)
            if os.path.isdir(os.path.join(cand, "rgb")):
                return cand

        raise FileNotFoundError(f"Cannot find eval views for {item_name} inside {self.eval_path}")
    
    def __getitem__(self, idx):
        item_depth_path = self.items_depth[idx]
        item_name = os.path.basename(item_depth_path)
        archive_name = os.path.basename(os.path.dirname(item_depth_path))

        item_path = os.path.join(self.data_path, archive_name, item_name)
        eval_item_path = self._resolve_eval_item_path(archive_name, item_name)
        depth_dir = os.path.join(item_depth_path, "depth")

        results = {}

        input_rgba_list = []
        input_depths = []
        input_cam_poses = []

        output_rgba_list = []
        output_cam_poses = []

        origin_elev, origin_azim = self.input_camera_params[0]

        global_ymin, global_ymax = 1e9, -1
        global_xmin, global_xmax = 1e9, -1

        # load 9 input views 
        for view_id, (elev, azim) in zip(self.input_view_ids, self.input_camera_params):
            image_path = os.path.join(item_path, "rgb", f"{view_id:03d}.png")
            rgba = self._load_rgba(image_path)          # [4, H, W]
            depth = self._load_depth(depth_dir, view_id)
            c2w = self._make_pose(elev, azim, origin_elev, origin_azim)

            alpha = rgba[3].cpu().numpy()
            bbox = self.find_nonzero_bbox(alpha)
            if bbox is not None:
                ymin, ymax, xmin, xmax = bbox
                global_ymin = min(global_ymin, ymin)
                global_ymax = max(global_ymax, ymax)
                global_xmin = min(global_xmin, xmin)
                global_xmax = max(global_xmax, xmax)

            input_rgba_list.append(rgba)
            input_depths.append(depth)
            input_cam_poses.append(c2w)

        #  load 16 eval views
        for view_idx, (elev, azim) in enumerate(self.eval_camera_params):
            image_path = os.path.join(eval_item_path, "rgb", f"{view_idx:03d}.png")
            rgba = self._load_rgba(image_path)          # [4, H, W]
            c2w = self._make_pose(elev, azim, origin_elev, origin_azim)

            alpha = rgba[3].cpu().numpy()
            bbox = self.find_nonzero_bbox(alpha)
            if bbox is not None:
                ymin, ymax, xmin, xmax = bbox
                global_ymin = min(global_ymin, ymin)
                global_ymax = max(global_ymax, ymax)
                global_xmin = min(global_xmin, xmin)
                global_xmax = max(global_xmax, xmax)

            output_rgba_list.append(rgba)
            output_cam_poses.append(c2w)

        if len(input_rgba_list) == 0:
            raise ValueError(f"No input views loaded for {archive_name}/{item_name}")
        if len(output_rgba_list) == 0:
            raise ValueError(f"No eval views loaded for {archive_name}/{item_name}")

        # shared crop over both input + eval
        origin_size = input_rgba_list[0].shape[1]  # H, assume square
        if global_ymax < 0 or global_xmax < 0:
            min_res = 0
        else:
            res_ymax = origin_size - global_ymax
            res_ymin = global_ymin
            res_xmax = origin_size - global_xmax
            res_xmin = global_xmin
            min_res = int(min(res_ymax, res_ymin, res_xmax, res_xmin))
            min_res = max(min_res, 0)

        def crop_rgba(rgba):
            if min_res == 0:
                return rgba
            s = rgba.shape[-1]
            return rgba[:, min_res:(s - min_res), min_res:(s - min_res)]

        def crop_depth(depth):
            if min_res == 0:
                return depth
            s = depth.shape[-1]
            return depth[:, min_res:(s - min_res), min_res:(s - min_res)]

        images_in = []
        masks_in = []
        depths_in = []
        cam_poses_in = []

        for rgba, depth, c2w in zip(input_rgba_list, input_depths, input_cam_poses):
            rgba = crop_rgba(rgba)
            depth = crop_depth(depth)

            mask = rgba[3:4]  # [1, H, W]
            image = rgba[:3] * mask + (1 - mask)  # white bg
            image = image[[2, 1, 0]].contiguous()  # BGR -> RGB

            images_in.append(image)
            masks_in.append(mask.squeeze(0))
            depths_in.append(depth)
            cam_poses_in.append(c2w)

        images_out = []
        masks_out = []
        cam_poses_out = []

        for rgba, c2w in zip(output_rgba_list, output_cam_poses):
            rgba = crop_rgba(rgba)

            mask = rgba[3:4]
            image = rgba[:3] * mask + (1 - mask)
            image = image[[2, 1, 0]].contiguous()

            images_out.append(image)
            masks_out.append(mask.squeeze(0))
            cam_poses_out.append(c2w)

        images_in = torch.stack(images_in, dim=0)          # [V_in, 3, H, W]
        masks_in = torch.stack(masks_in, dim=0)            # [V_in, H, W]
        depths_in = torch.stack(depths_in, dim=0)          # [V_in, 1, H, W]
        cam_poses_in = torch.stack(cam_poses_in, dim=0)    # [V_in, 4, 4]

        images_out = torch.stack(images_out, dim=0)        # [V_out, 3, H, W]
        masks_out = torch.stack(masks_out, dim=0)          # [V_out, H, W]
        cam_poses_out = torch.stack(cam_poses_out, dim=0)  # [V_out, 4, 4]

        # ---------- resize input ----------
        images_input = F.interpolate(
            images_in.clone(),
            size=(self.cfg.input_size, self.cfg.input_size),
            mode="bilinear",
            align_corners=False,
        )
        depths_input = F.interpolate(
            depths_in.clone(),
            size=(self.cfg.splat_size, self.cfg.splat_size),
            mode="nearest",
        )
        masks_input = F.interpolate(
            masks_in.clone().unsqueeze(1),
            size=(self.cfg.splat_size, self.cfg.splat_size),
            mode="bilinear",
            align_corners=False,
        )

        images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        # build rays for input views
        rays_embeddings = []
        for i in range(self.cfg.num_views_input):
            rays_o, rays_d = get_rays(
                cam_poses_in[i],
                self.cfg.input_size,
                self.cfg.input_size,
                self.cfg.fovy,
            )  # [h, w, 3]
            rays_plucker = torch.cat(
                [torch.cross(rays_o, rays_d, dim=-1), rays_d],
                dim=-1
            )  # [h, w, 6]
            rays_embeddings.append(rays_plucker)

        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous()
        final_input = torch.cat([images_input, rays_embeddings], dim=1)  # [V_in, 9, H, W]

        results["input"] = final_input
        results["cam_poses_input"] = cam_poses_in
        results["depths_input"] = depths_input
        results["masks_input"] = masks_input

        # output / supervision
        if not self.cfg.self_supervised:
            results["images_output"] = F.interpolate(
                images_out.clone(),
                (self.cfg.output_size, self.cfg.output_size),
                mode="bilinear",
                align_corners=False,
            )
            results["masks_output"] = F.interpolate(
                masks_out.clone().unsqueeze(1),
                (self.cfg.output_size, self.cfg.output_size),
                mode="bilinear",
                align_corners=False,
            )
            cam_poses = cam_poses_out.clone()
        else:
            results["images_output"] = F.interpolate(
                images_in.clone(),
                (self.cfg.output_size, self.cfg.output_size),
                mode="bilinear",
                align_corners=False,
            )
            results["masks_output"] = F.interpolate(
                masks_in.clone().unsqueeze(1),
                (self.cfg.output_size, self.cfg.output_size),
                mode="bilinear",
                align_corners=False,
            )
            cam_poses = cam_poses_in.clone()

        # results = {
        #     [C, H, W]
        #     'input': ...,             (processed input images [V_in,9,256,256])
        #     'cam_poses_input': ...,   ([V,4,4])
        #     'depths_input': ...,      (.......)
        #     'masks_input': ...,       (.......)
        #     'images_output': ...,     ([V_out,3,512,512])
        #     'masks_output': ...,      (.......)
        #     'cam_view_output': ...,          (colmap coordinate)
        #     'cam_view_proj_output': ...,     (colmap coordinate)
        #     'cam_pos_output': ...,           (colmap coordinate)
        #     'object_id': f"{archive_name}/{item_name}"
        # }

        # OpenGL -> COLMAP for gaussian renderer
        cam_poses[:, :3, 1:3] *= -1

        cam_view = torch.inverse(cam_poses).transpose(1, 2)   # [V_out, 4, 4]
        cam_view_proj = cam_view @ self.projection_matrix      # [V_out, 4, 4]
        cam_pos = -cam_poses[:, :3, 3]                         # [V_out, 3]

        results["cam_view_output"] = cam_view
        results["cam_view_proj_output"] = cam_view_proj
        results["cam_pos_output"] = cam_pos
        results["object_id"] = f"{archive_name}/{item_name}"

        return results

