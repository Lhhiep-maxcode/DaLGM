# core/model.py

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from core.model_config import Options

import kiui

class GaussianRenderer:
    def __init__(self, cfg: Options):
        
        self.cfg = cfg
        self.bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        
        # intrinsics
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.cfg.fovy))
        self.projection_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.projection_matrix[0, 0] = 1 / self.tan_half_fov
        self.projection_matrix[1, 1] = 1 / self.tan_half_fov
        self.projection_matrix[2, 2] = (cfg.zfar + cfg.znear) / (cfg.zfar - cfg.znear)
        self.projection_matrix[3, 2] = - (cfg.zfar * cfg.znear) / (cfg.zfar - cfg.znear)
        self.projection_matrix[2, 3] = 1
    
    def compute_surface_depth(self, means3D, opacity, scales, rotations, view_matrix, campos, 
                             opacity_threshold=0.1, image_height=512, image_width=512):
        """
        Compute surface depth by finding the closest Gaussian with sufficient opacity.        
        opacity threshold is fixed to 0.1
        Returns surface_depth: [1, H, W]
        """
        device = means3D.device
        N = means3D.shape[0]
        
        # Transform Gaussians to camera space
        means3D_hom = torch.cat([means3D, torch.ones(N, 1, device=device)], dim=-1)  # [N, 4]
        means3D_cam = (view_matrix @ means3D_hom.T).T  # [N, 4]
        depths_3d = means3D_cam[:, 2]  # [N] - depth in camera space (z)
        
        # Filter out Gaussians behind camera or with low opacity
        valid_mask = (depths_3d > 0) & (opacity.squeeze(-1) >= opacity_threshold)  # [N]
        
        if valid_mask.sum() == 0:
            return torch.full((1, image_height, image_width), self.cfg.zfar, 
                            dtype=torch.float32, device=device)
        
        # Project Gaussians to image space
        # x_2D = (x_cam / z_cam) × fx + cx
        # y_2D = (y_cam / z_cam) × fy + cy    
        fx = image_width / (2 * self.tan_half_fov)
        fy = image_height / (2 * self.tan_half_fov)
        cx = image_width / 2
        cy = image_height / 2
        
        x_2d = (means3D_cam[:, 0] / means3D_cam[:, 2]) * fx + cx  # [N]
        y_2d = (means3D_cam[:, 1] / means3D_cam[:, 2]) * fy + cy  # [N]
        
        # Initialize surface depth map with far plane
        surface_depth = torch.full((image_height, image_width), self.cfg.zfar, 
                                  dtype=torch.float32, device=device)
        
        # For each valid Gaussian, update pixels within its influence
        valid_means3D = means3D[valid_mask]
        valid_opacity = opacity[valid_mask]
        valid_depths = depths_3d[valid_mask]
        valid_x2d = x_2d[valid_mask]
        valid_y2d = y_2d[valid_mask]
        valid_scales = scales[valid_mask]
        
        # Estimate influence radius for each Gaussian
        # each gaussian not only affects its center pixel, but also neighboring pixels
        influence_radius = valid_scales.max(dim=-1)[0] * 3.0  # 3 sigma coverage ~ 99.7%, maybe can change?
        influence_radius_pixels = influence_radius * fx / valid_depths  # Convert to pixel space
        
        # For each valid Gaussian, update the depth map
        for i in range(len(valid_means3D)):
            x_center = valid_x2d[i].item()
            y_center = valid_y2d[i].item()
            radius = max(influence_radius_pixels[i].item(), 1.0)
            
            # Bounding box in pixel space
            x_min = max(0, int(x_center - radius))
            x_max = min(image_width, int(x_center + radius) + 1)
            y_min = max(0, int(y_center - radius))
            y_max = min(image_height, int(y_center + radius) + 1)
            
            if x_min >= x_max or y_min >= y_max:
                continue
            
            # Update depth map: keep minimum depth
            current_depth = valid_depths[i].item()
            surface_depth[y_min:y_max, x_min:x_max] = torch.min(
                surface_depth[y_min:y_max, x_min:x_max],
                torch.full((y_max - y_min, x_max - x_min), current_depth, 
                          dtype=torch.float32, device=device)
            )
        
        return surface_depth.unsqueeze(0)  # [1, H, W]

    def render(self, gaussians, cam_view, cam_view_proj, cam_pos, bg_color=None, scale_modifier=1):
        # gaussians: [B, N, 14]
        # cam_view: [B, V, 4, 4]  (Transforming (world space -> camera space))
        # cam_view_proj: [B, V, 4, 4]  (Transforming (world space -> clip space) for perspective projection: projection_matrix @ cam_view) 
        # cam_pos: [B, V, 3]

        device = gaussians.device
        B, V = cam_view.shape[:2]

        # loop of loop...
        images = []
        alphas = []
        depths = []
        surface_depths = [] if self.cfg.compute_surface else None
        
        for b in range(B):

            # pos, opacity, scale, rotation, shs
            means3D = gaussians[b, :, 0:3].contiguous().float()
            opacity = gaussians[b, :, 3:4].contiguous().float()
            scales = gaussians[b, :, 4:7].contiguous().float()
            rotations = gaussians[b, :, 7:11].contiguous().float()
            rgbs = gaussians[b, :, 11:].contiguous().float() # [N, 3]

            for v in range(V):
                
                # render novel views
                view_matrix = cam_view[b, v].float()
                view_proj_matrix = cam_view_proj[b, v].float()
                campos = cam_pos[b, v].float()

                raster_settings = GaussianRasterizationSettings(
                    image_height=self.cfg.output_size,
                    image_width=self.cfg.output_size,
                    tanfovx=self.tan_half_fov,
                    tanfovy=self.tan_half_fov,
                    bg=self.bg_color if bg_color is None else bg_color,
                    scale_modifier=scale_modifier,
                    viewmatrix=view_matrix,
                    projmatrix=view_proj_matrix,
                    sh_degree=0,
                    campos=campos,
                    prefiltered=False,
                    debug=False,
                )

                rasterizer = GaussianRasterizer(raster_settings=raster_settings)

                # Rasterize visible Gaussians to image, obtain their radii (on screen).
                rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
                    means3D=means3D,
                    means2D=torch.zeros_like(means3D, dtype=torch.float32, device=device),
                    shs=None,
                    colors_precomp=rgbs,
                    opacities=opacity,
                    scales=scales,
                    rotations=rotations,
                    cov3D_precomp=None,
                )

                rendered_image = rendered_image.clamp(0, 1)

                images.append(rendered_image)
                alphas.append(rendered_alpha)
                depths.append(rendered_depth)
                
                # Compute surface depth if requested (for LGM without pixel_align)
                if self.cfg.compute_surface:
                    surf_depth = self.compute_surface_depth(
                        means3D, opacity, scales, rotations,
                        view_matrix, campos,
                        opacity_threshold=0.1,
                        image_height=self.cfg.output_size,
                        image_width=self.cfg.output_size
                    )
                    surface_depths.append(surf_depth)

        images = torch.stack(images, dim=0).view(B, V, 3, self.cfg.output_size, self.cfg.output_size)
        alphas = torch.stack(alphas, dim=0).view(B, V, 1, self.cfg.output_size, self.cfg.output_size)
        depths = torch.stack(depths, dim=0).view(B, V, 1, self.cfg.output_size, self.cfg.output_size)
        
        result = {
            "image": images, # [B, V, 3, H, W]
            "alpha": alphas, # [B, V, 1, H, W]
            "depth": depths, # [B, V, 1, H, W] - volume
        }
        
        if self.cfg.compute_surface:
            surface_depths = torch.stack(surface_depths, dim=0).view(B, V, 1, self.cfg.output_size, self.cfg.output_size)
            result["surface_depth"] = surface_depths  # [B, V, 1, H, W] - surface
        
        return result

    def save_ply(self, gaussians, path, compatible=True):
        # Target Gaussians example:
        # ------------------------------
        # property float x
        # property float y
        # property float z
        # property float f_dc_0
        # property float f_dc_1
        # property float f_dc_2
        # property float opacity
        # property float scale_0
        # property float scale_1
        # property float scale_2
        # property float rot_0
        # property float rot_1
        # property float rot_2
        # property float rot_3
        # ------------------------------

        assert gaussians.shape[0] == 1, 'only support batch size 1'

        from plyfile import PlyData, PlyElement

        # gaussians: [1, N, 14]
        means3D = gaussians[0, :, 0:3].contiguous().float()     # (N, 3)
        opacity = gaussians[0, :, 3:4].contiguous().float()     # (N, 1)
        scales = gaussians[0, :, 4:7].contiguous().float()      # (N, 3)
        rotations = gaussians[0, :, 7:11].contiguous().float()  # (N, 4)
        shs = gaussians[0, :, 11:].unsqueeze(1).contiguous().float() # [N, 1, 3]

        # filter out Gaussian with low opacity value
        mask = opacity.squeeze(-1) >= 0.005
        means3D = means3D[mask]
        opacity = opacity[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        shs = shs[mask]

        # invert activation to make it compatible with the original ply format
        if compatible:
            opacity = kiui.op.inverse_sigmoid(opacity)
            scales = torch.log(scales + 1e-8)
            shs = (shs - 0.5) / 0.28209479177387814

        xyzs = means3D.detach().cpu().numpy()
        f_dc = shs.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = opacity.detach().cpu().numpy()
        scales = scales.detach().cpu().numpy()
        rotations = rotations.detach().cpu().numpy()

        l = ['x', 'y', 'z']
        # All channels except the 3 DC
        for i in range(f_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        l.append('opacity')
        for i in range(scales.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotations.shape[1]):
            l.append('rot_{}'.format(i))

        dtype_full = [(attribute, 'f4') for attribute in l]

        elements = np.empty(xyzs.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyzs, f_dc, opacities, scales, rotations), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        PlyData([el]).write(path)
    
    def load_ply(self, path, compatible=True):

        from plyfile import PlyData, PlyElement

        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        print("Number of points at loading : ", xyz.shape[0])

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        shs = np.zeros((xyz.shape[0], 3))
        shs[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        shs[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        shs[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
          
        gaussians = np.concatenate([xyz, opacities, scales, rots, shs], axis=1)
        gaussians = torch.from_numpy(gaussians).float() # cpu

        if compatible:
            gaussians[..., 3:4] = torch.sigmoid(gaussians[..., 3:4])
            gaussians[..., 4:7] = torch.exp(gaussians[..., 4:7])
            gaussians[..., 11:] = 0.28209479177387814 * gaussians[..., 11:] + 0.5

        return gaussians