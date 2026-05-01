# main.py

import kiui.vis
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from core.model_config import Options
from core.unet import UNet
from core.gs import GaussianRenderer
from kiui.lpips import LPIPS
from core.utils import get_rays
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure



class LGM(nn.Module):
    def __init__(self, cfg: Options):
        super().__init__()

        self.cfg = cfg

        # UNet
        self.unet = UNet(
            9, 14, 
            down_channels=self.cfg.down_channels,
            down_attention=self.cfg.down_attention,
            mid_attention=self.cfg.mid_attention,
            up_channels=self.cfg.up_channels,
            up_attention=self.cfg.up_attention,
        )

        # x2 upsample
        self.upsample = nn.Conv2d(14, 14, kernel_size=3, stride=1, padding=1)
        # last conv
        self.conv = nn.Conv2d(14, 14, kernel_size=1)

        # Gaussian Renderer
        self.gs = GaussianRenderer(cfg)

        # activations...
        self.pos_act = lambda x: x.clamp(-1, 1)     # Dense Gaussians
        self.scale_act = lambda x: 0.1 * F.softplus(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        # self.opacity_act = lambda x: torch.ones_like(x)
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = lambda x: torch.sigmoid(x) # NOTE: may use sigmoid if train again

        self.lpips_loss = LPIPS(net='vgg')
        self.lpips_loss.requires_grad_(False)

        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # ignore lpips_loss mismatch
        missing, unexpected = super().load_state_dict(state_dict, strict=strict, assign=assign)
        if missing:
            print(f"[Warning] Ignored missing keys: {missing}")
        if unexpected:
            print(f"[Warning] Ignored unexpected keys: {unexpected}")
        return missing, unexpected

    def state_dict(self, **kwargs):
        # remove lpips_loss
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if 'lpips_loss' in k:
                del state_dict[k]
        return state_dict
    
    def prepare_default_rays(self, device, elevation=0):
        # prepare Plucker embedding for 4 input images

        from kiui.cam import orbit_camera
        from core.utils import get_rays

        cam_poses = np.stack([
            orbit_camera(elevation, 0, radius=self.cfg.cam_radius),
            orbit_camera(elevation, 90, radius=self.cfg.cam_radius),
            orbit_camera(elevation, 180, radius=self.cfg.cam_radius),
            orbit_camera(elevation, 270, radius=self.cfg.cam_radius),
        ], axis=0) # [4, 4, 4]
        cam_poses = torch.from_numpy(cam_poses)

        rays_embeddings = []
        for i in range(cam_poses.shape[0]):
            rays_o, rays_d = get_rays(cam_poses[i], self.cfg.splat_size, self.cfg.splat_size, self.cfg.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)

            ## visualize rays for plotting figure
            # kiui.vis.plot_image(rays_d * 0.5 + 0.5, save=True)

        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous().to(device) # [V, 6, h, w]
        
        return rays_embeddings
    
    def forward_gaussians(self, images):
        # images: [B, 9, 9, H, W]
        # return: Gaussians: [B, num_gauss * 14]

        B, V, C, H, W = images.shape
        images = images.view(B*V, C, H, W)

        x = self.unet(images)   # [B*5, 14, H, W]
        # x = F.interpolate(x, scale_factor=2.0, mode='nearest')  # [B*5, 14, 2*H, 2*W]
        # x = self.upsample(x)    # [B*5, 14, 2*H, 2*W]
        x = self.conv(x)        # [B*5, 14, 2*H, 2*W]

        x = x.reshape(B, self.cfg.num_views_input, 14, self.cfg.splat_size, self.cfg.splat_size)

        x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, 14)    # [B, 5, splat_size, splat_size, 14] --> [B, N, 14]
        
        if self.cfg.pixel_align:
            pos = x[..., 0:3]     # [B, N, 3]
        else:
            pos = self.pos_act(x[..., 0:3])     # [B, N, 3]

        opacity = self.opacity_act(x[..., 3:4]) # [B, N, 1]
        scale = self.scale_act(x[..., 4:7]) # [B, N, 3]
        rotation = self.rot_act(x[..., 7:11])   # [B, N, 3]
        rgbs = self.rgb_act(x[..., 11:])    # [B, N, 4]

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1)    # [B, N, 14]
        return gaussians

    def depth_loss(
        self,
        depth_3dgs,
        depth_mesh,
        alpha_3dgs=None,
        alpha_mesh=None,
        loss_type="l1",
        min_valid=10,
    ):
        """
        depth_* : [B, V, 1, H, W]
        alpha_* : [B, V, 1, H, W]
        """
        # Nao thử disparity loss xem sao (1/depth)

        B, V, _, H, W = depth_3dgs.shape
        losses = []

        for b in range(B):
            for v in range(V):
                # valid mask per view
                mask = alpha_3dgs[b, v] > 0.1 if alpha_3dgs is not None else torch.ones_like(depth_3dgs[b, v], dtype=torch.bool)
                if alpha_mesh is not None:
                    mask = mask & (alpha_mesh[b, v] > 0.01)

                if mask.sum() < min_valid:
                    continue

                d3 = depth_3dgs[b, v][mask]
                dm = depth_mesh[b, v][mask]

                if loss_type in ["l1", "l2", "huber", "berhu"]:
                    # per-view min–max scaling
                    # d3_min, d3_max = d3.min(), d3.max()
                    # dm_min, dm_max = dm.min(), dm.max()

                    # d3s = (d3 - d3_min) / (d3_max - d3_min + 1e-8)
                    # dms = (dm - dm_min) / (dm_max - dm_min + 1e-8)
                    
                    d3s = d3
                    dms = dm
                    diff = d3s - dms

                    if loss_type == "l1":
                        loss = diff.abs().mean()
                    elif loss_type == "l2":
                        loss = (diff ** 2).mean()
                    elif loss_type == "huber":
                        loss = F.smooth_l1_loss(d3s, dms)
                    elif loss_type == "berhu":
                        c = 0.2 * diff.abs().max().detach()
                        loss = torch.where(
                            diff.abs() <= c,
                            diff.abs(),
                            (diff ** 2 + c ** 2) / (2 * c),
                        ).mean()

                elif loss_type == "scale_invariant":
                    # Var(X) + 0.5 * (E(X))^2 ==> relative and absolute
                    log_d3 = torch.log(d3 + 1e-8)
                    log_dm = torch.log(dm + 1e-8)
                    diff = log_d3 - log_dm
                    loss = diff.pow(2).mean() - 0.5 * diff.mean().pow(2)

                else:
                    raise ValueError(f"Unknown loss type: {loss_type}")

                losses.append(loss)

        if len(losses) == 0:
            return torch.tensor(0.0, device=depth_3dgs.device)

        return torch.stack(losses).mean()

    def depth_gradient_loss(self, depth_3dgs, depth_mesh, alpha_3dgs=None, alpha_mesh=None, min_valid=10):
        """
        depth_* : [B, V, 1, H, W]
        alpha_* : [B, V, 1, H, W]
        """
        B, V, _, H, W = depth_3dgs.shape
        losses = []
        
        # Precompute Sobel filters once
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                            dtype=torch.float32, device=depth_3dgs.device).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-2, -1)
        
        for b in range(B):
            for v in range(V):
                # Valid mask (crop by 1 on each side to match conv output size)
                mask_3dgs = alpha_3dgs[b, v, 0, 1:-1, 1:-1] > 0.1 if alpha_3dgs is not None else torch.ones_like(depth_3dgs[b, v, 0, 1:-1, 1:-1], dtype=torch.bool)
                mask_mesh = depth_mesh[b, v, 0, 1:-1, 1:-1] > 0
                if alpha_mesh is not None:
                    mask_mesh = mask_mesh & (alpha_mesh[b, v, 0, 1:-1, 1:-1] > 0.01)
                valid_mask = mask_3dgs & mask_mesh
                
                if valid_mask.sum() < min_valid:
                    continue
                
                # Add batch and channel dimensions
                d_3dgs = depth_3dgs[b, v].unsqueeze(0)  # [1, 1, H, W]
                d_mesh = depth_mesh[b, v].unsqueeze(0)  # [1, 1, H, W]

                # Compute gradients (output is [1, 1, H-2, W-2])
                grad_x_3dgs = F.conv2d(d_3dgs, sobel_x)
                grad_y_3dgs = F.conv2d(d_3dgs, sobel_y)
                
                grad_x_mesh = F.conv2d(d_mesh, sobel_x)
                grad_y_mesh = F.conv2d(d_mesh, sobel_y)
                
                # Compute gradient magnitude
                grad_mag_3dgs = torch.sqrt(grad_x_3dgs ** 2 + grad_y_3dgs ** 2 + 1e-8)
                grad_mag_mesh = torch.sqrt(grad_x_mesh ** 2 + grad_y_mesh ** 2 + 1e-8)
                
                grad_mag_3dgs = grad_mag_3dgs.squeeze()[valid_mask]
                grad_mag_mesh = grad_mag_mesh.squeeze()[valid_mask]
                
                loss = F.l1_loss(grad_mag_3dgs, grad_mag_mesh)
                losses.append(loss)

        if len(losses) == 0:
            return torch.tensor(0.0, device=depth_3dgs.device)
            
        return torch.stack(losses).mean()

    def depth_metrics(self, pred_depth, gt_depth, pred_alpha, gt_alpha=None, min_valid=10):
        """
        abs_diff, abs_rel, sq_rel, delta<1.25
        """
        B, V = pred_depth.shape[:2]
        
        abs_diff_list = []
        abs_rel_list = []
        sq_rel_list = []
        delta_1_list = []
        
        for b in range(B):
            for v in range(V):
                pred_alpha_bv = pred_alpha[b, v].squeeze(0)  # [H, W]
                gt_depth_bv = gt_depth[b, v].squeeze(0)      # [H, W]
                pred_depth_bv = pred_depth[b, v].squeeze(0)  # [H, W]
                
                # Create mask
                mask = pred_alpha_bv > 0.1
                mask = mask & (gt_depth_bv > 0.01)
                
                if gt_alpha is not None:
                    gt_alpha_bv = gt_alpha[b, v].squeeze(0)  # [H, W]
                    mask = mask & (gt_alpha_bv > 0.01)
                
                if mask.sum() < min_valid:
                    continue
                
                pred = pred_depth_bv[mask]
                gt = gt_depth_bv[mask]
                
                abs_diff = torch.abs(pred - gt).mean()
                abs_diff_list.append(abs_diff)
                
                abs_rel = (torch.abs(pred - gt) / (gt + 1e-8)).mean()
                abs_rel_list.append(abs_rel)
                
                sq_rel = (((pred - gt) ** 2) / (gt + 1e-8)).mean()
                sq_rel_list.append(sq_rel)
                
                thresh = torch.max(pred / (gt + 1e-8), gt / (pred + 1e-8))
                delta_1 = (thresh < 1.25).float().mean()
                delta_1_list.append(delta_1)
        
        if len(abs_diff_list) == 0:
            return {
                'abs_diff': torch.tensor(0.0, device=pred_depth.device),
                'abs_rel': torch.tensor(0.0, device=pred_depth.device),
                'sq_rel': torch.tensor(0.0, device=pred_depth.device),
                'delta_1': torch.tensor(0.0, device=pred_depth.device),
            }
        
        return {
            'abs_diff': torch.stack(abs_diff_list).mean(),
            'abs_rel': torch.stack(abs_rel_list).mean(),
            'sq_rel': torch.stack(sq_rel_list).mean(),
            'delta_1': torch.stack(delta_1_list).mean(),
        }

    def gaussian_prune(self, gaussians, alpha_threshold=0.01, distance_threshold=0.02, 
                   scale_threshold=0.01, rot_threshold=0.1, rgb_threshold=0.1):
        # gaussians: [B, N, 14]
        # Format: [pos(3), opacity(1), scale(3), rotation(4), rgb(3)]

        pruned_gaussians = []
        H, W = self.cfg.splat_size, self.cfg.splat_size
        
        for b in range(gaussians.shape[0]):
            gaussians_b = gaussians[b]  # [N, 14]
            gaussians_b = gaussians_b.view(-1, H, W, 14)  # [V, h, w, 14]
            V = gaussians_b.shape[0]
            
            keep_mask = gaussians_b[..., 3] > alpha_threshold  # [V, h, w]
            
            for v in range(V):
                v_next = (v + 1) % V
                
                gaussians_v = gaussians_b[v].view(-1, 14)  # [h, w, 14] -> [h*w, 14]
                gaussians_next = gaussians_b[v_next].view(-1, 14)  # [h, w, 14] -> [h*w, 14]

                pos_v = gaussians_v[..., 0:3]
                opacity_v = gaussians_v[..., 3]
                scale_v = gaussians_v[..., 4:7]
                rot_v = gaussians_v[..., 7:11]
                rgb_v = gaussians_v[..., 11:14]

                pos_next = gaussians_next[..., 0:3]
                opacity_next = gaussians_next[..., 3]
                scale_next = gaussians_next[..., 4:7]
                rot_next = gaussians_next[..., 7:11]
                rgb_next = gaussians_next[..., 11:14]

                # Compute pairwise distances for all attributes
                spatial_dist = torch.norm(pos_v - pos_next, dim=1)  # [h*w]
                scale_dist = torch.norm(scale_v - scale_next, dim=1)  # [h*w]
                rgb_dist = torch.norm(rgb_v - rgb_next, dim=1)  # [h*w]
                
                # For rotation, use quaternion angular distance
                rot_dot = torch.abs(torch.sum(rot_v * rot_next, dim=1))
                rot_dist = 1.0 - rot_dot.clamp(max=1.0)

                # Combined similarity: duplicates if ALL attributes are similar
                is_duplicate = (
                    (spatial_dist < distance_threshold) &
                    (scale_dist < scale_threshold) &
                    (rot_dist < rot_threshold) &
                    (rgb_dist < rgb_threshold)
                )  # [h*w]

                remove_v = is_duplicate & (opacity_v < opacity_next)
                remove_next = is_duplicate & (opacity_next <= opacity_v)
                
                keep_mask[v].view(-1)[remove_v] = False
                keep_mask[v_next].view(-1)[remove_next] = False
        
            gaussians_b = gaussians_b[keep_mask]
            pruned_gaussians.append(gaussians_b)
        return pruned_gaussians

    def forward(self, data, lambda_mse=1, lambda_lpips=0.5, lambda_depth=0.01, lambda_grad=0.01, lambda_opacity=0.1, depth_loss_type='l1'):
        # data: output of the dataloader
        # data = {
        #     [C, H, W]
        #     'input': ...,             (processed input images [V_in,9,256,256])
        #     'cam_poses_input': ...,   ([V,4,4])
        #     'images_output': ...,     ([V_out,3,512,512])
        #     'masks_output': ...,      (.......)
        #     'cam_view_output': ...,          (colmap coordinate)
        #     'cam_view_proj_output': ...,     (colmap coordinate)
        #     'cam_pos_output': ...,           (colmap coordinate)
        # }
        # ------------
        # return: results = {
        #     'gaussians': ...,
        #     'images_pred': ...,
        #     'alphas_pred': ...,
        #     'loss_mse': ...,
        #     'loss_lpips': ...,
        #     'loss': ...,
        #     'psnr': ...,
        # }

        results = {}
        loss = 0

        images = data['input']  # [B, V, 9, H, W], input features (not necessarily orthogonal)

        # predicting 3DGS representation
        gaussians = self.forward_gaussians(images)  # [B, N, 14] = [B, V*h*w, 14]
        B, V_in, _, _ = data['cam_poses_input'].shape

        if self.cfg.pixel_align:
            rays_d = []
            rays_o = []
            cam_poses_input = data['cam_poses_input'].reshape(-1, 4, 4)  # [B, V_in, 4, 4] -> [B*V_in, 4, 4]
            for i in range(cam_poses_input.shape[0]):
                # get rays in world space
                ro, rd = get_rays(cam_poses_input[i], self.cfg.splat_size, self.cfg.splat_size, self.cfg.fovy) # [h, w, 3]
                rays_d.append(rd)
                rays_o.append(ro)
            rays_d = torch.stack(rays_d, dim=0)  # [B*V_in, h, w, 3]
            rays_o = torch.stack(rays_o, dim=0)  # [B*V_in, h, w, 3]
            rays_d = rays_d.view(B, V_in, self.cfg.splat_size, self.cfg.splat_size, 3) # [B, V_in, h, w, 3]
            rays_o = rays_o.view(B, V_in, self.cfg.splat_size, self.cfg.splat_size, 3) # [B, V_in, h, w, 3]

            pos = gaussians[..., 0:3]   # [B, V_in*h*w, 3]
            dist = pos.mean(dim=-1, keepdim=True).sigmoid() * self.cfg.max_distance   # [B, V_in*h*w, 1]
            pos = dist * rays_d.view(B, -1, 3) + rays_o.view(B, -1, 3)  # [B, V_in*h*w, 3]

            gaussians = torch.cat([pos, gaussians[..., 3:]], dim=-1)  # [B, V_in*h*w, 14]

            # get pixel-aligned depth
            cam_poses_colmap = data['cam_poses_input'].clone()  # OpenGL cam-to-world
            cam_poses_colmap[:, :, :3, 1:3] *= -1  # Convert to COLMAP cam-to-world
            pos = pos.view(B, V_in, self.cfg.splat_size * self.cfg.splat_size, 3)  # [B, V_in, h*w, 3]
            input_w2c = torch.inverse(cam_poses_colmap)  # [B, V_in, 4, 4]: world-to-COLMAP cam
            # [B, V_in, h*w, 3]: convert pos of 3DGS from world coordinate to COLMAP cam coordinate
            pos_cam = pos @ input_w2c[:, :, :3, :3].transpose(-1, -2).contiguous() + input_w2c[:, :, :3, 3:4].transpose(-1, -2).contiguous()
            # get z-axis as depth value
            depth = pos_cam[..., 2]  # [B, V_in, h*w]
            # disp_pred = 1.0 / depth.clamp(min=1e-3)  # [B, V_in, h*w]
            # disp_median = torch.median(disp_pred, dim=-1, keepdim=True)[0]  # [B, V_in, 1]
            # disp_var = (disp_pred - disp_median).abs().mean(dim=-1, keepdim=True)  # [B, V_in, 1]
            # disp_pred = (disp_pred - disp_median) / (disp_var + 1e-6)  # [B, V_in, h*w]
            depth = depth.view(B, V_in, 1, self.cfg.splat_size, self.cfg.splat_size)  # [B, V_in, 1, h, w]

        device = gaussians.device
        # gaussians = self.gaussian_prune(gaussians)  # list of [M_b, 14], M_b is the number of Gaussians after pruning for batch b
        results['gaussians'] = gaussians
        # results['average_kept_gaussians'] = torch.tensor(sum([g.shape[0] for g in gaussians]) / (len(gaussians) * self.cfg.splat_size * self.cfg.splat_size * V_in), device=device)

        # always use white background
        bg_color = torch.ones(3, dtype=torch.float32, device=device)

        # use the other views for rendering and supervision
        rendered_results = self.gs.render(gaussians, data['cam_view_output'], data['cam_view_proj_output'], data['cam_pos_output'], bg_color=bg_color)
        
        # Render from input views for depth metrics (needed for both compute_surface and pixel_align)
        if (self.cfg.compute_surface and not self.cfg.pixel_align) or self.cfg.pixel_align:
            # Convert cam_poses_input OpenGL to COLMAP format for Gaussian renderer
            cam_poses_input_colmap = data['cam_poses_input'].clone()  # [B, V_in, 4, 4]
            cam_poses_input_colmap[:, :, :3, 1:3] *= -1  
            
            # Compute camera matrices for input views (same as dataset.py)
            cam_view_input = torch.inverse(cam_poses_input_colmap).transpose(-1, -2)  # [B, V_in, 4, 4]
            
            projection_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
            projection_matrix[0, 0] = 1 / np.tan(0.5 * np.deg2rad(self.cfg.fovy))
            projection_matrix[1, 1] = 1 / np.tan(0.5 * np.deg2rad(self.cfg.fovy))
            projection_matrix[2, 2] = (self.cfg.zfar + self.cfg.znear) / (self.cfg.zfar - self.cfg.znear)
            projection_matrix[3, 2] = -(self.cfg.zfar * self.cfg.znear) / (self.cfg.zfar - self.cfg.znear)
            projection_matrix[2, 3] = 1
            
            cam_view_proj_input = cam_view_input @ projection_matrix  # [B, V_in, 4, 4]
            cam_pos_input = -cam_poses_input_colmap[:, :, :3, 3]  # [B, V_in, 3]
            
            # Render from input views to get depth/alpha
            rendered_results_input = self.gs.render(gaussians, cam_view_input, cam_view_proj_input, cam_pos_input, bg_color=bg_color)
        pred_images = rendered_results['image']  # [B, V_out, C, output_size, output_size]
        pred_alphas = rendered_results['alpha']  # [B, V_out, 1, output_size, output_size]
        pred_images = pred_images * pred_alphas + (1 - pred_alphas) * bg_color.view(1, 1, 3, 1, 1)
        if self.cfg.pixel_align:
            pred_depths = depth.view(B, V_in, 1, self.cfg.splat_size, self.cfg.splat_size)
        else:
            pred_depths = rendered_results['depth']  # [B, V_out, 1, output_size, output_size]

        results['images_pred'] = pred_images
        results['alphas_pred'] = pred_alphas
        results['depths_pred'] = pred_depths
        results['depths_pred_rasterized'] = rendered_results['depth']
        
        if self.cfg.compute_surface and 'surface_depth' in rendered_results:
            results['surface_depth'] = rendered_results['surface_depth']  # [B, V_out, 1, output_size, output_size]
        
        gt_images = data['images_output']   # [B, V_out, 3, output_size, output_size], ground-truth novel views
        gt_masks = data['masks_output']     # [B, V_out, 1, output_size, output_size], ground-truth masks
        gt_depths = data['depths_input']   # [B, V_in, 1, splat_size, splat_size], ground-truth depths
        gt_masks_in = data['masks_input']   # [B, V_in, 1, splat_size, splat_size], ground-truth masks for input views

        gt_images = gt_images * gt_masks + (1 - gt_masks) * bg_color.view(1, 1, 3, 1, 1)

        loss_mse_all = F.mse_loss(pred_images, gt_images) + self.cfg.lambda_alpha * F.mse_loss(pred_alphas, gt_masks)
        loss = loss + lambda_mse * (loss_mse_all) # + lambda_mse * (lambda_top - 1) * loss_mse_top

        if lambda_lpips > 0:
            loss_lpips_all = self.lpips_loss(
                # Rescale value from [0, 1] to [-1, -1] and resize to 256 to save memory cost
                F.interpolate(gt_images.view(-1, 3, self.cfg.output_size, self.cfg.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
                F.interpolate(pred_images.view(-1, 3, self.cfg.output_size, self.cfg.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
            ).mean()
            loss = loss + lambda_lpips * (loss_lpips_all) # + lambda_lpips * (lambda_top - 1) * loss_lpips_top

        if lambda_depth > 0 and self.cfg.pixel_align:
            # Flatten spatial dimensions for consistent normalization
            # disp_gt = 1.0 / gt_depths.clamp(min=1e-3)  # [B, V_in, 1, H, W]
            # disp_gt = disp_gt.view(B, V_in, -1)  # [B, V_in, H*W]
            # disp_median_gt = torch.median(disp_gt, dim=-1, keepdim=True)[0]  # [B, V_in, 1]
            # disp_var_gt = (disp_gt - disp_median_gt).abs().mean(dim=-1, keepdim=True)  # [B, V_in, 1]
            # disp_gt = (disp_gt - disp_median_gt) / (disp_var_gt + 1e-6)  # [B, V_in, H*W]
            # disp_gt = disp_gt.view(B, V_in, 1, self.cfg.splat_size, self.cfg.splat_size)  # [B, V_in, 1, h, w]
            
            loss_depth_all = self.depth_loss(
                depth,
                gt_depths,
                None,
                gt_masks_in,
                loss_type=depth_loss_type,
            )
            loss = loss + lambda_depth * (loss_depth_all)
            results['loss_depth'] = loss_depth_all
        
        if lambda_grad > 0 and self.cfg.pixel_align:
            loss_grad_all = self.depth_gradient_loss(
                depth,
                gt_depths,
                None,
                gt_masks_in,
            )
            loss = loss + lambda_grad * (loss_grad_all)
            results['loss_depth_grad'] = loss_grad_all
        
        if lambda_opacity > 0:
            # opacity regularization (use rendered alpha since gaussians is now a list)
            loss_opacity = pred_alphas.mean()
            loss = loss + lambda_opacity * loss_opacity

        results['loss'] = loss

        # metric
        with torch.no_grad():
            B, V_out, C, H, W = pred_images.shape

            # PSNR
            psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))
            results['psnr'] = psnr
            
            # SSIM
            ssim = self.ssim_metric(pred_images.view(B * V_out, C, H, W), gt_images.view(B * V_out, C, H, W))
            results['ssim'] = ssim

            # LPIPS
            lpips = self.lpips_loss(
                F.interpolate(gt_images.view(-1, 3, self.cfg.output_size, self.cfg.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
                F.interpolate(pred_images.view(-1, 3, self.cfg.output_size, self.cfg.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
            ).mean()
            results['lpips'] = lpips

            # Depth metrics
            if self.cfg.compute_surface and not self.cfg.pixel_align:
                surface_depths_pred_input = rendered_results_input.get('surface_depth', None)
                if surface_depths_pred_input is not None:
                    pred_alphas_input = rendered_results_input['alpha']
                    
                    # Resize surface depth to match gt_depths size
                    surface_depths_pred_resized = F.interpolate(
                        surface_depths_pred_input.view(B * V_in, 1, self.cfg.output_size, self.cfg.output_size),
                        size=(self.cfg.splat_size, self.cfg.splat_size),
                        mode='nearest'
                    ).view(B, V_in, 1, self.cfg.splat_size, self.cfg.splat_size)
                    
                    pred_alphas_resized = F.interpolate(
                        pred_alphas_input.view(B * V_in, 1, self.cfg.output_size, self.cfg.output_size),
                        size=(self.cfg.splat_size, self.cfg.splat_size),
                        mode='bilinear',
                        align_corners=False
                    ).view(B, V_in, 1, self.cfg.splat_size, self.cfg.splat_size)
                    
                    depth_metrics = self.depth_metrics(
                        surface_depths_pred_resized, 
                        gt_depths, 
                        pred_alphas_resized, 
                        gt_masks_in
                    )
                    results['abs_diff'] = depth_metrics['abs_diff']
                    results['abs_rel'] = depth_metrics['abs_rel']
                    results['sq_rel'] = depth_metrics['sq_rel']
                    results['delta_1'] = depth_metrics['delta_1']
                else:
                    results['abs_diff'] = torch.tensor(0.0, device=images.device)
                    results['abs_rel'] = torch.tensor(0.0, device=images.device)
                    results['sq_rel'] = torch.tensor(0.0, device=images.device)
                    results['delta_1'] = torch.tensor(0.0, device=images.device)
            elif self.cfg.pixel_align:
                pred_alphas_input = rendered_results_input['alpha']  # [B, V_in, 1, output_size, output_size]
                
                # Resize alpha from output_size to splat_size to match pred_depths
                pred_alphas_resized = F.interpolate(
                    pred_alphas_input.view(B * V_in, 1, self.cfg.output_size, self.cfg.output_size),
                    size=(self.cfg.splat_size, self.cfg.splat_size),
                    mode='bilinear',
                    align_corners=False
                ).view(B, V_in, 1, self.cfg.splat_size, self.cfg.splat_size)

                depth_metrics = self.depth_metrics(
                    pred_depths, 
                    gt_depths, 
                    pred_alphas_resized,
                    gt_masks_in
                )
                results['abs_diff'] = depth_metrics['abs_diff']
                results['abs_rel'] = depth_metrics['abs_rel']
                results['sq_rel'] = depth_metrics['sq_rel']
                results['delta_1'] = depth_metrics['delta_1']
            else:
                results['abs_diff'] = torch.tensor(0.0, device=images.device)
                results['abs_rel'] = torch.tensor(0.0, device=images.device)
                results['sq_rel'] = torch.tensor(0.0, device=images.device)
                results['delta_1'] = torch.tensor(0.0, device=images.device)

        return results