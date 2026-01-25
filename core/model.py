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
            rays_o, rays_d = get_rays(cam_poses[i], self.cfg.input_size, self.cfg.input_size, self.cfg.fovy) # [h, w, 3]
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
        x = self.conv(x)        # [B*5, 14, H, W]

        x = x.reshape(B, 9, 14, self.cfg.splat_size, self.cfg.splat_size)

        x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, 14)    # [B, 5, splat_size, splat_size, 14] --> [B, N, 14]
        
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
        alpha_3dgs,
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
                mask = alpha_3dgs[b, v] > 0.1
                mask = mask & (depth_mesh[b, v] > 0.01)
                if alpha_mesh is not None:
                    mask = mask & (alpha_mesh[b, v] > 0.01)

                if mask.sum() < min_valid:
                    continue

                d3 = depth_3dgs[b, v][mask]
                dm = depth_mesh[b, v][mask]

                if loss_type in ["l1", "l2", "huber", "berhu"]:
                    # per-view min–max scaling
                    d3_min, d3_max = d3.min(), d3.max()
                    dm_min, dm_max = dm.min(), dm.max()

                    d3s = (d3 - d3_min) / (d3_max - d3_min + 1e-8)
                    dms = (dm - dm_min) / (dm_max - dm_min + 1e-8)

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

    def depth_gradient_loss(self, depth_3dgs, depth_mesh, alpha_3dgs, alpha_mesh=None, min_valid=10):
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
                mask_3dgs = alpha_3dgs[b, v, 0, 1:-1, 1:-1] > 0.1
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

    def forward(self, data, lambda_mse=1, lambda_lpips=0.5, lambda_depth=0.01, lambda_grad=0.01, lambda_opacity=0.1, depth_loss_type='l1'):
        # data: output of the dataloader
        # data = {
        #     [C, H, W]
        #     'input': ...,             (processed input images 5x9x256x256)
        #     'cam_poses_input': ...,   
        #     'images_output': ...,     (9x3x512x512)
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

        images = data['input']  # [B, 5, 9, H, W], input features (not necessarily orthogonal)

        # predicting 3DGS representation
        gaussians = self.forward_gaussians(images)  # [B, N, 14]

        results['gaussians'] = gaussians

        # always use white background
        bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device)

        # use the other views for rendering and supervision
        rendered_results = self.gs.render(gaussians, data['cam_view_output'], data['cam_view_proj_output'], data['cam_pos_output'], bg_color=bg_color)
        pred_images = rendered_results['image']  # [B, V, C, output_size, output_size]
        pred_alphas = rendered_results['alpha']  # [B, V, 1, output_size, output_size]
        pred_images = pred_images * pred_alphas + (1 - pred_alphas) * bg_color.view(1, 1, 3, 1, 1)
        pred_depths = rendered_results['depth']  # [B, V, 1, output_size, output_size]

        results['images_pred'] = pred_images
        results['alphas_pred'] = pred_alphas
        results['depths_pred'] = pred_depths

        gt_images = data['images_output']   # [B, V, 3, output_size, output_size], ground-truth novel views
        gt_masks = data['masks_output']     # [B, V, 1, output_size, output_size], ground-truth masks
        gt_depths = data['depths_output']   # [B, V, 1, output_size, output_size], ground-truth depths

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

        if lambda_depth > 0:
            loss_depth_all = self.depth_loss(
                pred_depths,
                gt_depths,
                pred_alphas,
                gt_masks,
                loss_type=depth_loss_type,
            )
            loss = loss + lambda_depth * (loss_depth_all)
        
        if lambda_grad > 0:
            loss_grad_all = self.depth_gradient_loss(
                pred_depths,
                gt_depths,
                pred_alphas,
                gt_masks,
            )
            loss = loss + lambda_grad * (loss_grad_all)
        
        if lambda_opacity > 0:
            # opacity regularization
            loss_opacity = gaussians[..., 3:4].mean()
            loss = loss + lambda_opacity * loss_opacity

        results['loss'] = loss

        # metric
        with torch.no_grad():
            B, V, C, H, W = pred_images.shape

            # PSNR
            psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))
            results['psnr'] = psnr
            
            # SSIM
            ssim = self.ssim_metric(pred_images.view(B * V, C, H, W), gt_images.view(B * V, C, H, W))
            results['ssim'] = ssim

            # LPIPS
            lpips = self.lpips_loss(
                F.interpolate(gt_images.view(-1, 3, self.cfg.output_size, self.cfg.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
                F.interpolate(pred_images.view(-1, 3, self.cfg.output_size, self.cfg.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
            ).mean()
            results['lpips'] = lpips

        return results