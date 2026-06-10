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
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')  # [B*5, 14, 2*H, 2*W]
        x = self.upsample(x)    # [B*5, 14, 2*H, 2*W]
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
        lambda_rank=0.1,
        K=8,  # number of sampled pairs
    ):
        """
        depth_* : [B, V, 1, H, W]
        alpha_* : [B, V, 1, H, W]
        """
        B, V, _, H, W = depth_3dgs.shape
        BV = B * V

        # [BV, HW]
        d3_all = depth_3dgs.reshape(BV, -1)
        dm_all = depth_mesh.reshape(BV, -1)

        # Build valid mask once (vectorized)
        if alpha_3dgs is not None:
            mask = alpha_3dgs.reshape(BV, -1) > 0.1
        else:
            mask = torch.ones_like(d3_all, dtype=torch.bool)

        if alpha_mesh is not None:
            mask = mask & (alpha_mesh.reshape(BV, -1) > 0.01)

        counts = mask.sum(dim=1)  # [BV]
        valid_views = torch.where(counts >= min_valid)[0]
        if valid_views.numel() == 0:
            return depth_3dgs.new_zeros(())

        counts_f = counts.clamp_min(1).to(d3_all.dtype)

        # ------------------------
        # 1) Absolute depth term (vectorized over all views)
        # ------------------------
        diff = d3_all - dm_all

        if loss_type == "l1":
            elem = diff.abs()
            loss_depth_view = (elem * mask).sum(dim=1) / counts_f

        elif loss_type == "l2":
            elem = diff.square()
            loss_depth_view = (elem * mask).sum(dim=1) / counts_f

        elif loss_type == "huber":
            elem = F.smooth_l1_loss(d3_all, dm_all, reduction="none")
            loss_depth_view = (elem * mask).sum(dim=1) / counts_f

        elif loss_type == "berhu":
            abs_diff = diff.abs()
            neg_inf = torch.finfo(abs_diff.dtype).min
            masked_abs = abs_diff.masked_fill(~mask, neg_inf)
            c = 0.2 * masked_abs.max(dim=1).values.detach()
            c = c.clamp_min(1e-12)  # stability
            c_col = c.unsqueeze(1)
            elem = torch.where(
                abs_diff <= c_col,
                abs_diff,
                (diff.square() + c_col.square()) / (2.0 * c_col),
            )
            loss_depth_view = (elem * mask).sum(dim=1) / counts_f

        elif loss_type == "scale_invariant":
            log_d3 = torch.log(d3_all + 1e-8)
            log_dm = torch.log(dm_all + 1e-8)
            log_diff = log_d3 - log_dm
            mean_diff = (log_diff * mask).sum(dim=1) / counts_f
            mean_sq = (log_diff.square() * mask).sum(dim=1) / counts_f
            loss_depth_view = mean_sq - 0.5 * mean_diff.square()

        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # Start from vectorized depth loss (same per-view averaging logic)
        total = loss_depth_view[valid_views].sum()
        num_valid_views = valid_views.numel()

        # ------------------------
        # 2) Ranking term (loop only over valid views)
        # ------------------------
        if lambda_rank > 0 and K > 0:
            for idx in valid_views.tolist():
                n = int(counts[idx].item())
                if n <= 1:
                    continue

                m = mask[idx]
                d3 = d3_all[idx][m]  # [N]
                dm = dm_all[idx][m]  # [N]

                idx_j = torch.randint(0, n, (n, K), device=d3.device)

                dm_i = dm[:, None]       # [N,1]
                dm_j = dm[idx_j]         # [N,K]
                dm_diff = dm_i - dm_j    # [N,K]

                valid_pair = dm_diff.abs() > 1e-3
                if valid_pair.any():
                    rank = F.relu(dm_diff.sign() * (d3[idx_j] - d3[:, None]))  # [N,K]
                    total = total + lambda_rank * rank[valid_pair].mean()

        return total / num_valid_views

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
                mask = pred_alpha[b, v] > 0.1
                mask = mask & (gt_depth[b, v] > 0.01)
                if gt_alpha is not None:
                    mask = mask & (gt_alpha[b, v] > 0.01)
                
                if mask.sum() < min_valid:
                    continue
                
                pred = pred_depth[b, v][mask]
                gt = gt_depth[b, v][mask]
                
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

    def gaussian_prune(
        self,
        gaussians,
        alpha_threshold=0.01,
        distance_threshold=0.02,
        scale_threshold=0.01,
        rot_threshold=0.1,
        rgb_threshold=0.1,
    ):
        """
        Grid/voxel clustering in attribute space.
        Input:  gaussians [B, N, 14]
        Output: List[Tensor[M_b, 14]]
        """
        B, N, C = gaussians.shape
        assert C == 14, f"Expected 14 channels, got {C}"
        device = gaussians.device

        flat = gaussians.reshape(B * N, 14)  # keeps autograd graph
        batch_ids = torch.arange(B, device=device, dtype=torch.long).repeat_interleave(N) # [0, 0, 0, ..., 1, 1, 1, ..., B-1, B-1, ...]

        use_distance = distance_threshold > 0
        use_scale = scale_threshold > 0
        use_rot = rot_threshold > 0
        use_rgb = rgb_threshold > 0

        def _quantize(x: torch.Tensor, step: float) -> torch.Tensor:
            step = float(max(step, 1e-8))
            return torch.round(x / step).to(torch.int32)

        with torch.no_grad():
            valid = flat[:, 3] > alpha_threshold  # opacity prefilter
            if not valid.any():
                keep_mask = torch.zeros((B, N), dtype=torch.bool, device=device)
            elif not (use_distance or use_scale or use_rot or use_rgb):
                keep_mask = valid.view(B, N)  # opacity-only
            else:
                g = flat[valid].detach()                   # [M, 14]
                b = batch_ids[valid]                       # [M]
                orig_idx = torch.where(valid)[0]           # index in [B*N]

                pos = g[:, 0:3] # [M, 3]
                opa = g[:, 3]   # [M]
                scale = g[:, 4:7]   # [M, 3]
                rot = g[:, 7:11]    # [M, 4]
                rgb = g[:, 11:14]   # [M, 3]

                # Quaternion canonicalization: q and -q represent same rotation
                max_abs_idx = rot.abs().argmax(dim=1, keepdim=True)
                sign = torch.gather(rot, 1, max_abs_idx).sign()
                sign = torch.where(sign == 0, torch.ones_like(sign), sign)
                rot = rot * sign

                # Build integer grid key in attribute space
                # row1: [0 | 2,1,50]
                # row2: [0 | 2,0,50]
                # row3: [1 | -1,1,50]
                key_parts = [b[:, None].to(torch.int32)]
                if use_distance:
                    key_parts.append(_quantize(pos, distance_threshold))
                if use_scale:
                    key_parts.append(_quantize(scale, scale_threshold))
                if use_rot:
                    key_parts.append(_quantize(rot, rot_threshold))
                if use_rgb:
                    key_parts.append(_quantize(rgb, rgb_threshold))

                keys = torch.cat(key_parts, dim=1).to(torch.int64)  # [M, D]

                # Group clusters
                _, inv = torch.unique(keys, dim=0, sorted=False, return_inverse=True)  # inv: [M]
                num_clusters = int(inv.max().item()) + 1

                # Representative = max opacity per cluster
                max_opa = torch.full((num_clusters,), -1e10, device=device, dtype=opa.dtype)    # [K = num_clusters]
                max_opa.scatter_reduce_(0, inv, opa, reduce="amax", include_self=True)  # [K]

                # Candidates with max opacity (handle ties)
                is_max = opa >= (max_opa[inv] - 1e-12)
                cand = torch.where(is_max)[0]   # [K'], K' >= K

                # Tie-break deterministically: smallest local index
                rep_local = torch.full((num_clusters,), g.shape[0], dtype=torch.long, device=device)
                rep_local.scatter_reduce_(0, inv[cand], cand, reduce="amin", include_self=True)
                rep_local = rep_local[rep_local < g.shape[0]]  # [K]

                keep_global = orig_idx[rep_local]  # indices in flattened [B*N]

                keep_flat = torch.zeros(B * N, dtype=torch.bool, device=device)
                keep_flat[keep_global] = True
                keep_mask = keep_flat.view(B, N)

        # Important: gather from original tensor -> kept gaussians keep gradient
        pruned_gaussians = [gaussians[b][keep_mask[b]] for b in range(B)]
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
        B, V, _, _ = data['cam_poses_input'].shape

        if self.cfg.pixel_align:
            rays_d = []
            rays_o = []
            cam_poses_input = data['cam_poses_input'].reshape(-1, 4, 4)  # [B, V, 4, 4] -> [B*V, 4, 4]
            for i in range(cam_poses_input.shape[0]):
                # get rays in world space
                ro, rd = get_rays(cam_poses_input[i], self.cfg.splat_size, self.cfg.splat_size, self.cfg.fovy) # [h, w, 3]
                rays_d.append(rd)
                rays_o.append(ro)
            rays_d = torch.stack(rays_d, dim=0)  # [B*V, h, w, 3]
            rays_o = torch.stack(rays_o, dim=0)  # [B*V, h, w, 3]
            rays_d = rays_d.view(B, V, self.cfg.splat_size, self.cfg.splat_size, 3) # [B, V, h, w, 3]
            rays_o = rays_o.view(B, V, self.cfg.splat_size, self.cfg.splat_size, 3) # [B, V, h, w, 3]

            pos = gaussians[..., 0:3]   # [B, V*h*w, 3]
            dist = pos.mean(dim=-1, keepdim=True).sigmoid() * self.cfg.max_distance   # [B, V*h*w, 1]
            pos = dist * rays_d.view(B, -1, 3) + rays_o.view(B, -1, 3)  # [B, V*h*w, 3]

            gaussians = torch.cat([pos, gaussians[..., 3:]], dim=-1)  # [B, V*h*w, 14]

            # get pixel-aligned depth
            cam_poses_colmap = data['cam_poses_input'].clone()  # OpenGL cam-to-world
            cam_poses_colmap[:, :, :3, 1:3] *= -1  # Convert to COLMAP cam-to-world
            pos = pos.view(B, V, self.cfg.splat_size * self.cfg.splat_size, 3)  # [B, V, h*w, 3]
            input_w2c = torch.inverse(cam_poses_colmap)  # [B, V, 4, 4]: world-to-COLMAP cam
            # [B, V, h*w, 3]: convert pos of 3DGS from world coordinate to COLMAP cam coordinate
            pos_cam = pos @ input_w2c[:, :, :3, :3].transpose(-1, -2).contiguous() + input_w2c[:, :, :3, 3:4].transpose(-1, -2).contiguous()
            # get z-axis as depth value
            depth = pos_cam[..., 2]  # [B, V, h*w]
            # disp_pred = 1.0 / depth.clamp(min=1e-3)  # [B, V, h*w]
            # disp_median = torch.median(disp_pred, dim=-1, keepdim=True)[0]  # [B, V, 1]
            # disp_var = (disp_pred - disp_median).abs().mean(dim=-1, keepdim=True)  # [B, V, 1]
            # disp_pred = (disp_pred - disp_median) / (disp_var + 1e-6)  # [B, V, h*w]
            depth = depth.view(B, V, 1, self.cfg.splat_size, self.cfg.splat_size)  # [B, V, 1, h, w]

        device = gaussians.device
        gaussians = self.gaussian_prune(
            gaussians,
            alpha_threshold=self.cfg.alpha_threshold,
            distance_threshold=self.cfg.distance_threshold,
            scale_threshold=self.cfg.scale_threshold,
            rot_threshold=self.cfg.rot_threshold,
            rgb_threshold=self.cfg.rgb_threshold
        )  # list of [M_b, 14], M_b is the number of Gaussians after pruning for batch b
        results['gaussians'] = gaussians
        results['average_kept_gaussians'] = torch.tensor(sum([g.shape[0] for g in gaussians]) / (len(gaussians) * self.cfg.splat_size * self.cfg.splat_size * V), device=device)

        # always use white background
        bg_color = torch.ones(3, dtype=torch.float32, device=device)

        # use the other views for rendering and supervision
        rendered_results = self.gs.render(gaussians, data['cam_view_output'], data['cam_view_proj_output'], data['cam_pos_output'], bg_color=bg_color)
        pred_images = rendered_results['image']  # [B, V, C, output_size, output_size]
        pred_alphas = rendered_results['alpha']  # [B, V, 1, output_size, output_size]
        pred_images = pred_images * pred_alphas + (1 - pred_alphas) * bg_color.view(1, 1, 3, 1, 1)
        if self.cfg.pixel_align:
            pred_depths = depth.view(B, V, 1, self.cfg.splat_size, self.cfg.splat_size)
        else:
            pred_depths = rendered_results['depth']  # [B, V, 1, output_size, output_size]

        results['images_pred'] = pred_images
        results['alphas_pred'] = pred_alphas
        results['depths_pred'] = pred_depths
        results['depths_pred_rasterized'] = rendered_results['depth']
        
        gt_images = data['images_output']   # [B, V, 3, output_size, output_size], ground-truth novel views
        gt_masks = data['masks_output']     # [B, V, 1, output_size, output_size], ground-truth masks
        gt_depths = data['depths_input']   # [B, V, 1, splat_size, splat_size], ground-truth depths
        gt_masks_in = data['masks_input']   # [B, V, 1, splat_size, splat_size], ground-truth masks for input views

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
            # disp_gt = 1.0 / gt_depths.clamp(min=1e-3)  # [B, V, 1, H, W]
            # disp_gt = disp_gt.view(B, V, -1)  # [B, V, H*W]
            # disp_median_gt = torch.median(disp_gt, dim=-1, keepdim=True)[0]  # [B, V, 1]
            # disp_var_gt = (disp_gt - disp_median_gt).abs().mean(dim=-1, keepdim=True)  # [B, V, 1]
            # disp_gt = (disp_gt - disp_median_gt) / (disp_var_gt + 1e-6)  # [B, V, H*W]
            # disp_gt = disp_gt.view(B, V, 1, self.cfg.splat_size, self.cfg.splat_size)  # [B, V, 1, h, w]
            
            loss_depth_all = self.depth_loss(
                depth,
                gt_depths,
                None,
                gt_masks_in,
                loss_type=depth_loss_type,
                lambda_rank=self.cfg.lambda_depth_rank/self.cfg.lambda_depth,
                K=self.cfg.depth_rank_K,
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

            # Depth metrics
            # depth_metrics = self.depth_metrics(pred_depths, gt_depths, pred_alphas, gt_masks)
            results['abs_diff'] = torch.tensor(0.0, device=images.device)
            results['abs_rel'] = torch.tensor(0.0, device=images.device)
            results['sq_rel'] = torch.tensor(0.0, device=images.device)
            results['delta_1'] = torch.tensor(0.0, device=images.device)

        return results