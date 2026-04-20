from core.model_config import AllConfigs, Options
from core.model import LGM
from accelerate import Accelerator
from safetensors.torch import load_file
from core.dataset import ObjaverseDataset as Dataset
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import LambdaLR

import torch
import tyro
import kiui
import wandb
import numpy as np
import os
import random

def main():
    
    cfg = tyro.cli(AllConfigs)

    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps
    )

    model = LGM(cfg)

    # Load model checkpoint for FINE-TUNING
    if cfg.fine_tune and cfg.resume is not None:
        # (cfg.resume in file type)
        if cfg.resume.endswith('safetensors'):
            ckpt = load_file(cfg.resume, device='cpu')
        else:
            ckpt = torch.load(cfg.resume, map_location='cpu')
        
        # tolerant load (only load matching shapes)
        # model.load_state_dict(ckpt, strict=False)
        state_dict = model.state_dict()
        for k, v in ckpt.items():
            if k in state_dict: 
                if state_dict[k].shape == v.shape:
                    state_dict[k].copy_(v)
                else:
                    accelerator.print(f'[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
            else:
                accelerator.print(f'[WARN] unexpected param {k}: {v.shape}')

    val_dataset = Dataset(
        data_path=cfg.data_path,
        depth1_path=cfg.depth1_path,
        depth2_path=cfg.depth2_path,
        depth3_path=cfg.depth3_path,
        depth4_path=cfg.depth4_path,
        eval_path=cfg.eval_path,
        cfg=cfg,
        type="val",
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=True
    )

    # accelerate
    model, val_dataloader = accelerator.prepare(
        model, val_dataloader
    )

    if not cfg.fine_tune and cfg.resume is not None:
        # NOTE: cfg.resume (dir type) must be saved by accelerator.save_state()
        accelerator.load_state(cfg.resume, strict=False)

    if accelerator.is_main_process:
        accelerator.print(f'[INFO] start evaluation for {len(val_dataset)} objects...')
    # eval
    with torch.no_grad():
        model.eval()
        total_psnr = 0
        total_ssim = 0
        total_lpips = 0
        total_abs_diff = 0
        total_abs_rel = 0
        total_sq_rel = 0
        total_delta_1 = 0
        if accelerator.is_main_process:
            pbar2 = tqdm(val_dataloader, desc=f"[Evaluation]")

        for i, data in enumerate(val_dataloader):
            out = model(data)

            # Move metrics to CPU immediately to save GPU memory
            psnr = out['psnr'].detach().cpu()
            ssim = out['ssim'].detach().cpu()
            lpips = out['lpips'].detach().cpu()
            abs_diff = out['abs_diff'].detach().cpu()
            abs_rel = out['abs_rel'].detach().cpu()
            sq_rel = out['sq_rel'].detach().cpu()
            delta_1 = out['delta_1'].detach().cpu()
            total_psnr += psnr
            total_ssim += ssim
            total_lpips += lpips
            total_abs_diff += abs_diff
            total_abs_rel += abs_rel
            total_sq_rel += sq_rel
            total_delta_1 += delta_1

            if accelerator.is_main_process:
                pbar2.update(1)
                if i % 5 == 0:
                    gt_images = data['images_output'].detach().cpu().numpy()    # [B, V, 3, output_size, output_size]
                    gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3)
                    kiui.utils.write_image(f'{cfg.workspace}/{i}_eval_gt_images.jpg', gt_images)

                    pred_images = out['images_pred'].detach().cpu().numpy()     # [B, V, 3, output_size, output_size]
                    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                    kiui.utils.write_image(f'{cfg.workspace}/{i}_eval_pred_images.jpg', pred_images)
            
            # Clear large tensors from GPU memory
            del out
            torch.cuda.empty_cache()

        if accelerator.is_main_process:
            pbar2.close()

        # Move totals back to GPU for gathering (accelerator expects GPU tensors)
        total_psnr = total_psnr.to(accelerator.device)
        total_ssim = total_ssim.to(accelerator.device)
        total_lpips = total_lpips.to(accelerator.device)
        total_abs_diff = total_abs_diff.to(accelerator.device)
        total_abs_rel = total_abs_rel.to(accelerator.device)
        total_sq_rel = total_sq_rel.to(accelerator.device)
        total_delta_1 = total_delta_1.to(accelerator.device)

        total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
        total_ssim = accelerator.gather_for_metrics(total_ssim).mean()
        total_lpips = accelerator.gather_for_metrics(total_lpips).mean()
        total_abs_diff = accelerator.gather_for_metrics(total_abs_diff).mean()
        total_abs_rel = accelerator.gather_for_metrics(total_abs_rel).mean()
        total_sq_rel = accelerator.gather_for_metrics(total_sq_rel).mean()
        total_delta_1 = accelerator.gather_for_metrics(total_delta_1).mean()
        if accelerator.is_main_process:
            total_psnr /= len(val_dataloader)
            total_ssim /= len(val_dataloader)
            total_lpips /= len(val_dataloader)
            total_abs_diff /= len(val_dataloader)
            total_abs_rel /= len(val_dataloader)
            total_sq_rel /= len(val_dataloader)
            total_delta_1 /= len(val_dataloader)
            accelerator.print(f'[EVAL] psnr: {total_psnr:.4f}, ssim: {total_ssim:.4f}, lpips: {total_lpips:.4f}, abs_diff: {total_abs_diff:.4f}, abs_rel: {total_abs_rel:.4f}, sq_rel: {total_sq_rel:.4f}, delta_1: {total_delta_1:.4f}')


if __name__ == "__main__":
    # 1. Seed everything
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2. Force deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True, warn_only=True)

    main()