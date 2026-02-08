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

def transformer_lr_lambda(step, d_model=512, warmup_steps=4000, peak_lr=1e-4):
    """
    Paper formula:
    lr = d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})
    """
    step = max(step, 1)
    return peak_lr * (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5)) / ((warmup_steps ** -0.5) * (d_model ** -0.5))

def main():
    
    cfg = tyro.cli(AllConfigs)

    wandb.login(key=cfg.wandb_key)

    run = wandb.init(
        project=cfg.wandb_project_name,  # Specify your project
        name=cfg.wandb_experiment_name,
        id=cfg.wandb_experiment_id,
        resume=("must" if cfg.wandb_experiment_id else None),
        config={                        # Track hyperparameters and metadata
            "epochs": cfg.num_epochs, 
            "input_size": cfg.input_size,
            "splat_size": cfg.splat_size,
            "output_size": cfg.output_size,
            "num_views_input": cfg.num_views_input,
            "num_views_output": cfg.num_views_output,
            "lambda_lpips_start": cfg.lambda_lpips_start, 
            "lambda_lpips_end": cfg.lambda_lpips_end,
            "lambda_mse_start": cfg.lambda_mse_start,
            "lambda_mse_end": cfg.lambda_mse_end,
            "lambda_alpha": cfg.lambda_alpha,
            "lambda_depth": cfg.lambda_depth,
            "lambda_grad": cfg.lambda_grad,
            "lambda_opacity": cfg.lambda_opacity,
            "depth_loss_type": cfg.depth_loss_type,         
        },
    )

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

    train_dataset = Dataset(
        data_path=cfg.data_path, 
        depth1_path=cfg.depth1_path, 
        depth2_path=cfg.depth2_path, 
        depth3_path=cfg.depth3_path, 
        depth4_path=cfg.depth4_path, cfg=cfg, type='train')
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True
    )

    test_dataset = Dataset(
        data_path=cfg.data_path, 
        depth1_path=cfg.depth1_path, 
        depth2_path=cfg.depth2_path, 
        depth3_path=cfg.depth3_path, 
        depth4_path=cfg.depth4_path, cfg=cfg, type='test')
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=True
    )

    # val_dataset = Dataset(data_path=cfg.data_path, cfg=cfg, type='val')
    # val_dataloader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=cfg.batch_size,
    #     shuffle=False,
    #     num_workers=0,
    #     drop_last=False,
    #     pin_memory=True
    # )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0, weight_decay=0.05, betas=(0.9, 0.95))

    # TODO: can consider to tuning the pct_start
    # scheduler (per-step)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: transformer_lr_lambda(
            step,
            d_model=512,
            warmup_steps=cfg.warmup_steps,
            peak_lr=cfg.lr
        )
    )

    # accelerate
    model, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, scheduler
    )

    if not cfg.fine_tune and cfg.resume is not None:
        # NOTE: cfg.resume (dir type) must be saved by accelerator.save_state()
        # Continue training by loading all state of optimizer, model, scheduler
        accelerator.load_state(cfg.resume, strict=False)

    best_psnr_eval = 0

    for epoch in range(cfg.num_epochs):
        model.train()
        total_loss = 0
        total_psnr = 0
        total_ssim = 0
        total_lpips = 0
        total_abs_diff = 0
        total_abs_rel = 0
        total_sq_rel = 0
        total_delta_1 = 0
        # Create tqdm only on main process
        if accelerator.is_main_process:
            print(f"----------Epoch {epoch + 1}----------")
            pbar = tqdm(total=len(train_dataloader), desc=f"[T] E{epoch+1}/{cfg.num_epochs}")
            

        for i, data in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                # Accumulate to simulate large batch training
                step_ratio = (epoch + i / len(train_dataloader)) / cfg.num_epochs
                lambda_lpips = cfg.lambda_lpips_start * (cfg.lambda_lpips_end / cfg.lambda_lpips_start) ** step_ratio
                lambda_mse = cfg.lambda_mse_start * (cfg.lambda_mse_end / cfg.lambda_mse_start) ** step_ratio
                lambda_depth = cfg.lambda_depth
                lambda_grad = cfg.lambda_grad
                lambda_opacity = cfg.lambda_opacity
                depth_loss_type = cfg.depth_loss_type

                out = model(
                    data, 
                    lambda_mse=lambda_mse, 
                    lambda_lpips=lambda_lpips, 
                    lambda_depth=lambda_depth, 
                    lambda_grad=lambda_grad, 
                    lambda_opacity=lambda_opacity, 
                    depth_loss_type=depth_loss_type
                )

                loss = out['loss']
                psnr = out['psnr']
                ssim = out['ssim']
                lpips = out['lpips']
                abs_diff = out['abs_diff']
                abs_rel = out['abs_rel']
                sq_rel = out['sq_rel']
                delta_1 = out['delta_1']


                accelerator.backward(loss)

                # synchronize to update model  
                if accelerator.sync_gradients:
                    # gradient clipping to avoid exploding gradients
                    accelerator.clip_grad_norm_(model.parameters(), cfg.gradient_clip)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                loss_val = loss.detach().item()
                psnr_val = psnr.detach().item()
                ssim_val = ssim.detach().item()
                lpips_val = lpips.detach().item()
                abs_diff_val = abs_diff.detach().item()
                abs_rel_val = abs_rel.detach().item()
                sq_rel_val = sq_rel.detach().item()
                delta_1_val = delta_1.detach().item()


                total_loss += loss_val
                total_psnr += psnr_val
                total_ssim += ssim_val
                total_lpips += lpips_val
                total_abs_diff += abs_diff_val
                total_abs_rel += abs_rel_val
                total_sq_rel += sq_rel_val
                total_delta_1 += delta_1_val

            if accelerator.is_main_process:
                pbar.update(1)
                mem_free, mem_total = torch.cuda.mem_get_info()
                pbar.set_postfix({
                    "ls": loss_val,
                    "psnr": psnr_val,
                    "vr": round((mem_total-mem_free)/1024**3),
                })

                if i % (2 * cfg.gradient_accumulation_steps) == 0:
                    run.log({
                        "Learning rate (10 steps)": scheduler.get_last_lr()[0], 
                        "lambda MSE (10 steps)": lambda_mse, 
                        "lambda LPIPS (10 steps)": lambda_lpips,
                        "Train loss (10 steps)": loss_val, 
                        "Train psnr (10 steps)": psnr_val,
                        "Train ssim (10 steps)": ssim_val,
                        "Train lpips (10 steps)": lpips_val,
                        "Train depth loss (10 steps)": out.get('loss_depth', torch.tensor(0)).detach().item(),
                        "Train depth grad loss (10 steps)": out.get('loss_depth_grad', torch.tensor(0)).detach().item(),
                        # "Train abs_diff (10 steps)": abs_diff_val,
                        # "Train abs_rel (10 steps)": abs_rel_val,
                        # "Train sq_rel (10 steps)": sq_rel_val,
                        # "Train delta_1 (10 steps)": delta_1_val,
                    })

                # save log images
                if i % 500 == 0:
                    with torch.no_grad():
                        gt_images = data['images_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                        gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3)    # [B * output_size, V * output_size, 3]
                        kiui.write_image(f'{cfg.workspace}/{epoch}_{i}_train_gt_images.jpg', gt_images)
                    
                        gt_mask = data['masks_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                        gt_mask = gt_mask.transpose(0, 3, 1, 4, 2).reshape(-1, gt_mask.shape[1] * gt_mask.shape[3], 1)    # [B * output_size, V * output_size, 3]
                        kiui.write_image(f'{cfg.workspace}/{epoch}_{i}_train_gt_mask.jpg', gt_mask)

                        gt_depth = data['depths_input'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                        gt_depth = gt_depth.transpose(0, 3, 1, 4, 2).reshape(-1, gt_depth.shape[1] * gt_depth.shape[3], 1)  # [B * output_size, V * output_size, 3]
                        kiui.write_image(f'{cfg.workspace}/{epoch}_{i}_train_gt_depth.jpg', gt_depth)

                        pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                        pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)  # [B * output_size, V * output_size, 3]
                        kiui.write_image(f'{cfg.workspace}/{epoch}_{i}_train_pred_images.jpg', pred_images)

                        pred_mask = out['alphas_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                        pred_mask = pred_mask.transpose(0, 3, 1, 4, 2).reshape(-1, pred_mask.shape[1] * pred_mask.shape[3], 1)  # [B * output_size, V * output_size, 3]
                        kiui.write_image(f'{cfg.workspace}/{epoch}_{i}_train_pred_mask.jpg', pred_mask)

                        pred_depth = out['depths_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                        pred_depth = pred_depth / np.max(pred_depth)  # normalize to [0, 1] for better visualization
                        pred_depth = pred_depth.transpose(0, 3, 1, 4, 2).reshape(-1, pred_depth.shape[1] * pred_depth.shape[3], 1)  # [B * output_size, V * output_size, 3]
                        kiui.write_image(f'{cfg.workspace}/{epoch}_{i}_train_pred_depth.jpg', pred_depth)

                        pred_depth_rasterized = out['depths_pred_rasterized'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                        pred_depth_rasterized = pred_depth_rasterized / np.max(pred_depth_rasterized)  # normalize to [0, 1] for better visualization
                        pred_depth_rasterized = pred_depth_rasterized.transpose(0, 3, 1, 4, 2).reshape(-1, pred_depth_rasterized.shape[1] * pred_depth_rasterized.shape[3], 1)  # [B * output_size, V * output_size, 3]
                        kiui.write_image(f'{cfg.workspace}/{epoch}_{i}_train_pred_depth_rasterized.jpg', pred_depth_rasterized)
        
            del out, loss, psnr, ssim, lpips
        if accelerator.is_main_process:
            pbar.close()
        
        torch.cuda.empty_cache()

        total_loss_tensor = torch.tensor(total_loss, device=accelerator.device)
        total_psnr_tensor = torch.tensor(total_psnr, device=accelerator.device)
        
        total_loss = accelerator.gather_for_metrics(total_loss_tensor).mean().item()
        total_psnr = accelerator.gather_for_metrics(total_psnr_tensor).mean().item()

        if accelerator.is_main_process:
            total_loss /= len(train_dataloader)
            total_psnr /= len(train_dataloader)
            total_ssim /= len(train_dataloader)
            total_lpips /= len(train_dataloader)
            total_abs_diff /= len(train_dataloader)
            total_abs_rel /= len(train_dataloader)
            total_sq_rel /= len(train_dataloader)
            total_delta_1 /= len(train_dataloader)
            accelerator.print(f"[TRAIN INFO] Epoch: {epoch + 1} loss: {total_loss:.6f} psnr: {total_psnr:.4f} ssim: {total_ssim:.4f} lpips: {total_lpips:.4f} abs_diff: {total_abs_diff:.4f} abs_rel: {total_abs_rel:.4f} sq_rel: {total_sq_rel:.4f} delta_1: {total_delta_1:.4f}")
            run.log({"Train loss (Epoch)": total_loss, "Train psnr (Epoch)": total_psnr, "Train ssim (Epoch)": total_ssim, "Train lpips (Epoch)": total_lpips, "Train abs_diff (Epoch)": total_abs_diff, "Train abs_rel (Epoch)": total_abs_rel, "Train sq_rel (Epoch)": total_sq_rel, "Train delta_1 (Epoch)": total_delta_1})

        accelerator.wait_for_everyone()
        accelerator.save_state(output_dir=f'{cfg.workspace}/lastest')

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
                pbar2 = tqdm(test_dataloader, desc=f"[E] E{epoch + 1}/{cfg.num_epochs}")

            for i, data in enumerate(test_dataloader):
                out = model(data)

                psnr = out['psnr']
                ssim = out['ssim']
                lpips = out['lpips']
                total_psnr += psnr.detach().item()
                total_ssim += ssim.detach().item()
                total_lpips += lpips.detach().item()

                # abs_diff = out['abs_diff']
                # abs_rel = out['abs_rel']
                # sq_rel = out['sq_rel']
                # delta_1 = out['delta_1']
                # total_abs_diff += abs_diff.detach().item()
                # total_abs_rel += abs_rel.detach().item()
                # total_sq_rel += sq_rel.detach().item()
                # total_delta_1 += delta_1.detach().item()

                if accelerator.is_main_process:
                    pbar2.update(1)
                    if i % 100 == 0:
                        gt_images = data['images_output'].detach().cpu().numpy()    # [B, V, 3, output_size, output_size]
                        gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3)
                        kiui.utils.write_image(f'{cfg.workspace}/{epoch}_{i}_eval_gt_images.jpg', gt_images)

                        pred_images = out['images_pred'].detach().cpu().numpy()     # [B, V, 3, output_size, output_size]
                        pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                        kiui.utils.write_image(f'{cfg.workspace}/{epoch}_{i}_eval_pred_images.jpg', pred_images)

                del out, psnr, ssim, lpips

            if accelerator.is_main_process:
                pbar2.close()
            torch.cuda.empty_cache()

            total_psnr_tensor = torch.tensor(total_psnr, device=accelerator.device)
            total_ssim_tensor = torch.tensor(total_ssim, device=accelerator.device)
            total_lpips_tensor = torch.tensor(total_lpips, device=accelerator.device)
            # total_abs_diff_tensor = torch.tensor(total_abs_diff, device=accelerator.device)
            # total_abs_rel_tensor = torch.tensor(total_abs_rel, device=accelerator.device)
            # total_sq_rel_tensor = torch.tensor(total_sq_rel, device=accelerator.device)
            # total_delta_1_tensor = torch.tensor(total_delta_1, device=accelerator.device)
            
            total_psnr = accelerator.gather_for_metrics(total_psnr_tensor).mean().item()
            total_ssim = accelerator.gather_for_metrics(total_ssim_tensor).mean().item()
            total_lpips = accelerator.gather_for_metrics(total_lpips_tensor).mean().item()
            # total_abs_diff = accelerator.gather_for_metrics(total_abs_diff_tensor).mean().item()
            # total_abs_rel = accelerator.gather_for_metrics(total_abs_rel_tensor).mean().item()
            # total_sq_rel = accelerator.gather_for_metrics(total_sq_rel_tensor).mean().item()
            # total_delta_1 = accelerator.gather_for_metrics(total_delta_1_tensor).mean().item()
            
            if accelerator.is_main_process:
                total_psnr /= len(test_dataloader)
                total_ssim /= len(test_dataloader)
                total_lpips /= len(test_dataloader)
                # total_abs_diff /= len(test_dataloader)
                # total_abs_rel /= len(test_dataloader)
                # total_sq_rel /= len(test_dataloader)
                # total_delta_1 /= len(test_dataloader)
                run.log({"Test psnr (Epoch)": total_psnr, "Test ssim (Epoch)": total_ssim, "Test lpips (Epoch)": total_lpips, "Test abs_diff (Epoch)": total_abs_diff, "Test abs_rel (Epoch)": total_abs_rel, "Test sq_rel (Epoch)": total_sq_rel, "Test delta_1 (Epoch)": total_delta_1})
                accelerator.print(f"[EVAL INFO] Epoch: {epoch + 1} psnr: {total_psnr:.4f} ssim: {total_ssim:.4f} lpips: {total_lpips:.4f} abs_diff: {total_abs_diff:.4f} abs_rel: {total_abs_rel:.4f} sq_rel: {total_sq_rel:.4f} delta_1: {total_delta_1:.4f}")

            if total_psnr > best_psnr_eval:
                best_psnr_eval = total_psnr
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    accelerator.print("Best found => Saving model....")
                    accelerator.save_model(model, f'{cfg.workspace}/best')


if __name__ == "__main__":
    main()