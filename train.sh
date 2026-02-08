nohup accelerate launch --config_file accelerate_configs/gpu1.yaml main.py big \
    --resume /workspace/LGM-from-sratch/pretrained/model_fp16_fixrot.safetensors --fine_tune \
    --workspace workspace --data_path /workspace/100-dataset-9-views \
    --depth1_path /workspace/100-dataset-9-views \
    --depth2_path None \
    --depth3_path None \
    --depth4_path None \
    --lambda_depth 0.01 --lambda_grad -1 --lambda_opacity -1 --depth_loss_type l1 \
    --lambda_mse_start 1.0 --lambda_mse_end 1.0 \
    --lambda_lpips_start 1.0 --lambda_lpips_end 1.0 \
    --num_workers 4 --batch_size 1 --mixed_precision fp16 --input_size 160 --splat_size 80 --pixel_align --self_supervised \
    --output_size 256 --num_epochs 100 --train_size 0.8 --num_views_input 9 --num_views_output 9 \
    --lr 1e-5 --gradient_accumulation_steps 4 --warmup_steps 100 \
    --wandb_project_name LGM_4001 --wandb_experiment_id None \
    --wandb_experiment_name adaptive_LGM-with-depthlossL1-no_grad-opacity \
    --wandb_key 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 
    > train.log 2>&1 &