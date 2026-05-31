nohup accelerate launch --config_file accelerate_configs/gpu2.yaml main.py big \
    --resume /workspace/LGM-from-sratch/best_phase1/best_phase1_model.safetensors --fine_tune \
    --workspace workspace \
    --data_path /path/to/dataset \
    --depth1_path /path/to/depth_dataset \
    --lambda_depth 0.5 --lambda_depth_rank 0.3 --depth_loss_type l1 \
    --batch_size 6 --mixed_precision fp16 --input_size 160 --splat_size 160 --pixel_align \
    --output_size 512 --num_epochs 50 --train_size 0.8 --num_views_input 9 --num_views_output 9 \
    --lr 1e-4 --gradient_accumulation_steps 4 --warmup_steps 2500 \
    --wandb_project_name YOUR_PROJECT_NAME \
    --wandb_experiment_id YOUR_EXPERIMENT_ID \
    --wandb_experiment_name YOUR_EXPERIMENT_NAME \
    --wandb_key YOUR_WANDB_KEY \
    > train.log 2>&1 &