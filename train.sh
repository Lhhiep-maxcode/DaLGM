nohup accelerate launch --config_file accelerate_configs/gpu2.yaml main.py big \
    --resume best_phase1/best_phase1_model.safetensors --fine_tune \
    --workspace workspace \
    --data_path ../10k-dataset-9-views \
    --depth1_path ../10k-dataset-9-views \
    --lambda_depth 0.5 --lambda_grad -1 --lambda_opacity -1 --lambda_depth_rank 0.3 --depth_loss_type l1 \
    --num_workers 4 --batch_size 6 --mixed_precision fp16 --input_size 160 --splat_size 160 --pixel_align \
    --output_size 512 --num_epochs 50 --train_size 0.8 --num_views_input 9 --num_views_output 9 \
    --alpha_threshold 0.004 --distance_threshold -1 --scale_threshold -1 --rot_threshold -1 --rgb_threshold -1 \
    --lr 1e-4 --gradient_accumulation_steps 4 --warmup_steps 2500 \
    --wandb_project_name YOUR_PROJECT_NAME \
    --wandb_experiment_id None \
    --wandb_experiment_name YOUR_EXPERIMENT_NAME \
    --wandb_key YOUR_WANDB_KEY \
    > train.log 2>&1 &