nohup accelerate launch --config_file accelerate_configs/gpu2.yaml main.py big \
    --resume /workspace/LGM-from-sratch/best_phase1/model.safetensors --fine_tune \
    --workspace workspace --data_path /workspace/10k-dataset-9-views \
    --depth1_path /workspace/10k-dataset-9-views \
    --depth2_path None \
    --depth3_path None \
    --depth4_path None \
    --lambda_depth 0.5 --lambda_grad 0.5 --lambda_opacity -1 --depth_loss_type l1 \
    --lambda_mse_start 1.0 --lambda_mse_end 1.0 \
    --lambda_lpips_start 1.0 --lambda_lpips_end 1.0 \
    --num_workers 4 --batch_size 3 --mixed_precision fp16 --input_size 160 --splat_size 160 --pixel_align \
    --output_size 512 --num_epochs 50 --train_size 0.8 --num_views_input 9 --num_views_output 9 \
    --lr 1e-4 --gradient_accumulation_steps 4 --warmup_steps 10000 \
    --wandb_project_name LGM_4001 --wandb_experiment_id None \
    --wandb_experiment_name adaptive_LGM-with-depthlossL1-grad-no_opacity-phase2 \
    --wandb_key 2643e7f5dd32fdc64ae63918abf4238ad72c0d60 \
    --gdrive-service-account /workspace/LGM-from-sratch/lgm-uploader-48be37d7c3eb.json \
    --gdrive-folder-id 1VbvnK5DOyeYvGqdvKJcq8yJHX64uarxB \
    > train.log 2>&1 &