# DaLGM

The official implementation of DaLGM, a depth-aware extension of the Large Multi-View Gaussian Model (LGM) for feed-forward 3D object reconstruction. Given 9 input views, the model predicts a 3D Gaussian Splatting (3DGS) representation and renders high-quality novel views while improving geometric fidelity and training efficiency through depth supervision and Gaussian pruning.

---

## Overview

The pipeline takes multi-view RGB images of an object as input, predicts a set of 3D Gaussians via a UNet, and renders novel views using a differentiable Gaussian rasterizer. Key extensions over the original LGM include:

- **Adaptive input views** — input views are sampled randomly from fixed azimuth bands during training, improving robustness
- **Pixel-aligned Gaussians** — each Gaussian is placed along a camera ray at a learned depth, giving better geometric grounding and direct depth map extraction
- **Depth supervision** — pixel-aligned depth is supervised against ground-truth depth maps using L1/L2/Huber/BerHu losses with depth-aware RANKING loss
- **Gaussian pruning** — voxel-grid clustering removes duplicate/low-opacity Gaussians before rendering

---

## Setup

### 1. Install dependencies

Recommend for reproduciblity:

- CUDA version: 13.0 or 12.8
- GPU type: NVIDIA RTX5880Ada
- Num GPUs: 2

Clone the repository:

```bash
git clone https://github.com/Lhhiep-maxcode/DaLGM.git
cd DaLGM
```

Create and activate a Conda environment:

```bash
conda create -n dalgm python=3.12 -y
conda activate dalgm
```

Install all dependencies (replace `13.0` with your CUDA version, e.g., `12.8`). Currently, the installation script has been verified to work with CUDA 13.0 and CUDA 12.8.

```bash
bash setup.sh 13.0
```

This will install PyTorch, xFormers, `diff-gaussian-rasterization`, `nvdiffrast`, and all Python requirements, then download the pretrained checkpoint.

#### Backup (Only need if command `bash setup.sh 13.0` failed)

Manual install if needed:

```bash
git clone https://github.com/Lhhiep-maxcode/DaLGM.git
cd DaLGM

conda create -n dalgm python=3.12 -y
conda activate dalgm

pip install torch torchvision --index-url TORCH_INDEX_URL

pip install xformers --index-url TORCH_INDEX_URL

git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization

pip install ./diff-gaussian-rasterization --no-build-isolation

pip install ./wheels/nvdiffrast-0.3.3-py3-none-any.whl

pip install -r requirements.txt

mkdir -p pretrained
cd pretrained
wget https://huggingface.co/Hiepppp/LGM/resolve/main/model_fp16_fixrot.safetensors
cd ..

mkdir -p best_phase1
cd best_phase1
pip install gdown
gdown 1t1HkFyPrvCdMgmQi__cIEx1pnf4hbWtf
cd ..
cd ..

pip install kaggle

mkdir 10k-dataset-9-views
kaggle datasets download laihoanghiep/10k-dataset-9-views-depth
kaggle datasets download laihoanghiep/10k-dataset-9-views

unzip 10k-dataset-9-views-depth.zip -d 10k-dataset-9-views
unzip 10k-dataset-9-views.zip -d 10k-dataset-9-views
```

### 2. Data

After success installation, the dataset follows this layout:

```
dataset_root/
├── archive_001/
│   └── object_name/
│       ├── rgb/
│       │   ├── 000.png (elev: 0, azim: 0)
│       │   ├── 001.png (elev: 0, azim: 5.625)
|       |   ├── ...
│       │   └── 063.png (elev: 0, azim: 354.375)
│       │   └── 064.png (elev: 90, azim: 180)
│       └── depth/
│           ├── 000.npz
│           └── ...
```

---

## Training

Review the `train.sh` script and modify it if necessary. For reproducibility, you only need to update the following variables to match your local environment:

- `data_path`
- `depth1_path`
- `wandb_project_name`
- `wandb_experiment_id` (can be set to `None`)
- `wandb_key`

Once the configuration is ready, start training with:

```bash
bash train.sh
```

Or manually:

```bash
accelerate launch --config_file accelerate_configs/gpu2.yaml main.py big \
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
```

---

## Evaluation

### Gaussian-level eval (PSNR / SSIM / LPIPS)

```bash
python eval.py big \
    --resume /path/to/checkpoint.safetensors --fine_tune \
    --data_path /path/to/dataset \
    --depth1_path /path/to/depth_dataset \
    --workspace eval_output \
    --pixel_align --input_size 160 --splat_size 160
```

### Mesh-level eval (RGB + depth + Chamfer Distance)

**Step 1 — Export Gaussians to .ply:**

```bash
python export_lgm_gaussians.py \
    --config big \
    --resume /path/to/checkpoint.safetensors --fine-tune \
    --data-path /path/to/dataset \
    --depth1-path /path/to/depth_dataset \
    --eval-path /path/to/eval_dataset \
    --outdir workspace/lgm_assets \
    --pixel-align --input-size 160 --splat-size 160
```

**Step 2 — Convert .ply to .glb:**

```bash
python batch_convert_lgm_ply_to_glb.py \
    --config big \
    --ply-root workspace/lgm_assets/meshes \
    --nerf-iters 512 --mesh-iters 2048 --uv-iters 512
```

**Step 3 — Evaluate meshes:**

```bash
python eval_lgm_mesh.py \
    --data-path /path/to/dataset \
    --depth1-path /path/to/depth_dataset \
    --eval-path /path/to/eval_dataset \
    --mesh-path workspace/lgm_assets/meshes \
    --outdir workspace/lgm_mesh_eval \
    --depth-source eval
```

---

## Inference

### From real multi-view images (3D reconstruction)

```bash
python 3Dreconstruct_infer.py big \
    --resume pretrained/model_fp16_fixrot.safetensors --fine_tune \
    --workspace output/ \
    --pixel_align --input_size 160 --splat_size 160
```

Edit the `path` variable at the bottom of `3Dreconstruct_infer.py` to point to your image folder. Expected folder layout: `rgb/000.png`, `rgb/001.png`, etc.

### From text prompt (3D generation via MVDream)

```bash
python 3Dgen_infer.py big \
    --resume pretrained/model_fp16_fixrot.safetensors \
    --test_path /path/to/image.png \
    --workspace output/
```

---

## Requirements

Main dependencies:

- Python 3.10+
- PyTorch 2.x + CUDA 12.8
- `diff-gaussian-rasterization` (custom, from [ashawkey](https://github.com/ashawkey/diff-gaussian-rasterization))
- `nvdiffrast`
- `nerfacc`, `xformers`, `kiui`, `accelerate`, `trimesh`
