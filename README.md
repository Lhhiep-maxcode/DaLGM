# DaLGM

The official implementation of DaLGM, a depth-aware extension of the Large Multi-View Gaussian Model (LGM) for feed-forward 3D object reconstruction. Given 9 input views, the model predicts a 3D Gaussian Splatting (3DGS) representation and renders high-quality novel views while improving geometric fidelity and training efficiency through depth supervision and Gaussian pruning.

---

## I. Overview

The pipeline takes multi-view RGB images of an object as input, predicts a set of 3D Gaussians via a UNet, and renders novel views using a differentiable Gaussian rasterizer. Key extensions over the original LGM include:

- **Adaptive input views** — input views are sampled randomly from fixed azimuth bands during training, improving robustness
- **Pixel-aligned Gaussians** — each Gaussian is placed along a camera ray at a learned depth, giving better geometric grounding and direct depth map extraction
- **Depth supervision** — pixel-aligned depth is supervised against ground-truth depth maps using L1/L2/Huber/BerHu losses with depth-aware RANKING loss
- **Gaussian pruning** — voxel-grid clustering removes duplicate/low-opacity Gaussians before rendering

---

## II. Setup

### 1. Install dependencies

Recommend for reproduciblity:

- **CUDA version**: 13.0 or 12.8
- **GPU type**: NVIDIA RTX5880Ada
- **Num GPUs**: 2
- **Min available space**: 190 GB

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

### 2. Training Data

After success installation, the training dataset follows this layout:

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
#### Naming convention:
- `000`–`063`: Side views with **0° elevation** and azimuth angles uniformly sampled from **0°** to **354.375°** (step size: **5.625°**).
- `064`: Top-down view with **90° elevation** and **180° azimuth**.

---

## III. Training

Review the `train.sh` script and modify it if necessary. For reproducibility, you only need to update the following variables to match your local environment:

- `data_path`
- `depth1_path`
- `wandb_project_name`
- `wandb_experiment_id` (can be set to `None`)
- `wandb_experiment_name`
- `wandb_key`

**Optional**: If you want to try with different value of threshold for proposed pruning algorithm + weight value for depth ranking loss. Try to adjust `alpha_threshold`, `distance_threshold`, `scale_threshold`, `rot_threshold`, `rgb_threshold`, and `lambda_depth_rank`.

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

## IV. Evaluation
 
We evaluate on two benchmarks: **GSO** and **ABO**. The pipeline has two evaluation levels.
 
### 1. Download the best checkpoint
 
```bash
pip install kaggle
kaggle datasets download memaybeo12/best-depthloss-depth-ranking-2 -p checkpoints --unzip
```
 
### 2. Download evaluation data
 
#### GSO

```bash
# RGB input views
kaggle datasets download laihoanghiep/100-gso-rgba-input -p data/gso/rgb --unzip

# Novel views
kaggle datasets download laihoanghiep/100-gso-16-views-for-eval -p data/gso/eval --unzip

# Ground-truth meshes
kaggle datasets download laihoanghiep/100-gso-mesh-gt -p data/gso/mesh_gt --unzip
```

#### ABO

```bash
# RGB input views
kaggle datasets download laihoanghiep/100-abo-rgb-input -p data/abo/rgb --unzip

# Novel views
kaggle datasets download laihoanghiep/100-abo-16-views-for-eval -p data/abo/eval --unzip

# Ground-truth meshes
kaggle datasets download laihoanghiep/100-abo-mesh-gt -p data/abo/mesh_gt --unzip
```
#### Naming Convention:
- The input dataset for evaluation (`100-abo-rgb-input`, `100-gso-rgb-input`) follows convention as training dataset
- The 16-view dataset for evaluation (`100-abo-16-views-for-eval`, `100-gso-16-views-for-eval`) follows convention as following:
    - `000` - `007`: Side views with **30° elevation** and azimuth angles uniformly sampled from **0°** to **315°** (step size: **45°**)
    - `008` - `015`: Side views with **60° elevation** and azimuth angles uniformly sampled from **0°** to **315°** (step size: **45°**)

 
### 3. Run evaluation
 
Convert the exported Gaussians to meshes, then compute geometric metrics:
 
```bash
# Convert Gaussians to meshes
python export_lgm_gaussians.py \
    --config big \
    --resume checkpoints/model.safetensors \
    --fine-tune \
    --data-path data/<benchmark>/rgb \
    --eval-path data/<benchmark>/eval \
    --outdir workspace/lgm_mesh_assets_<benchmark> \
    --val-size 1 \
    --input-size 160 \
    --splat-size 160 \
    --output-size 512 \
    --num-views-input 9 \
    --num-views-output 16 \
    --pixel-align \
    --batch-size 2 \
    --num-workers 4 \
    --mixed-precision fp16 \
    --convert \
    --nerf-iters 512 \
    --mesh-iters 1024 \
    --uv-iters 0
 
# Compute mesh metrics
python eval_lgm_mesh.py \
    --data-path data/<benchmark>/rgb \
    --eval-path data/<benchmark>/eval \
    --mesh-path workspace/lgm_mesh_assets_<benchmark>/meshes \
    --gt-mesh-path data/<benchmark>/mesh_gt \
    --outdir workspace/lgm_mesh_eval_<benchmark> \
    --val-size 1 \
    --input-size 160 \
    --splat-size 160 \
    --output-size 512 \
    --depth-render-size 512 \
    --num-views-input 9 \
    --num-views-output 16 \
    --pixel-align \
    --batch-size 1 \
    --depth-source eval \
    --flip-uv-y
```
  
---
 
## V. Inference
 
### From real multi-view images (3D reconstruction)
 
```bash
python 3Dreconstruct_infer.py big \
    --resume checkpoints/best-depthloss-depth-ranking-2/model.safetensors --fine_tune \
    --workspace output/ \
    --pixel_align --input_size 160 --splat_size 160
```
 
Edit the `path` variable at the bottom of `3Dreconstruct_infer.py` to point to your image folder. Expected folder layout: `rgb/000.png`, `rgb/001.png`, etc.

## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{dalgm2026,
  title={DaLGM: Depth-Aware Geometry Supervision and Efficient Gaussian Pruning for Feed-Forward 3D Reconstruction},
  author={Hoang Hiep Lai and Duy Thanh Tran and Thanh Long Vu and Thi Chau Ma},
  journal={The Visual Computer},
  year={2026},
  doi={xxxxxxxx},
  note={Under review}
}
```
