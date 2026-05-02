from __future__ import annotations

"""
Export predicted LGM Gaussians to .ply, optionally convert each .ply to .glb.


Typical 4-view export only:
python export_lgm_gaussians.py \
  --config big \
  --resume /kaggle/input/models/lihonghip/lgm_4000/pytorch/default/9/model.safetensors \
  --fine-tune \
  --data-path /kaggle/input/datasets/laihoanghiep/10k-dataset-9-views \
  --depth1-path /kaggle/input/datasets/laihoanghiep/10k-dataset-9-views-depth \
  --eval-path /kaggle/input/datasets/tdthanh/1k-objaverse-16-views-for-eval \
  --outdir /kaggle/working/workspace/lgm_mesh_assets \
  --input-size 256 --splat-size 128 --output-size 512 \
  --num-views-input 4 --num-views-output 16 \
  --no-pixel-align \
  --batch-size 2

Typical 9-view pixel-align:
python export_lgm_gaussians.py \
  --config big \
  --resume /path/to/model.safetensors \
  --fine-tune \
  --data-path /kaggle/input/datasets/laihoanghiep/10k-dataset-9-views \
  --depth1-path /kaggle/input/datasets/laihoanghiep/10k-dataset-9-views-depth \
  --eval-path /kaggle/input/datasets/tdthanh/1k-objaverse-16-views-for-eval \
  --outdir /kaggle/working/workspace/lgm_mesh_assets_9v \
  --input-size 160 --splat-size 160 --output-size 512 \
  --num-views-input 9 --num-views-output 16 \
  --pixel-align \
  --batch-size 1

Add --convert to call batch_convert_lgm_ply_to_glb.py after PLY export.
"""

import argparse
import copy
import csv
import os
import random
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from safetensors.torch import load_file
from tqdm.auto import tqdm

from core.dataset import ObjaverseDataset as Dataset
from core.model import LGM
from core.model_config import config_defaults
from core.utils import get_rays


def none_if_text(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    if str(x).strip().lower() in {"", "none", "null"}:
        return None
    return x


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export LGM model predictions as Gaussian .ply files, optionally convert to .glb.")

    p.add_argument("--config", default="big", choices=sorted(config_defaults.keys()), help="Model config preset from core.model_config, e.g. big/small/tiny/lrm.")
    p.add_argument("--resume", required=True, help="Checkpoint path: .safetensors/.pt or accelerator state dir when --no-fine-tune.")
    p.add_argument("--fine-tune", dest="fine_tune", action="store_true", help="Load a plain model checkpoint, matching your old eval.py --fine_tune behavior.")
    p.add_argument("--no-fine-tune", dest="fine_tune", action="store_false", help="Load accelerator state dir instead of a plain checkpoint.")
    p.set_defaults(fine_tune=True)

    p.add_argument("--data-path", required=True)
    p.add_argument("--depth1-path", default=None)
    p.add_argument("--depth2-path", default=None)
    p.add_argument("--depth3-path", default=None)
    p.add_argument("--depth4-path", default=None)
    p.add_argument("--eval-path", required=True)
    p.add_argument("--outdir", required=True, help="Output folder. PLY/GLB are saved under outdir/meshes/<archive>/<object>.*")
    p.add_argument("--manifest-name", default="lgm_export_manifest.csv")

    p.add_argument("--train-size", type=float, default=0.8)
    p.add_argument("--test-size", type=float, default=0.1)
    p.add_argument("--val-size", type=float, default=0.1)
    p.add_argument("--object-start", type=int, default=None)
    p.add_argument("--object-end", type=int, default=None)
    p.add_argument("--max-objects", type=int, default=None)
    p.add_argument("--object-list", default=None,
                   help="Optional CSV/text file with fixed benchmark object IDs. When set, only these objects are exported/converted.")

    # Important model/dataset/render params.
    p.add_argument("--input-size", type=int, default=None)
    p.add_argument("--splat-size", type=int, default=None)
    p.add_argument("--output-size", type=int, default=None)
    p.add_argument("--num-views-input", type=int, default=None)
    p.add_argument("--num-views-output", type=int, default=None)
    p.add_argument("--num-views-total", type=int, default=65)
    p.add_argument("--fovy", type=float, default=60.0)
    p.add_argument("--cam-radius", type=float, default=1.5)
    p.add_argument("--znear", type=float, default=0.5)
    p.add_argument("--zfar", type=float, default=2.5)
    p.add_argument("--max-distance", type=float, default=None, help="Used by pixel-align branch; defaults to config value.")
    p.add_argument("--pixel-align", dest="pixel_align", action="store_true")
    p.add_argument("--no-pixel-align", dest="pixel_align", action="store_false")
    p.set_defaults(pixel_align=None)
    p.add_argument("--self-supervised", dest="self_supervised", action="store_true")
    p.add_argument("--no-self-supervised", dest="self_supervised", action="store_false")
    p.set_defaults(self_supervised=False)
    p.add_argument("--compute-surface", dest="compute_surface", action="store_true")
    p.add_argument("--no-compute-surface", dest="compute_surface", action="store_false")
    p.set_defaults(compute_surface=True)

    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--mixed-precision", default=None, choices=[None, "no", "fp16", "bf16"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing .ply files.")

    # Optional conversion stage.
    p.add_argument("--convert", action="store_true", help="After PLY export, run batch_convert_lgm_ply_to_glb.py on the exported PLY folder.")
    p.add_argument("--converter-script", default="batch_convert_lgm_ply_to_glb.py", help="Path to batch_convert_lgm_ply_to_glb.py.")
    p.add_argument("--convert-source-script", default="convert.py", help="Original LGM convert.py used as source for Converter class.")
    p.add_argument("--nerf-iters", type=int, default=512)
    p.add_argument("--nerf-resolution", type=int, default=128)
    p.add_argument("--mesh-iters", type=int, default=2048)
    p.add_argument("--mesh-resolution", type=int, default=512)
    p.add_argument("--mesh-decimate-target", type=int, default=50000)
    p.add_argument("--uv-iters", type=int, default=512)
    p.add_argument("--texture-resolution", type=int, default=1024)
    p.add_argument("--uv-padding", type=int, default=2)
    p.add_argument("--max-ply-points", "--convert-max-ply-points", dest="convert_max_ply_points", type=int, default=None,
                   help="Passed to converter: skip PLY files with more than this many points/Gaussians.")
    p.add_argument("--target-glbs", "--convert-target-glbs", "--max-glbs", dest="target_glbs", type=int, default=None,
                   help="Passed to converter: stop when this many usable GLB files exist or have been converted.")
    p.add_argument("--convert-log-name", default="lgm_convert_manifest.csv",
                   help="Passed to converter: CSV log filename written under outdir/meshes.")
    p.add_argument("--overwrite-glb", action="store_true")

    return p.parse_args()


def build_cfg(args: argparse.Namespace):
    cfg = copy.deepcopy(config_defaults[args.config])

    cfg.resume = args.resume
    cfg.fine_tune = bool(args.fine_tune)
    cfg.data_path = args.data_path
    cfg.depth1_path = none_if_text(args.depth1_path)
    cfg.depth2_path = none_if_text(args.depth2_path)
    cfg.depth3_path = none_if_text(args.depth3_path)
    cfg.depth4_path = none_if_text(args.depth4_path)
    cfg.eval_path = args.eval_path
    cfg.workspace = args.outdir

    cfg.train_size = args.train_size
    cfg.test_size = args.test_size
    cfg.val_size = args.val_size
    cfg.num_views_total = args.num_views_total
    cfg.fovy = args.fovy
    cfg.cam_radius = args.cam_radius
    cfg.znear = args.znear
    cfg.zfar = args.zfar
    cfg.self_supervised = args.self_supervised
    cfg.compute_surface = args.compute_surface

    if args.input_size is not None:
        cfg.input_size = args.input_size
    if args.splat_size is not None:
        cfg.splat_size = args.splat_size
    if args.output_size is not None:
        cfg.output_size = args.output_size
    if args.num_views_input is not None:
        cfg.num_views_input = args.num_views_input
    if args.num_views_output is not None:
        cfg.num_views_output = args.num_views_output
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.mixed_precision is not None:
        cfg.mixed_precision = args.mixed_precision
    if args.pixel_align is not None:
        cfg.pixel_align = bool(args.pixel_align)
    if args.max_distance is not None:
        cfg.max_distance = args.max_distance

    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def move_to_device(x: Any, device: str) -> Any:
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        # Keep string lists as-is; move tensor lists recursively.
        if all(isinstance(v, str) for v in x):
            return list(x)
        return type(x)(move_to_device(v, device) for v in x)
    return x


def load_checkpoint_into_model(model: torch.nn.Module, cfg, device: str) -> None:
    if not cfg.fine_tune:
        raise RuntimeError("This script currently supports plain checkpoints with --fine-tune. For accelerator state dirs, export with your eval.py logic or add Accelerator.load_state here.")

    if cfg.resume is None:
        raise ValueError("--resume is required")
    if str(cfg.resume).endswith(".safetensors"):
        ckpt = load_file(cfg.resume, device="cpu")
    else:
        ckpt = torch.load(cfg.resume, map_location="cpu")

    state_dict = model.state_dict()
    loaded = 0
    skipped = 0
    for k, v in ckpt.items():
        if k in state_dict and state_dict[k].shape == v.shape:
            state_dict[k].copy_(v)
            loaded += 1
        else:
            skipped += 1
            if k in state_dict:
                print(f"[WARN] mismatching shape for {k}: ckpt {tuple(v.shape)} != model {tuple(state_dict[k].shape)}; ignored")
            else:
                print(f"[WARN] unexpected param {k}: {tuple(v.shape)}")
    print(f"[INFO] checkpoint loaded tensors={loaded}, skipped={skipped}")
    model.to(device)


@torch.no_grad()
def predict_gaussians_only(model: LGM, data: dict, cfg) -> list[torch.Tensor]:
    """Mirror the Gaussian part of LGM.forward without rendering metrics."""
    images = data["input"]  # [B, V, 9, H, W]
    gaussians = model.forward_gaussians(images)  # [B, V*splat*splat, 14]
    B, V_in = data["cam_poses_input"].shape[:2]

    if cfg.pixel_align:
        rays_d = []
        rays_o = []
        cam_poses_input = data["cam_poses_input"].reshape(-1, 4, 4)
        for i in range(cam_poses_input.shape[0]):
            ro, rd = get_rays(cam_poses_input[i], cfg.splat_size, cfg.splat_size, cfg.fovy)
            rays_d.append(rd)
            rays_o.append(ro)
        rays_d = torch.stack(rays_d, dim=0).view(B, V_in, cfg.splat_size, cfg.splat_size, 3)
        rays_o = torch.stack(rays_o, dim=0).view(B, V_in, cfg.splat_size, cfg.splat_size, 3)

        pos = gaussians[..., 0:3]
        dist = pos.mean(dim=-1, keepdim=True).sigmoid() * cfg.max_distance
        pos = dist * rays_d.view(B, -1, 3) + rays_o.view(B, -1, 3)
        gaussians = torch.cat([pos, gaussians[..., 3:]], dim=-1)

    return model.gaussian_prune(gaussians)


def object_ids_from_batch(data: dict) -> list[str]:
    object_id = data["object_id"]
    if isinstance(object_id, str):
        return [object_id]
    if isinstance(object_id, (list, tuple)):
        return [str(x) for x in object_id]
    return [str(x) for x in list(object_id)]


def load_object_list(path: Optional[str]) -> Optional[set[str]]:
    if path is None or str(path).strip().lower() in {"", "none", "null"}:
        return None

    allowed: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        sample = f.read(4096)
        f.seek(0)

        if "," in sample or "object_id" in sample:
            reader = csv.DictReader(f)
            for row in reader:
                oid = str(row.get("object_id", "")).strip()
                archive = str(row.get("archive_name", "")).strip()
                item = str(row.get("item_name", "")).strip()
                ply_path = str(row.get("ply_path", "")).strip()
                glb_path = str(row.get("glb_path", "")).strip()

                if not oid and archive and item:
                    oid = f"{archive}/{item}"
                if not oid and ply_path:
                    stem = os.path.splitext(os.path.basename(ply_path))[0]
                    parent = os.path.basename(os.path.dirname(ply_path))
                    oid = f"{parent}/{stem}" if parent else stem
                if not oid and glb_path:
                    stem = os.path.splitext(os.path.basename(glb_path))[0]
                    parent = os.path.basename(os.path.dirname(glb_path))
                    oid = f"{parent}/{stem}" if parent else stem

                if oid:
                    allowed.add(oid)
                    allowed.add(oid.split("/")[-1])
        else:
            for line in f:
                oid = line.strip()
                if oid:
                    allowed.add(oid)
                    allowed.add(oid.split("/")[-1])

    return allowed


def is_allowed_object(object_id: str, allowed: Optional[set[str]]) -> bool:
    if allowed is None:
        return True
    return object_id in allowed or object_id.split("/")[-1] in allowed


def write_manifest(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_converter(args: argparse.Namespace, cfg, ply_root: str, manifest_path: str) -> None:
    script = Path(args.converter_script)
    if not script.is_file():
        # Also try relative to this script's CWD.
        raise FileNotFoundError(f"Cannot find converter script: {script}. Put batch_convert_lgm_ply_to_glb.py in the repo root or pass --converter-script.")

    cmd = [
        sys.executable, str(script),
        "--config", args.config,
        "--convert-script", args.convert_source_script,
        "--ply-root", ply_root,
        "--manifest", manifest_path,
        "--input-size", str(cfg.input_size),
        "--splat-size", str(cfg.splat_size),
        "--output-size", str(cfg.output_size),
        "--num-views-input", str(cfg.num_views_input),
        "--num-views-output", str(cfg.num_views_output),
        "--fovy", str(cfg.fovy),
        "--cam-radius", str(cfg.cam_radius),
        "--znear", str(cfg.znear),
        "--zfar", str(cfg.zfar),
        "--nerf-iters", str(args.nerf_iters),
        "--nerf-resolution", str(args.nerf_resolution),
        "--mesh-iters", str(args.mesh_iters),
        "--mesh-resolution", str(args.mesh_resolution),
        "--mesh-decimate-target", str(args.mesh_decimate_target),
        "--uv-iters", str(args.uv_iters),
        "--texture-resolution", str(args.texture_resolution),
        "--uv-padding", str(args.uv_padding),
        "--device", args.device,
    ]
    if cfg.pixel_align:
        cmd.append("--pixel-align")
    else:
        cmd.append("--no-pixel-align")
    if args.convert_max_ply_points is not None:
        cmd.extend(["--max-ply-points", str(args.convert_max_ply_points)])
    if args.target_glbs is not None:
        cmd.extend(["--target-glbs", str(args.target_glbs)])
    if args.convert_log_name:
        cmd.extend(["--convert-log-name", str(args.convert_log_name)])
    if args.object_list is not None:
        cmd.extend(["--object-list", str(args.object_list)])
    if args.overwrite_glb:
        cmd.append("--overwrite")

    print("[INFO] running converter:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    cfg = build_cfg(args)

    os.makedirs(args.outdir, exist_ok=True)
    mesh_root = os.path.join(args.outdir, "meshes")
    os.makedirs(mesh_root, exist_ok=True)
    manifest_path = os.path.join(args.outdir, args.manifest_name)

    print("[INFO] config:")
    if is_dataclass(cfg):
        print(asdict(cfg))
    else:
        print(cfg)
    print("[INFO] output mesh root:", mesh_root)

    allowed_object_ids = load_object_list(args.object_list)
    if allowed_object_ids is not None:
        print(f"[INFO] using fixed object list: {args.object_list} ({len(allowed_object_ids)} ID aliases)")

    dataset = Dataset(
        data_path=cfg.data_path,
        depth1_path=cfg.depth1_path,
        depth2_path=cfg.depth2_path,
        depth3_path=cfg.depth3_path,
        depth4_path=cfg.depth4_path,
        eval_path=cfg.eval_path,
        cfg=cfg,
        type="val",
    )

    # Optional extra slicing without editing the original Dataset class.
    if args.object_start is not None or args.object_end is not None or args.max_objects is not None:
        start = args.object_start
        end = args.object_end
        dataset.items_depth = dataset.items_depth[start:end]
        if args.max_objects is not None:
            dataset.items_depth = dataset.items_depth[: args.max_objects]

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
        pin_memory=True,
    )

    model = LGM(cfg)
    load_checkpoint_into_model(model, cfg, args.device)
    model.eval()

    rows: list[dict] = []
    exported_or_selected = 0
    with torch.no_grad():
        for data in tqdm(loader, desc="Export LGM Gaussians"):
            object_ids_all = object_ids_from_batch(data)
            selected = [(i, oid) for i, oid in enumerate(object_ids_all) if is_allowed_object(oid, allowed_object_ids)]
            if not selected:
                continue

            selected_indices = [i for i, _ in selected]
            object_ids = [oid for _, oid in selected]
            exported_or_selected += len(object_ids)

            # Fast skip when every selected PLY already exists and no overwrite.
            planned = []
            for oid in object_ids:
                ply_path = os.path.join(mesh_root, *oid.split("/")) + ".ply"
                planned.append((oid, ply_path))
            if (not args.overwrite) and all(os.path.exists(p) for _, p in planned):
                for oid, ply_path in planned:
                    rows.append({"object_id": oid, "ply_path": ply_path, "glb_path": ply_path[:-4] + ".glb", "status": "skipped_existing"})
                write_manifest(manifest_path, rows)
                continue

            data_dev = move_to_device(data, args.device)
            gaussians_all = predict_gaussians_only(model, data_dev, cfg)
            if len(gaussians_all) != len(object_ids_all):
                raise RuntimeError(f"Batch mismatch: got {len(gaussians_all)} Gaussian sets for {len(object_ids_all)} object IDs")

            gaussians_list = [gaussians_all[i] for i in selected_indices]

            for oid, g in zip(object_ids, gaussians_list):
                ply_path = os.path.join(mesh_root, *oid.split("/")) + ".ply"
                os.makedirs(os.path.dirname(ply_path), exist_ok=True)
                glb_path = ply_path[:-4] + ".glb"
                if os.path.exists(ply_path) and not args.overwrite:
                    status = "skipped_existing"
                else:
                    model.gs.save_ply(g.detach().unsqueeze(0), ply_path, compatible=True)
                    status = "exported"
                rows.append({"object_id": oid, "ply_path": ply_path, "glb_path": glb_path, "status": status})

            write_manifest(manifest_path, rows)
            if args.device == "cuda":
                torch.cuda.empty_cache()

    print(f"[INFO] saved manifest: {manifest_path}")
    print(f"[INFO] PLY root: {mesh_root}")
    if allowed_object_ids is not None:
        print(f"[INFO] selected/exported objects from fixed list in this run: {exported_or_selected}")
        if exported_or_selected == 0:
            raise RuntimeError("No objects matched --object-list. Check object_id/archive naming and split settings.")

    if args.convert:
        run_converter(args, cfg, mesh_root, manifest_path)
        print(f"[INFO] GLB files should now be next to the PLY files under: {mesh_root}")


if __name__ == "__main__":
    main()
