from __future__ import annotations

"""
Batch convert LGM Gaussian .ply files to textured .glb files using the Converter class
from your existing convert.py, but with CLI controls for iteration counts.

Example:
python batch_convert_lgm_ply_to_glb.py \
  --config big \
  --ply-root /kaggle/working/workspace/lgm_mesh_assets/meshes \
  --input-size 256 --splat-size 128 --output-size 512 \
  --num-views-input 4 --num-views-output 16 \
  --fovy 60 --cam-radius 1.5 \
  --nerf-iters 512 --mesh-iters 2048 --uv-iters 512
"""

import argparse
import copy
import importlib.util
import os
import random
import sys
import types
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm.auto import tqdm

from core.model_config import config_defaults


def none_if_text(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    if str(x).strip().lower() in {"", "none", "null"}:
        return None
    return x


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch convert LGM .ply Gaussian files to .glb meshes.")
    p.add_argument("--config", default="big", choices=sorted(config_defaults.keys()))
    p.add_argument("--convert-script", default="convert.py", help="Path to original convert.py.")
    p.add_argument("--ply-root", required=True, help="Root folder containing .ply files recursively.")
    p.add_argument("--manifest", default=None, help="Optional CSV manifest from export_lgm_gaussians.py; if provided, only these rows are converted.")

    p.add_argument("--input-size", type=int, default=None)
    p.add_argument("--splat-size", type=int, default=None)
    p.add_argument("--output-size", type=int, default=512)
    p.add_argument("--num-views-input", type=int, default=None)
    p.add_argument("--num-views-output", type=int, default=None)
    p.add_argument("--fovy", type=float, default=60.0)
    p.add_argument("--cam-radius", type=float, default=1.5)
    p.add_argument("--znear", type=float, default=0.5)
    p.add_argument("--zfar", type=float, default=2.5)
    p.add_argument("--pixel-align", dest="pixel_align", action="store_true")
    p.add_argument("--no-pixel-align", dest="pixel_align", action="store_false")
    p.set_defaults(pixel_align=None)
    p.add_argument("--force-cuda-rast", action="store_true")

    p.add_argument("--nerf-iters", type=int, default=512)
    p.add_argument("--nerf-resolution", type=int, default=128)
    p.add_argument("--mesh-iters", type=int, default=2048)
    p.add_argument("--mesh-resolution", type=int, default=512)
    p.add_argument("--mesh-decimate-target", type=int, default=50000)
    p.add_argument("--uv-iters", type=int, default=512)
    p.add_argument("--texture-resolution", type=int, default=1024)
    p.add_argument("--uv-padding", type=int, default=2)

    p.add_argument("--object-start", type=int, default=None)
    p.add_argument("--object-end", type=int, default=None)
    p.add_argument("--max-objects", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_cfg(args: argparse.Namespace):
    cfg = copy.deepcopy(config_defaults[args.config])
    cfg.test_path = None
    cfg.output_size = args.output_size
    cfg.fovy = args.fovy
    cfg.cam_radius = args.cam_radius
    cfg.znear = args.znear
    cfg.zfar = args.zfar
    cfg.force_cuda_rast = args.force_cuda_rast
    if args.input_size is not None:
        cfg.input_size = args.input_size
    if args.splat_size is not None:
        cfg.splat_size = args.splat_size
    if args.num_views_input is not None:
        cfg.num_views_input = args.num_views_input
    if args.num_views_output is not None:
        cfg.num_views_output = args.num_views_output
    if args.pixel_align is not None:
        cfg.pixel_align = bool(args.pixel_align)
    return cfg


def load_converter_class(convert_script: str):
    """Load Converter from convert.py without executing the bottom tyro CLI block."""
    path = Path(convert_script)
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find convert.py at: {path}")
    src = path.read_text(encoding="utf-8")

    # Your convert.py ends with:
    # cfg = tyro.cli(AllConfigs)
    # converter = Converter(cfg).cuda()
    markers = ["\ncfg = tyro.cli(AllConfigs)", "\nif __name__ =="]
    cut_positions = [src.find(m) for m in markers if src.find(m) >= 0]
    if not cut_positions:
        raise RuntimeError("Could not find the bottom CLI block in convert.py; refusing to import because it may execute conversion immediately.")
    src_head = src[: min(cut_positions)]

    module = types.ModuleType("lgm_convert_dynamic")
    module.__file__ = str(path)
    exec(compile(src_head, str(path), "exec"), module.__dict__)
    if "Converter" not in module.__dict__:
        raise RuntimeError("Converter class not found after loading convert.py")
    return module.__dict__["Converter"]


def discover_plys(args: argparse.Namespace) -> list[str]:
    if args.manifest is not None:
        import pandas as pd
        df = pd.read_csv(args.manifest)
        if "ply_path" not in df.columns:
            raise ValueError(f"Manifest {args.manifest} does not contain a ply_path column")
        plys = [str(p) for p in df["ply_path"].dropna().tolist()]
    else:
        plys = []
        for root, _, files in os.walk(args.ply_root):
            for fname in sorted(files):
                if fname.lower().endswith(".ply"):
                    plys.append(os.path.join(root, fname))
        plys = sorted(plys)

    if args.object_start is not None or args.object_end is not None:
        plys = plys[args.object_start: args.object_end]
    if args.max_objects is not None:
        plys = plys[: args.max_objects]
    return plys


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.device != "cuda":
        raise RuntimeError("The original convert.py Converter uses CUDA/nvdiffrast/nerfacc; run with --device cuda.")

    cfg = build_cfg(args)
    Converter = load_converter_class(args.convert_script)
    plys = discover_plys(args)
    if not plys:
        raise RuntimeError(f"No .ply files found under {args.ply_root}")

    print(f"[INFO] converting {len(plys)} PLY files to GLB")
    for ply_path in tqdm(plys, desc="PLY -> GLB"):
        glb_path = os.path.splitext(ply_path)[0] + ".glb"
        if os.path.exists(glb_path) and not args.overwrite:
            print("[SKIP]", glb_path)
            continue

        cfg.test_path = ply_path
        try:
            converter = Converter(cfg).cuda()
            converter.fit_nerf(iters=args.nerf_iters, resolution=args.nerf_resolution)
            converter.fit_mesh(iters=args.mesh_iters, resolution=args.mesh_resolution, decimate_target=args.mesh_decimate_target)
            converter.fit_mesh_uv(iters=args.uv_iters, resolution=args.mesh_resolution, texture_resolution=args.texture_resolution, padding=args.uv_padding)
            converter.export_mesh(glb_path)
            print("[OK]", glb_path)
        except Exception as exc:
            print(f"[ERROR] {ply_path}: {repr(exc)}")
        finally:
            # Best effort cleanup between objects.
            try:
                del converter
            except Exception:
                pass
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
