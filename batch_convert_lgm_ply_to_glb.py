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
import csv
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
    p.add_argument("--object-list", default=None,
                   help="Optional CSV/text file containing fixed benchmark object IDs. If set, only matching PLYs are converted.")
    p.add_argument("--benchmark-name", default="benchmark_objects.csv",
                   help="CSV filename written under --ply-root containing only usable GLB objects. Use this as --object-list for later checkpoints.")

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
    p.add_argument("--max-objects", type=int, default=None, help="Maximum number of PLY candidates to scan/try converting.")
    p.add_argument("--max-ply-points", "--max-points", dest="max_ply_points", type=int, default=None,
                   help="Skip a PLY before conversion if its vertex/Gaussian count is larger than this value.")
    p.add_argument("--target-glbs", "--max-glbs", dest="target_glbs", type=int, default=None,
                   help="Stop once this many usable GLB files exist or have been created successfully.")
    p.add_argument("--convert-log-name", default="lgm_convert_manifest.csv",
                   help="CSV log written under --ply-root with per-PLY convert status.")
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


def count_ply_vertices(path: str) -> Optional[int]:
    """Read only the PLY header and return the vertex count.

    This is much cheaper than GaussianRenderer.load_ply(), and lets us skip
    large Gaussian clouds before Converter(cfg).cuda() allocates GPU resources.
    """
    with open(path, "rb") as f:
        for raw in f:
            line = raw.decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex"):
                parts = line.split()
                if len(parts) >= 3:
                    return int(parts[2])
            if line == "end_header":
                break
    return None


def object_id_from_ply(ply_path: str, ply_root: str) -> tuple[str, str, str]:
    """Return (object_id, archive_name, item_name) from a PLY path.

    Expected layout: ply_root/archive_xxx/object_id.ply.
    For flatter layouts, object_id falls back to the file stem.
    """
    rel = os.path.relpath(ply_path, ply_root)
    rel_no_ext = os.path.splitext(rel)[0]
    parts = Path(rel_no_ext).parts
    if len(parts) >= 2:
        archive_name = parts[-2]
        item_name = parts[-1]
        object_id = f"{archive_name}/{item_name}"
    else:
        archive_name = ""
        item_name = parts[-1]
        object_id = item_name
    return object_id, archive_name, item_name


def load_object_list(path: Optional[str]) -> Optional[set[str]]:
    if path is None or str(path).strip().lower() in {"", "none", "null"}:
        return None

    allowed: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        sample = f.read(4096)
        f.seek(0)

        # CSV: supports object_id, archive_name/item_name, ply_path, glb_path.
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
            # Plain text: one object_id per line.
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


def write_convert_log(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["object_id", "archive_name", "item_name", "ply_path", "glb_path", "point_count", "status", "error"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_benchmark_list(path: str, rows: list[dict]) -> None:
    """Write only usable GLB rows. This is the fixed object set for fair eval."""
    usable_status = {"converted", "skipped_existing_glb"}
    usable = []
    seen = set()

    for r in rows:
        oid = str(r.get("object_id", "")).strip()
        glb_path = str(r.get("glb_path", "")).strip()
        if not oid or oid in seen:
            continue
        if r.get("status") in usable_status and glb_path and os.path.exists(glb_path):
            usable.append({
                "object_id": oid,
                "archive_name": r.get("archive_name", ""),
                "item_name": r.get("item_name", oid.split("/")[-1]),
                "ply_path": r.get("ply_path", ""),
                "glb_path": glb_path,
                "point_count": r.get("point_count", ""),
            })
            seen.add(oid)

    if not usable:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["object_id", "archive_name", "item_name", "ply_path", "glb_path", "point_count"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(usable)


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

    allowed = load_object_list(args.object_list)
    if allowed is not None:
        before = len(plys)
        plys = [p for p in plys if is_allowed_object(object_id_from_ply(p, args.ply_root)[0], allowed)]
        print(f"[INFO] object-list filter: {before} -> {len(plys)} PLY candidates")

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

    convert_log_path = os.path.join(args.ply_root, args.convert_log_name)
    benchmark_path = os.path.join(args.ply_root, args.benchmark_name)
    rows: list[dict] = []
    success_glbs = 0
    skipped_big = 0
    errors = 0

    print(f"[INFO] scanning {len(plys)} PLY candidates")
    if args.max_ply_points is not None:
        print(f"[INFO] skip PLY if point_count > {args.max_ply_points}")
    if args.target_glbs is not None:
        print(f"[INFO] stop when usable GLB count reaches {args.target_glbs}")

    for ply_path in tqdm(plys, desc="PLY -> GLB"):
        if args.target_glbs is not None and success_glbs >= args.target_glbs:
            print(f"[INFO] reached target_glbs={args.target_glbs}; stopping conversion loop.")
            break

        glb_path = os.path.splitext(ply_path)[0] + ".glb"
        object_id, archive_name, item_name = object_id_from_ply(ply_path, args.ply_root)
        point_count = count_ply_vertices(ply_path)

        if args.max_ply_points is not None and point_count is not None and point_count > args.max_ply_points:
            skipped_big += 1
            print(f"[SKIP_BIG] {ply_path} point_count={point_count} > {args.max_ply_points}")
            rows.append({
                "object_id": object_id,
                "archive_name": archive_name,
                "item_name": item_name,
                "ply_path": ply_path,
                "glb_path": glb_path,
                "point_count": point_count,
                "status": "skipped_too_many_points",
                "error": "",
            })
            write_convert_log(convert_log_path, rows)
            write_benchmark_list(benchmark_path, rows)
            continue

        if os.path.exists(glb_path) and not args.overwrite:
            success_glbs += 1
            print(f"[SKIP_EXISTING] {glb_path} ({success_glbs} usable GLB)")
            rows.append({
                "object_id": object_id,
                "archive_name": archive_name,
                "item_name": item_name,
                "ply_path": ply_path,
                "glb_path": glb_path,
                "point_count": point_count if point_count is not None else "",
                "status": "skipped_existing_glb",
                "error": "",
            })
            write_convert_log(convert_log_path, rows)
            write_benchmark_list(benchmark_path, rows)
            continue

        cfg.test_path = ply_path
        converter = None
        try:
            converter = Converter(cfg).cuda()
            converter.fit_nerf(iters=args.nerf_iters, resolution=args.nerf_resolution)
            converter.fit_mesh(iters=args.mesh_iters, resolution=args.mesh_resolution, decimate_target=args.mesh_decimate_target)
            converter.fit_mesh_uv(iters=args.uv_iters, resolution=args.mesh_resolution, texture_resolution=args.texture_resolution, padding=args.uv_padding)
            converter.export_mesh(glb_path)
            success_glbs += 1
            print(f"[OK] {glb_path} ({success_glbs} usable GLB)")
            rows.append({
                "object_id": object_id,
                "archive_name": archive_name,
                "item_name": item_name,
                "ply_path": ply_path,
                "glb_path": glb_path,
                "point_count": point_count if point_count is not None else "",
                "status": "converted",
                "error": "",
            })
        except Exception as exc:
            errors += 1
            print(f"[ERROR] {ply_path}: {repr(exc)}")
            rows.append({
                "object_id": object_id,
                "archive_name": archive_name,
                "item_name": item_name,
                "ply_path": ply_path,
                "glb_path": glb_path,
                "point_count": point_count if point_count is not None else "",
                "status": "error",
                "error": repr(exc),
            })
        finally:
            # Best effort cleanup between objects.
            try:
                del converter
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            write_convert_log(convert_log_path, rows)
            write_benchmark_list(benchmark_path, rows)

    print("[INFO] conversion summary:")
    print(f"  usable_glbs={success_glbs}")
    print(f"  skipped_too_many_points={skipped_big}")
    print(f"  errors={errors}")
    print(f"  log={convert_log_path}")
    print(f"  benchmark_object_list={benchmark_path}")

    if args.target_glbs is not None and success_glbs < args.target_glbs:
        raise RuntimeError(
            f"Only {success_glbs} usable GLB files were produced/found, "
            f"but --target-glbs={args.target_glbs}. Increase --max-objects, "
            "raise --max-ply-points, or inspect the convert log for errors."
        )


if __name__ == "__main__":
    main()
