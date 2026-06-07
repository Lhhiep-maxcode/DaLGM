from __future__ import annotations

"""
Evaluate LGM converted meshes (.glb) with RGB, depth, and optional mesh metrics.

Typical use after converting each predicted Gaussian .ply to .glb with convert.py:

python eval_lgm_mesh.py \
  --data-path /kaggle/input/datasets/laihoanghiep/10k-dataset-9-views \
  --depth1-path /kaggle/input/datasets/laihoanghiep/10k-dataset-9-views-depth \
  --eval-path /kaggle/input/datasets/tdthanh/1k-objaverse-16-views-for-eval \
  --mesh-path /kaggle/working/workspace/lgm_converted_meshes \
  --outdir /kaggle/working/workspace/lgm_mesh_eval \
  --input-size 256 --output-size 512 --splat-size 128 \
  --num-views-input 4 --num-views-output 16 \
  --train-size 0.8 --test-size 0.1 --val-size 0.1 \
  --depth-source eval --depth-render-size 512

Notes:
- RGB metrics are computed on the 16 eval views, matching LGM eval.py.
- depth-source=input matches the original LGM depth metric setup: depth is evaluated on
  the 4 input views from depth1_path and resized to splat_size/depth_render_size.
- depth-source=eval evaluates depth on the 16 eval views, but requires eval depth files
  000.npy/.npz ... 015.npy/.npz under eval_path/.../depth or --eval-depth-path.
- Mesh metrics need --gt-mesh-path and compare predicted GLB to GT GLB by surface sampling.
"""

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import csv
import cv2
import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import trimesh
from scipy.spatial import cKDTree
from torchmetrics.image import StructuralSimilarityIndexMeasure
from tqdm import tqdm

import nvdiffrast.torch as dr
from kiui.cam import get_perspective, orbit_camera

try:
    import lpips as lpips_lib
except Exception:
    lpips_lib = None


INPUT_VIEW_IDS = [0, 16, 32, 48]
INPUT_CAMERA_PARAMS = [(0.0, 0.0), (0.0, 90.0), (0.0, 180.0), (0.0, 270.0)]
EVAL_CAMERA_PARAMS = [(30.0, 45.0 * i) for i in range(8)] + [(60.0, 45.0 * i) for i in range(8)]


@dataclass
class MeshEvalConfig:
    data_path: str
    depth1_path: Optional[str]
    eval_path: str
    mesh_path: str
    outdir: str = "/kaggle/working/workspace/lgm_mesh_eval"
    csv_name: str = "lgm_mesh_eval_results.csv"
    resume_csv: Optional[str] = None

    depth2_path: Optional[str] = None
    depth3_path: Optional[str] = None
    depth4_path: Optional[str] = None
    eval_depth_path: Optional[str] = None
    gt_mesh_path: Optional[str] = None

    train_size: float = 0.8
    test_size: float = 0.1
    val_size: float = 0.1
    object_start: Optional[int] = None
    object_end: Optional[int] = None
    max_objects: Optional[int] = None
    object_list: Optional[str] = None
    allow_missing_object_list: bool = False

    input_size: int = 256
    splat_size: int = 128
    output_size: int = 512
    depth_render_size: int = 512
    num_views_input: int = 4
    num_views_output: int = 16
    batch_size: int = 1
    input_view_preset: str = "auto"  # auto | 4 | 9
    pixel_align: bool = False  # kept for branch tracking; mesh eval itself renders GLB geometry

    fovy: float = 60.0
    cam_radius: float = 1.5
    znear: float = 0.5
    zfar: float = 2.5

    depth_source: str = "none"  # eval | input | none
    save_preview_every: int = 1
    preview_only: bool = False

    mesh_scale: float = 1.0
    mesh_rot_x_deg: float = 0.0
    mesh_rot_y_deg: float = 0.0
    mesh_rot_z_deg: float = 0.0
    mesh_translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    flip_uv_y: bool = False

    mesh_num_samples: int = 100_000
    mesh_sample_seed: int = 42
    mesh_fscore_thresholds: tuple[float, ...] = (0.1, 0.2, 0.5)

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def none_if_text(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    if str(x).strip().lower() in {"", "none", "null"}:
        return None
    return x


def parse_args() -> MeshEvalConfig:
    p = argparse.ArgumentParser(description="Evaluate LGM converted GLB meshes.")
    p.add_argument("--data-path", required=True)
    p.add_argument("--depth1-path", default=None)
    p.add_argument("--depth2-path", default=None)
    p.add_argument("--depth3-path", default=None)
    p.add_argument("--depth4-path", default=None)
    p.add_argument("--eval-path", required=True)
    p.add_argument("--eval-depth-path", default=None)
    p.add_argument("--mesh-path", required=True)
    p.add_argument("--gt-mesh-path", default=None)
    p.add_argument("--outdir", default="/kaggle/working/workspace/lgm_mesh_eval")
    p.add_argument("--csv-name", default="lgm_mesh_eval_results.csv")
    p.add_argument("--resume-csv", default=None)

    p.add_argument("--train-size", type=float, default=0.8)
    p.add_argument("--test-size", type=float, default=0.1)
    p.add_argument("--val-size", type=float, default=0.1)
    p.add_argument("--object-start", type=int, default=None)
    p.add_argument("--object-end", type=int, default=None)
    p.add_argument("--max-objects", type=int, default=None)
    p.add_argument("--object-list", default=None,
                   help="Optional CSV/text file containing fixed benchmark object IDs. Eval will run only these objects.")
    p.add_argument("--allow-missing-object-list", action="store_true",
                   help="If set, skip benchmark objects that have no predicted GLB. Default is strict: raise an error if any are missing.")

    p.add_argument("--input-size", type=int, default=256)
    p.add_argument("--splat-size", type=int, default=128)
    p.add_argument("--output-size", type=int, default=512)
    p.add_argument("--depth-render-size", type=int, default=512)
    p.add_argument("--num-views-input", type=int, default=4)
    p.add_argument("--num-views-output", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=1, help="DataLoader batch size for loading samples; meshes are rendered one by one because geometry is variable-size.")
    p.add_argument("--input-view-preset", choices=["auto", "4", "9"], default="auto", help="Input view layout for crop/depth compatibility. auto chooses from --num-views-input.")
    p.add_argument("--pixel-align", dest="pixel_align", action="store_true", help="Stored in summary for branch tracking; mesh eval uses GLB geometry.")
    p.add_argument("--no-pixel-align", dest="pixel_align", action="store_false")
    p.set_defaults(pixel_align=False)

    p.add_argument("--fovy", type=float, default=60.0)
    p.add_argument("--cam-radius", type=float, default=1.5)
    p.add_argument("--znear", type=float, default=0.5)
    p.add_argument("--zfar", type=float, default=2.5)
    p.add_argument("--depth-source", choices=["eval", "input", "none"], default="none")

    p.add_argument("--save-preview-every", type=int, default=1)
    p.add_argument("--preview-only", action="store_true")

    p.add_argument("--mesh-scale", type=float, default=1.0)
    p.add_argument("--mesh-rot-x-deg", type=float, default=0.0)
    p.add_argument("--mesh-rot-y-deg", type=float, default=0.0)
    p.add_argument("--mesh-rot-z-deg", type=float, default=0.0)
    p.add_argument("--mesh-translation", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    p.add_argument("--flip-uv-y", dest="flip_uv_y", action="store_true")
    p.add_argument("--no-flip-uv-y", dest="flip_uv_y", action="store_false")
    p.set_defaults(flip_uv_y=True)

    p.add_argument("--mesh-num-samples", type=int, default=100_000)
    p.add_argument("--mesh-sample-seed", type=int, default=42)
    p.add_argument("--mesh-fscore-thresholds", type=float, nargs="+", default=(0.1, 0.2, 0.5))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    a = p.parse_args()
    cfg = MeshEvalConfig(**vars(a))
    cfg.depth1_path = none_if_text(cfg.depth1_path)
    cfg.depth2_path = none_if_text(cfg.depth2_path)
    cfg.depth3_path = none_if_text(cfg.depth3_path)
    cfg.depth4_path = none_if_text(cfg.depth4_path)
    cfg.eval_depth_path = none_if_text(cfg.eval_depth_path)
    cfg.gt_mesh_path = none_if_text(cfg.gt_mesh_path)
    cfg.resume_csv = none_if_text(cfg.resume_csv)
    cfg.object_list = none_if_text(cfg.object_list)
    cfg.mesh_translation = tuple(float(x) for x in cfg.mesh_translation)
    cfg.mesh_fscore_thresholds = tuple(float(x) for x in cfg.mesh_fscore_thresholds)
    return cfg


def rotation_matrix_np(rx: float = 0, ry: float = 0, rz: float = 0) -> np.ndarray:
    rx, ry, rz = map(math.radians, [rx, ry, rz])
    rx_m = np.array([[1, 0, 0, 0], [0, math.cos(rx), -math.sin(rx), 0], [0, math.sin(rx), math.cos(rx), 0], [0, 0, 0, 1]], dtype=np.float32)
    ry_m = np.array([[math.cos(ry), 0, math.sin(ry), 0], [0, 1, 0, 0], [-math.sin(ry), 0, math.cos(ry), 0], [0, 0, 0, 1]], dtype=np.float32)
    rz_m = np.array([[math.cos(rz), -math.sin(rz), 0, 0], [math.sin(rz), math.cos(rz), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    return rz_m @ ry_m @ rx_m


def apply_mesh_transform(mesh: trimesh.Trimesh, cfg: MeshEvalConfig) -> trimesh.Trimesh:
    mesh = mesh.copy()
    trans_m = np.eye(4, dtype=np.float32)
    trans_m[:3, 3] = np.array(cfg.mesh_translation, dtype=np.float32)
    scale_m = np.eye(4, dtype=np.float32)
    scale_m[:3, :3] *= float(cfg.mesh_scale)
    rot_m = rotation_matrix_np(cfg.mesh_rot_x_deg, cfg.mesh_rot_y_deg, cfg.mesh_rot_z_deg)
    mesh.apply_transform(trans_m @ rot_m @ scale_m)
    return mesh


def load_rgba_cv2(path: str) -> torch.Tensor:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 3:
        alpha = np.ones(img.shape[:2], dtype=np.uint8) * 255
        img = np.concatenate([img, alpha[..., None]], axis=-1)
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).contiguous()  # BGRA


def load_depth_file(depth_dir: str, view_name: str) -> torch.Tensor:
    npz_path = os.path.join(depth_dir, f"{view_name}.npz")
    npy_path = os.path.join(depth_dir, f"{view_name}.npy")
    if os.path.exists(npz_path):
        f = np.load(npz_path)
        if "depth" in f:
            arr = f["depth"]
        elif "data" in f:
            arr = f["data"]
        else:
            raise KeyError(f"No 'depth' or 'data' key in {npz_path}. Keys={list(f.keys())}")
    elif os.path.exists(npy_path):
        arr = np.load(npy_path)
    else:
        raise FileNotFoundError(f"Depth not found: {depth_dir}/{view_name}.npz/.npy")
    arr = arr.astype(np.float32)
    if arr.ndim == 1:
        side = int(np.sqrt(arr.shape[0]))
        arr = arr.reshape(side, side)
    elif arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = arr[0]
    return torch.from_numpy(arr).unsqueeze(0)  # [1,H,W]


def find_nonzero_bbox(alpha_channel: np.ndarray):
    ys, xs = np.where(alpha_channel > 1e-6)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return ys.min(), ys.max(), xs.min(), xs.max()


def resize_chw(x: torch.Tensor, size: int, mode: str = "bilinear") -> torch.Tensor:
    if mode == "nearest":
        return F.interpolate(x.unsqueeze(0).float(), size=(size, size), mode=mode)[0]
    return F.interpolate(x.unsqueeze(0).float(), size=(size, size), mode=mode, align_corners=False)[0]


def make_pose(elev: float, azim: float, cfg: MeshEvalConfig, origin_elev: float = 0.0, origin_azim: float = 0.0) -> torch.Tensor:
    c2w = torch.from_numpy(
        orbit_camera(
            -(elev - origin_elev),
            (azim - origin_azim),
            radius=cfg.cam_radius,
            opengl=True,
        )
    ).float()
    c2w[:3, 3] *= cfg.cam_radius / 1.5
    return c2w


def get_input_camera_setup(cfg: MeshEvalConfig) -> tuple[list[int], list[tuple[float, float]]]:
    """Return the input view IDs and camera params for the 4-view or 9-view LGM branch."""
    preset = cfg.input_view_preset
    if preset == "auto":
        preset = "9" if int(cfg.num_views_input) == 9 else "4"
    if preset == "9":
        view_ids = [0, 8, 16, 24, 32, 40, 48, 56, 64]
        camera_params = [(0.0, 45.0 * i) for i in range(8)] + [(89.89, 180.0)]
    elif preset == "4":
        view_ids = [0, 16, 32, 48]
        camera_params = [(0.0, 0.0), (0.0, 90.0), (0.0, 180.0), (0.0, 270.0)]
    else:
        raise ValueError(f"Unsupported input_view_preset={cfg.input_view_preset}")
    if cfg.num_views_input > len(view_ids):
        raise ValueError(f"num_views_input={cfg.num_views_input} is larger than preset {preset} provides ({len(view_ids)} views).")
    return view_ids[: cfg.num_views_input], camera_params[: cfg.num_views_input]


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
                mesh_path = str(row.get("glb_path", "")).strip() or str(row.get("ply_path", "")).strip()

                if not oid and archive and item:
                    oid = f"{archive}/{item}"
                if not oid and mesh_path:
                    stem = os.path.splitext(os.path.basename(mesh_path))[0]
                    parent = os.path.basename(os.path.dirname(mesh_path))
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


def build_glb_index(root: Optional[str]) -> dict[str, str]:
    index: dict[str, str] = {}
    if root is None:
        return index
    for cur, dirs, files in os.walk(root):
        dirs.sort()
        for fname in sorted(files):
            # Predicted LGM Gaussian .ply files are not triangle meshes.
            # Only index GLB here so eval never accidentally loads a raw Gaussian PLY.
            if not fname.lower().endswith(".glb"):
                continue
            full = os.path.join(cur, fname)
            stem = os.path.splitext(fname)[0]
            parent = os.path.basename(cur)
            grand = os.path.basename(os.path.dirname(cur))

            keys = [stem, parent]
            if stem == "mesh":
                keys.append(parent)
                keys.append(f"{grand}/{parent}")
            else:
                keys.append(f"{grand}/{stem}")
                keys.append(f"{parent}/{stem}")

            for k in keys:
                if k and k not in index:
                    index[k] = full
    return index


class LGMMeshEvalDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: MeshEvalConfig):
        self.cfg = cfg
        for p in [cfg.data_path, cfg.eval_path, cfg.mesh_path]:
            if not os.path.isdir(p):
                raise FileNotFoundError(f"Directory not found: {p}")
        if cfg.gt_mesh_path is not None and not os.path.isdir(cfg.gt_mesh_path):
            raise FileNotFoundError(f"Directory not found: {cfg.gt_mesh_path}")

        depth_roots = [cfg.depth1_path, cfg.depth2_path, cfg.depth3_path, cfg.depth4_path]
        subfolders = []
        for root in depth_roots:
            if root is not None and os.path.isdir(root):
                subfolders.extend(
                    os.path.join(root, sub)
                    for sub in sorted(os.listdir(root))
                    if os.path.isdir(os.path.join(root, sub))
                )
        self.has_depth = bool(subfolders)

        items = []
        if self.has_depth:
            for sub in sorted(subfolders):
                archive = os.path.basename(sub)
                for item in sorted(os.listdir(sub)):
                    item_path = os.path.join(sub, item)
                    if os.path.isdir(os.path.join(item_path, "depth")):
                        items.append((archive, item, item_path))
        else:
            # Không có depth path → discover từ data_path
            for archive in sorted(os.listdir(cfg.data_path)):
                archive_path = os.path.join(cfg.data_path, archive)
                if not os.path.isdir(archive_path):
                    continue
                for item in sorted(os.listdir(archive_path)):
                    item_path = os.path.join(archive_path, item)
                    if os.path.isdir(os.path.join(item_path, "rgb")):
                        items.append((archive, item, item_path))

        if cfg.val_size > 0:
            items = items[-int(cfg.val_size * len(items)):]
        else:
            items = []
        if cfg.object_start is not None or cfg.object_end is not None:
            items = items[cfg.object_start: cfg.object_end]
        if cfg.max_objects is not None:
            items = items[: cfg.max_objects]

        allowed = load_object_list(cfg.object_list)
        if allowed is not None:
            before = len(items)
            items = [
                (archive, item, item_path)
                for archive, item, item_path in items
                if is_allowed_object(f"{archive}/{item}", allowed)
            ]
            print(f"[dataset] object-list filter: {before} -> {len(items)} objects")

        self.mesh_index = build_glb_index(cfg.mesh_path)
        self.gt_mesh_index = build_glb_index(cfg.gt_mesh_path)

        items = [
            (archive, item, item_path)
            for archive, item, item_path in items
            if f"{archive}/{item}" in self.mesh_index or item in self.mesh_index
        ]
        print(f"[INFO] After mesh filtering: {len(items)} objects remain.")
        self.items = items

        if allowed is not None and not cfg.allow_missing_object_list:
            missing = [
                f"{archive}/{item}"
                for archive, item, _ in items
                if f"{archive}/{item}" not in self.mesh_index and item not in self.mesh_index
            ]
            if missing:
                preview = ", ".join(missing[:10])
                raise FileNotFoundError(
                    f"{len(missing)} benchmark objects have no predicted GLB under {cfg.mesh_path}. "
                    f"First missing: {preview}. "
                    "This usually means export/convert failed for this checkpoint; rerun conversion or pass --allow-missing-object-list to skip them."
                )

        if allowed is not None and cfg.allow_missing_object_list:
            before = len(items)
            items = [
                (archive, item, item_path)
                for archive, item, item_path in items
                if f"{archive}/{item}" in self.mesh_index or item in self.mesh_index
            ]
            print(f"[dataset] existing-GLB filter: {before} -> {len(items)} objects")

        self.items = items
        print(f"[dataset] objects={len(self.items)}, predicted_meshes_indexed={len(self.mesh_index)}, gt_meshes_indexed={len(self.gt_mesh_index)}")

    def __len__(self):
        return len(self.items)

    def resolve_eval_item_path(self, archive_name: str, item_name: str) -> str:
        p = os.path.join(self.cfg.eval_path, archive_name, item_name)
        if os.path.isdir(os.path.join(p, "rgb")):
            return p
        p = os.path.join(self.cfg.eval_path, item_name)
        if os.path.isdir(os.path.join(p, "rgb")):
            return p
        for a in sorted(os.listdir(self.cfg.eval_path)):
            cand = os.path.join(self.cfg.eval_path, a, item_name)
            if os.path.isdir(os.path.join(cand, "rgb")):
                return cand
        raise FileNotFoundError(f"Cannot find eval views for {item_name} inside {self.cfg.eval_path}")

    def resolve_eval_depth_dir(self, archive_name: str, item_name: str, eval_item_path: str) -> str:
        if self.cfg.eval_depth_path is None:
            return os.path.join(eval_item_path, "depth")
        candidates = [
            os.path.join(self.cfg.eval_depth_path, archive_name, item_name, "depth"),
            os.path.join(self.cfg.eval_depth_path, item_name, "depth"),
        ]
        for a in sorted(os.listdir(self.cfg.eval_depth_path)):
            candidates.append(os.path.join(self.cfg.eval_depth_path, a, item_name, "depth"))
        for p in candidates:
            if os.path.isdir(p):
                return p
        raise FileNotFoundError(f"Cannot find eval depth for {archive_name}/{item_name} under {self.cfg.eval_depth_path}")

    def resolve_mesh(self, archive_name: str, item_name: str) -> str:
        object_id = f"{archive_name}/{item_name}"
        for k in [object_id, item_name]:
            if k in self.mesh_index:
                return self.mesh_index[k]
        raise FileNotFoundError(f"Cannot find predicted mesh for {object_id}. Indexed {len(self.mesh_index)} mesh files under {self.cfg.mesh_path}")

    def resolve_gt_mesh(self, archive_name: str, item_name: str) -> Optional[str]:
        if self.cfg.gt_mesh_path is None:
            return None
        object_id = f"{archive_name}/{item_name}"
        for k in [object_id, item_name]:
            if k in self.gt_mesh_index:
                return self.gt_mesh_index[k]
        raise FileNotFoundError(f"Cannot find GT mesh for {object_id}. Indexed {len(self.gt_mesh_index)} mesh files under {self.cfg.gt_mesh_path}")

    def __getitem__(self, idx):
        archive_name, item_name, item_depth_path = self.items[idx]
        item_path = os.path.join(self.cfg.data_path, archive_name, item_name)
        eval_item_path = self.resolve_eval_item_path(archive_name, item_name)
        input_depth_dir = os.path.join(item_depth_path, "depth")

        input_view_ids, input_camera_params = get_input_camera_setup(self.cfg)
        origin_elev, origin_azim = input_camera_params[0]
        input_rgba, input_depths, input_poses = [], [], []
        output_rgba, output_poses = [], []
        global_ymin, global_ymax = 1e9, -1
        global_xmin, global_xmax = 1e9, -1

        for view_id, (elev, azim) in zip(input_view_ids, input_camera_params):
            rgba = load_rgba_cv2(os.path.join(item_path, "rgb", f"{view_id:03d}.png"))
            depth = load_depth_file(input_depth_dir, f"{view_id:03d}") if self.has_depth else torch.zeros(1, rgba.shape[1], rgba.shape[2])
            c2w = make_pose(elev, azim, self.cfg, origin_elev, origin_azim)
            bbox = find_nonzero_bbox(rgba[3].numpy())
            if bbox is not None:
                ymin, ymax, xmin, xmax = bbox
                global_ymin, global_ymax = min(global_ymin, ymin), max(global_ymax, ymax)
                global_xmin, global_xmax = min(global_xmin, xmin), max(global_xmax, xmax)
            input_rgba.append(rgba)
            input_depths.append(depth)
            input_poses.append(c2w)

        for view_idx, (elev, azim) in enumerate(EVAL_CAMERA_PARAMS[: self.cfg.num_views_output]):
            rgba = load_rgba_cv2(os.path.join(eval_item_path, "rgb", f"{view_idx:03d}.png"))
            c2w = make_pose(elev, azim, self.cfg, origin_elev, origin_azim)
            bbox = find_nonzero_bbox(rgba[3].numpy())
            if bbox is not None:
                ymin, ymax, xmin, xmax = bbox
                global_ymin, global_ymax = min(global_ymin, ymin), max(global_ymax, ymax)
                global_xmin, global_xmax = min(global_xmin, xmin), max(global_xmax, xmax)
            output_rgba.append(rgba)
            output_poses.append(c2w)

        origin_size = input_rgba[0].shape[1]
        if global_ymax < 0 or global_xmax < 0:
            min_res = 0
        else:
            min_res = int(min(origin_size - global_ymax, global_ymin, origin_size - global_xmax, global_xmin))
            min_res = max(min_res, 0)

        def crop_img(x: torch.Tensor) -> torch.Tensor:
            if min_res == 0:
                return x
            s = x.shape[-1]
            return x[:, min_res:(s - min_res), min_res:(s - min_res)]

        images_out, masks_out = [], []
        for rgba in output_rgba:
            rgba = crop_img(rgba)
            mask = rgba[3:4]
            image = rgba[:3] * mask + (1 - mask)
            image = image[[2, 1, 0]].contiguous()  # BGR -> RGB
            images_out.append(resize_chw(image, self.cfg.output_size, "bilinear"))
            masks_out.append(resize_chw(mask, self.cfg.output_size, "bilinear"))

        result = {
            "archive_name": archive_name,
            "item_name": item_name,
            "object_id": f"{archive_name}/{item_name}",
            "mesh_file": self.resolve_mesh(archive_name, item_name),
            "gt_mesh_file": self.resolve_gt_mesh(archive_name, item_name),
            "images_output": torch.stack(images_out, 0),
            "masks_output": torch.stack(masks_out, 0),
            "cam_poses_output": torch.stack(output_poses, 0),
        }

        if self.has_depth and self.cfg.depth_source == "input":
            depth_imgs, depth_masks = [], []
            for rgba, depth in zip(input_rgba, input_depths):
                rgba = crop_img(rgba)
                depth = crop_img(depth)
                mask = rgba[3:4]
                depth_imgs.append(resize_chw(depth, self.cfg.depth_render_size, "nearest"))
                depth_masks.append(resize_chw(mask, self.cfg.depth_render_size, "bilinear"))
            result["gt_depths"] = torch.stack(depth_imgs, 0)
            result["depth_masks"] = torch.stack(depth_masks, 0)
            result["cam_poses_depth"] = torch.stack(input_poses, 0)
        elif self.has_depth and self.cfg.depth_source == "eval":
            eval_depth_dir = self.resolve_eval_depth_dir(archive_name, item_name, eval_item_path)
            depth_imgs, depth_masks = [], []
            for view_idx, rgba in enumerate(output_rgba[: self.cfg.num_views_output]):
                rgba = crop_img(rgba)
                mask = rgba[3:4]
                depth = load_depth_file(eval_depth_dir, f"{view_idx:03d}")
                depth = crop_img(depth)
                depth_imgs.append(resize_chw(depth, self.cfg.depth_render_size, "nearest"))
                depth_masks.append(resize_chw(mask, self.cfg.depth_render_size, "bilinear"))
            result["gt_depths"] = torch.stack(depth_imgs, 0)
            result["depth_masks"] = torch.stack(depth_masks, 0)
            result["cam_poses_depth"] = torch.stack(output_poses, 0)

        return result


def load_mesh_any(path: str) -> trimesh.Trimesh:
    asset = trimesh.load(path, force="scene", process=False)
    if isinstance(asset, trimesh.Trimesh):
        mesh = asset
    elif isinstance(asset, trimesh.Scene):
        geoms = [g for g in asset.geometry.values() if isinstance(g, trimesh.Trimesh) and len(g.vertices) and len(g.faces)]
        if len(geoms) == 0:
            raise ValueError(f"No mesh geometry in {path}")
        if len(geoms) == 1:
            mesh = geoms[0]
        else:
            # Texture may be lost when concatenating; LGM converted GLB is usually single-geometry.
            mesh = trimesh.util.concatenate(geoms)
    else:
        raise TypeError(f"Unsupported mesh type: {type(asset)}")
    mesh = mesh.copy()
    mesh.remove_unreferenced_vertices()
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"Empty mesh: {path}")
    return mesh


def get_texture_image(mesh: trimesh.Trimesh):
    material = getattr(mesh.visual, "material", None)
    if material is None:
        return None
    tex_img = getattr(material, "baseColorTexture", None)
    if tex_img is None:
        tex_img = getattr(material, "image", None)
    return tex_img


def mesh_to_render_tensors(mesh: trimesh.Trimesh, cfg: MeshEvalConfig):
    device = cfg.device
    verts = torch.as_tensor(np.asarray(mesh.vertices, dtype=np.float32), device=device)
    faces = torch.as_tensor(np.asarray(mesh.faces, dtype=np.int32), device=device)

    out = {"verts": verts, "faces": faces, "uv": None, "tex": None, "vcolor": None}
    if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
        tex_img = get_texture_image(mesh)
        if tex_img is not None:
            uv = np.asarray(mesh.visual.uv, dtype=np.float32).copy()
            if cfg.flip_uv_y:
                uv[:, 1] = 1.0 - uv[:, 1]
            tex = np.asarray(tex_img.convert("RGB"), dtype=np.float32) / 255.0
            out["uv"] = torch.as_tensor(uv, dtype=torch.float32, device=device)
            out["tex"] = torch.as_tensor(tex, dtype=torch.float32, device=device)
            return out

    try:
        vc = np.asarray(mesh.visual.to_color().vertex_colors[:, :3], dtype=np.float32) / 255.0
        if vc.shape[0] == verts.shape[0]:
            out["vcolor"] = torch.as_tensor(vc, dtype=torch.float32, device=device)
            return out
    except Exception:
        pass

    out["vcolor"] = torch.ones_like(verts, dtype=torch.float32, device=device) * 0.7
    return out


class NVDRMeshRenderer:
    def __init__(self, cfg: MeshEvalConfig):
        self.cfg = cfg
        self.device = cfg.device
        self.glctx = dr.RasterizeCudaContext() if cfg.device == "cuda" else dr.RasterizeGLContext()
        self.proj = torch.from_numpy(get_perspective(cfg.fovy)).float().to(cfg.device)

    @torch.no_grad()
    def render_one(self, tensors: dict, c2w: torch.Tensor, size: int):
        verts = tensors["verts"]
        faces = tensors["faces"]
        c2w = c2w.to(self.device).float()

        verts_h = F.pad(verts, pad=(0, 1), mode="constant", value=1.0)
        v_cam = verts_h @ torch.inverse(c2w).T
        v_clip = v_cam @ self.proj.T
        v_clip = v_clip.unsqueeze(0).contiguous()

        rast, rast_db = dr.rasterize(self.glctx, v_clip, faces, (size, size))
        alpha = torch.clamp(rast[..., -1:], 0, 1).contiguous()
        alpha = dr.antialias(alpha, rast, v_clip, faces).clamp(0, 1)

        # OpenGL camera looks along -Z, so use -z_cam as positive depth.
        depth_values = (-v_cam[:, 2:3]).contiguous()
        depth, _ = dr.interpolate(depth_values.unsqueeze(0), rast, faces)
        depth = depth * alpha

        if tensors["uv"] is not None and tensors["tex"] is not None:
            texc, texc_db = dr.interpolate(tensors["uv"].unsqueeze(0), rast, faces, rast_db=rast_db, diff_attrs="all")
            image = dr.texture(tensors["tex"].unsqueeze(0), texc, uv_da=texc_db)
        else:
            image, _ = dr.interpolate(tensors["vcolor"].unsqueeze(0), rast, faces)

        image = image.clamp(0, 1)
        image = image * alpha + (1.0 - alpha)
        rgb_chw = image[0].permute(2, 0, 1).contiguous()
        depth_chw = depth[0].permute(2, 0, 1).contiguous()
        alpha_chw = alpha[0].permute(2, 0, 1).contiguous()
        return rgb_chw, depth_chw, alpha_chw

    @torch.no_grad()
    def render_many(self, tensors: dict, poses: torch.Tensor, size: int):
        rgbs, depths, alphas = [], [], []
        for i in range(poses.shape[0]):
            rgb, depth, alpha = self.render_one(tensors, poses[i], size)
            rgbs.append(rgb)
            depths.append(depth)
            alphas.append(alpha)
        return torch.stack(rgbs, 0), torch.stack(depths, 0), torch.stack(alphas, 0)


def compute_rgb_metrics(pred_rgb: torch.Tensor, gt_rgb: torch.Tensor, lpips_metric, ssim_metric, device: str):
    pred_rgb = pred_rgb.to(device).float().clamp(0, 1)
    gt_rgb = gt_rgb.to(device).float().clamp(0, 1)
    mse = F.mse_loss(pred_rgb, gt_rgb)
    psnr = -10.0 * torch.log10(mse + 1e-8)
    ssim = ssim_metric(pred_rgb, gt_rgb)
    if lpips_metric is None:
        lp = torch.tensor(float("nan"), device=device)
    else:
        pred_256 = F.interpolate(pred_rgb * 2 - 1, size=(256, 256), mode="bilinear", align_corners=False)
        gt_256 = F.interpolate(gt_rgb * 2 - 1, size=(256, 256), mode="bilinear", align_corners=False)
        lp = lpips_metric(pred_256, gt_256).mean()
    return {"psnr": psnr.detach(), "ssim": ssim.detach(), "lpips": lp.detach()}


def compute_depth_metrics(pred_depth: torch.Tensor, gt_depth: torch.Tensor, pred_alpha: torch.Tensor, gt_mask: torch.Tensor, device: str, min_valid: int = 10):
    pred_depth = pred_depth.to(device).float()
    gt_depth = gt_depth.to(device).float()
    pred_alpha = pred_alpha.to(device).float()
    gt_mask = gt_mask.to(device).float()
    abs_diff_list, abs_rel_list, sq_rel_list, delta_1_list = [], [], [], []
    for v in range(pred_depth.shape[0]):
        mask = (pred_alpha[v, 0] > 0.1) & (gt_mask[v, 0] > 0.01) & (gt_depth[v, 0] > 0.01) & (pred_depth[v, 0] > 0.0)
        if mask.sum() < min_valid:
            continue
        pred = pred_depth[v, 0][mask]
        gt = gt_depth[v, 0][mask]
        diff = pred - gt
        abs_diff_list.append(diff.abs().mean())
        abs_rel_list.append((diff.abs() / (gt + 1e-8)).mean())
        sq_rel_list.append(((diff ** 2) / (gt + 1e-8)).mean())
        thresh = torch.max(pred / (gt + 1e-8), gt / (pred + 1e-8))
        delta_1_list.append((thresh < 1.25).float().mean())
    if not abs_diff_list:
        z = torch.tensor(float("nan"), device=device)
        return {"abs_diff": z, "abs_rel": z, "sq_rel": z, "delta_1": z}
    return {
        "abs_diff": torch.stack(abs_diff_list).mean().detach(),
        "abs_rel": torch.stack(abs_rel_list).mean().detach(),
        "sq_rel": torch.stack(sq_rel_list).mean().detach(),
        "delta_1": torch.stack(delta_1_list).mean().detach(),
    }


def mesh_fscore_key(threshold: float) -> str:
    text = f"{float(threshold):g}".replace("-", "m").replace(".", "_")
    return f"fscore_{text}"


def sample_surface_points(mesh: trimesh.Trimesh, count: int, seed: Optional[int] = None) -> np.ndarray:
    if seed is not None:
        state = np.random.get_state()
        np.random.seed(int(seed))
    try:
        points, _ = trimesh.sample.sample_surface(mesh, int(count))
    finally:
        if seed is not None:
            np.random.set_state(state)
    return np.asarray(points, dtype=np.float32)

def normalize_mesh_to_unit_bbox(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    v = np.asarray(mesh.vertices, dtype=np.float32)
    vmin = v.min(axis=0)
    vmax = v.max(axis=0)
    center = (vmin + vmax) * 0.5
    scale = float((vmax - vmin).max())
    if scale > 1e-8:
        mesh.vertices = (v - center) * (2.0 / scale)
    return mesh

def compute_mesh_metrics(pred_mesh: trimesh.Trimesh, gt_mesh_path: str, cfg: MeshEvalConfig):
    gt_mesh = load_mesh_any(gt_mesh_path)

    pred_mesh = normalize_mesh_to_unit_bbox(pred_mesh)
    gt_mesh = normalize_mesh_to_unit_bbox(gt_mesh)
    
    pred_points = sample_surface_points(pred_mesh, cfg.mesh_num_samples, cfg.mesh_sample_seed)
    gt_points = sample_surface_points(gt_mesh, cfg.mesh_num_samples, None if cfg.mesh_sample_seed is None else cfg.mesh_sample_seed + 1)
    tree_gt = cKDTree(gt_points)
    tree_pred = cKDTree(pred_points)
    try:
        pred_to_gt, _ = tree_gt.query(pred_points, k=1, workers=-1)
        gt_to_pred, _ = tree_pred.query(gt_points, k=1, workers=-1)
    except TypeError:
        pred_to_gt, _ = tree_gt.query(pred_points, k=1)
        gt_to_pred, _ = tree_pred.query(gt_points, k=1)
    pred_to_gt = np.asarray(pred_to_gt, dtype=np.float32)
    gt_to_pred = np.asarray(gt_to_pred, dtype=np.float32)
    out = {"cd": float(pred_to_gt.mean() + gt_to_pred.mean())}
    for t in cfg.mesh_fscore_thresholds:
        precision = float((pred_to_gt < t).mean())
        recall = float((gt_to_pred < t).mean())
        f = 0.0 if precision + recall <= 0 else 2.0 * precision * recall / (precision + recall)
        out[mesh_fscore_key(t)] = float(f)
    return out


def save_rgb_preview(path: str, gt: torch.Tensor, pred: torch.Tensor, max_views: int = 16):
    gt_np = gt[:max_views].detach().cpu().numpy().transpose(0, 2, 3, 1)
    pr_np = pred[:max_views].detach().cpu().numpy().transpose(0, 2, 3, 1)
    n = min(len(gt_np), len(pr_np), max_views)
    canvas = np.concatenate([np.concatenate(gt_np[:n], axis=1), np.concatenate(pr_np[:n], axis=1)], axis=0)
    imageio.imwrite(path, (np.clip(canvas, 0, 1) * 255).astype(np.uint8))


def depth_vis(depth: np.ndarray, alpha: Optional[np.ndarray] = None) -> np.ndarray:
    d = depth.copy()
    if alpha is not None:
        d[alpha <= 0.1] = np.nan
    valid = np.isfinite(d) & (d > 0)
    out = np.ones((*d.shape, 3), dtype=np.float32)
    if valid.sum() == 0:
        return out
    lo, hi = np.percentile(d[valid], [2, 98])
    x = np.clip((d - lo) / (hi - lo + 1e-8), 0, 1)
    try:
        import matplotlib.pyplot as plt
        out = plt.get_cmap("viridis")(x)[..., :3].astype(np.float32)
        out[~valid] = 1.0
    except Exception:
        out = np.repeat(x[..., None], 3, axis=-1).astype(np.float32)
        out[~valid] = 1.0
    return out


def save_depth_preview(path: str, gt_depth: torch.Tensor, pred_depth: torch.Tensor, pred_alpha: torch.Tensor, max_views: int = 16):
    gt = gt_depth[:max_views, 0].detach().cpu().numpy()
    pr = pred_depth[:max_views, 0].detach().cpu().numpy()
    al = pred_alpha[:max_views, 0].detach().cpu().numpy()
    n = min(len(gt), len(pr), max_views)
    gt_imgs = [depth_vis(gt[i]) for i in range(n)]
    pr_imgs = [depth_vis(pr[i], al[i]) for i in range(n)]
    canvas = np.concatenate([np.concatenate(gt_imgs, axis=1), np.concatenate(pr_imgs, axis=1)], axis=0)
    imageio.imwrite(path, (np.clip(canvas, 0, 1) * 255).astype(np.uint8))


def base_row(cfg: MeshEvalConfig, idx: int, object_id: str):
    row = {
        "idx": idx,
        "object_id": object_id,
        "mesh_file": "",
        "psnr": np.nan,
        "ssim": np.nan,
        "lpips": np.nan,
        "abs_diff": np.nan,
        "abs_rel": np.nan,
        "sq_rel": np.nan,
        "delta_1": np.nan,
        "time_sec": np.nan,
        "error": "",
    }
    for t in cfg.mesh_fscore_thresholds:
        row[mesh_fscore_key(t)] = np.nan
    row["cd"] = np.nan
    return row


def write_summary(rows: list[dict], cfg: MeshEvalConfig):
    df = pd.DataFrame(rows)
    ok = df[df["error"].fillna("").eq("")].copy() if len(df) else df
    summary = {"num_objects_total": int(len(df)), "num_objects_ok": int(len(ok)), "config": asdict(cfg)}
    metric_cols = ["psnr", "ssim", "lpips", "abs_diff", "abs_rel", "sq_rel", "delta_1", "cd"] + [mesh_fscore_key(t) for t in cfg.mesh_fscore_thresholds]
    for col in metric_cols:
        if len(ok) and col in ok.columns:
            val = ok[col].mean()
            summary[col] = None if pd.isna(val) else float(val)
        else:
            summary[col] = None
    path = os.path.join(cfg.outdir, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print("Saved summary:", path)


def complete_rows(prev_df: pd.DataFrame, cfg: MeshEvalConfig) -> pd.DataFrame:
    if len(prev_df) == 0:
        return prev_df
    done = prev_df["object_id"].notna() & (~prev_df["object_id"].astype(str).str.strip().eq(""))
    done &= prev_df.get("error", "").fillna("").eq("")
    required = ["psnr", "ssim", "lpips"]
    if cfg.depth_source != "none":
        required += ["abs_diff", "abs_rel", "sq_rel", "delta_1"]
    if cfg.gt_mesh_path is not None:
        required += ["cd"] + [mesh_fscore_key(t) for t in cfg.mesh_fscore_thresholds]
    for col in required:
        if col not in prev_df.columns:
            done &= False
        else:
            done &= prev_df[col].notna()
    return prev_df.loc[done].copy()


def run_one(sample: dict, renderer: NVDRMeshRenderer, cfg: MeshEvalConfig, idx: int, lpips_metric, ssim_metric):
    row = base_row(cfg, idx, sample["object_id"])
    row["mesh_file"] = sample["mesh_file"]
    start = time.time()

    pred_mesh = apply_mesh_transform(load_mesh_any(sample["mesh_file"]), cfg)
    tensors = mesh_to_render_tensors(pred_mesh, cfg)

    pred_rgb, _, _ = renderer.render_many(tensors, sample["cam_poses_output"], cfg.output_size)
    gt_rgb = sample["images_output"].to(cfg.device)
    gt_mask = sample["masks_output"].to(cfg.device)
    gt_rgb = gt_rgb * gt_mask + (1.0 - gt_mask)
    rgb_m = compute_rgb_metrics(pred_rgb, gt_rgb, lpips_metric, ssim_metric, cfg.device)

    row.update({
        "psnr": float(rgb_m["psnr"].detach().cpu()),
        "ssim": float(rgb_m["ssim"].detach().cpu()),
        "lpips": float(rgb_m["lpips"].detach().cpu()),
    })

    if cfg.depth_source != "none" and "gt_depths" in sample:
        _, pred_depth, pred_alpha = renderer.render_many(tensors, sample["cam_poses_depth"], cfg.depth_render_size)
        gt_depth = sample["gt_depths"].to(cfg.device)
        depth_mask = sample["depth_masks"].to(cfg.device)
        depth_m = compute_depth_metrics(pred_depth, gt_depth, pred_alpha, depth_mask, cfg.device)
        row.update({k: float(v.detach().cpu()) for k, v in depth_m.items()})
    else:
        pred_depth = pred_alpha = gt_depth = None

    if sample.get("gt_mesh_file") is not None:
        mesh_m = compute_mesh_metrics(pred_mesh, sample["gt_mesh_file"], cfg)
        row.update(mesh_m)

    row["time_sec"] = time.time() - start

    obj_dir = os.path.join(cfg.outdir, *sample["object_id"].split("/"))
    os.makedirs(obj_dir, exist_ok=True)
    if cfg.save_preview_every > 0 and idx % cfg.save_preview_every == 0:
        save_rgb_preview(os.path.join(obj_dir, "preview_rgb.png"), gt_rgb, pred_rgb, max_views=min(16, gt_rgb.shape[0]))
        if cfg.depth_source != "none" and pred_depth is not None:
            save_depth_preview(os.path.join(obj_dir, "preview_depth.png"), gt_depth, pred_depth, pred_alpha, max_views=min(16, gt_depth.shape[0]))

    return row


def main():
    cfg = parse_args()
    os.makedirs(cfg.outdir, exist_ok=True)
    print("Config:")
    print(json.dumps(asdict(cfg), indent=2))
    print("Device:", cfg.device)

    torch.set_grad_enabled(False)
    dataset = LGMMeshEvalDataset(cfg)
    if len(dataset) == 0:
        raise RuntimeError("No objects found. Check depth paths and split settings.")

    renderer = NVDRMeshRenderer(cfg)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(cfg.device)
    if lpips_lib is not None:
        lpips_metric = lpips_lib.LPIPS(net="vgg").to(cfg.device).eval()
    else:
        print("[WARN] package 'lpips' is not installed; lpips will be NaN.")
        lpips_metric = None

    csv_out = os.path.join(cfg.outdir, cfg.csv_name)
    resume_csv = cfg.resume_csv or (csv_out if os.path.exists(csv_out) else None)
    if resume_csv is not None and os.path.exists(resume_csv):
        prev = pd.read_csv(resume_csv)
        kept = complete_rows(prev, cfg)
        rows = kept.to_dict(orient="records")
        done_ids = set(kept["object_id"].astype(str)) if len(kept) else set()
        print(f"Resuming from {resume_csv}: {len(done_ids)} complete objects skipped; {len(prev) - len(kept)} old/incomplete rows recomputed.")
    else:
        rows = []
        done_ids = set()

    max_iter = 1 if cfg.preview_only else len(dataset)
    subset = torch.utils.data.Subset(dataset, list(range(max_iter)))
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: batch,
    )

    global_idx = 0
    for batch in tqdm(loader, desc="LGM mesh eval"):
        for sample in batch:
            idx = global_idx
            global_idx += 1
            object_id = sample["object_id"]
            if object_id in done_ids:
                print("[SKIP]", object_id)
                continue
            try:
                row = run_one(sample, renderer, cfg, idx, lpips_metric, ssim_metric)
            except Exception as exc:
                row = base_row(cfg, idx, object_id)
                row["error"] = repr(exc)
                print("[ERROR]", object_id, row["error"])
            rows.append(row)
            done_ids.add(object_id)
            pd.DataFrame(rows).to_csv(csv_out, index=False)
            if cfg.device == "cuda":
                torch.cuda.empty_cache()

    print("Saved CSV:", csv_out)
    write_summary(rows, cfg)


if __name__ == "__main__":
    main()
