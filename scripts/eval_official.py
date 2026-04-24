"""
Standalone KITTI Eigen-split evaluation following the "official / paper"
protocol used by BTS / AdaBins / NewCRFs / ZoeDepth / Depth Anything:

    1. Read RGB + GT at their **native resolution** (no crop).
    2. Run the model on the image with the **short side resized to
       --input_size (default 518)** and both sides rounded up to multiples
       of 14, preserving aspect ratio.
    3. Bilinearly resize the predicted depth back to the GT resolution.
    4. Cap predictions to [min_depth, max_depth] (default 0.001, 80 m).
    5. Apply the **Garg crop**
           y ∈ [0.40810811 H, 0.99189189 H]
           x ∈ [0.03594771 W, 0.96405229 W]
       to both prediction and GT before computing metrics.
    6. Mask out GT pixels outside (min_depth, max_depth).
    7. Report the 7 standard KITTI metrics.

Usage
-----
    # Baseline (raw pretrained, no fine-tuning)
    python scripts/eval_official.py --config configs/kitti.yaml \
        --metrics-json logs/official/baseline_official.json

    # Fine-tuned ckpt
    python scripts/eval_official.py --config configs/kitti.yaml \
        --checkpoint logs/.../epoch09-val_abs_rel0.0478.ckpt \
        --metrics-json logs/official/lr1e-5_best_official.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

# Ensure we can import from ../src when launched from project root
PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.datamodule import _parse_split_file, _resolve_depth_path  # noqa: E402
from src.model import VideoDepthLightningModule  # noqa: E402


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Garg crop ratios (standard in KITTI Eigen eval since Garg et al. ECCV'16)
GARG_Y0, GARG_Y1 = 0.40810811, 0.99189189
GARG_X0, GARG_X1 = 0.03594771, 0.96405229


# ----------------------------------------------------------------------
def round_to_14(x: int) -> int:
    return max(14, int(round(x / 14.0)) * 14)


def preprocess_rgb(img: Image.Image, input_size: int,
                   device: torch.device) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Resize short-side to `input_size`, round both sides to a multiple of 14,
    ImageNet-normalize. Returns (tensor, original_hw).
    """
    W, H = img.size
    if H <= W:
        new_h = input_size
        new_w = int(round(W * input_size / H))
    else:
        new_w = input_size
        new_h = int(round(H * input_size / W))
    new_h = round_to_14(new_h)
    new_w = round_to_14(new_w)

    img_r = img.resize((new_w, new_h), Image.BICUBIC)
    arr = np.asarray(img_r, dtype=np.float32) / 255.0  # HWC
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    t = (t - IMAGENET_MEAN) / IMAGENET_STD
    return t.to(device), (H, W)


def load_depth_gt(path: Path) -> np.ndarray:
    """KITTI depth: 16-bit PNG, meters = value / 256."""
    return np.asarray(Image.open(path), dtype=np.int32).astype(np.float32) / 256.0


def garg_mask(h: int, w: int) -> np.ndarray:
    y0, y1 = int(GARG_Y0 * h), int(GARG_Y1 * h)
    x0, x1 = int(GARG_X0 * w), int(GARG_X1 * w)
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def kitti_metrics(pred: np.ndarray, gt: np.ndarray,
                  mask: np.ndarray) -> Dict[str, float]:
    eps = 1e-7
    p = np.clip(pred[mask], eps, None)
    g = np.clip(gt[mask], eps, None)
    thresh = np.maximum(p / g, g / p)
    d1 = float((thresh < 1.25).mean())
    d2 = float((thresh < 1.25 ** 2).mean())
    d3 = float((thresh < 1.25 ** 3).mean())
    abs_rel = float(np.mean(np.abs(p - g) / g))
    sq_rel = float(np.mean(((p - g) ** 2) / g))
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    rmse_log = float(np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2)))
    return dict(abs_rel=abs_rel, sq_rel=sq_rel, rmse=rmse,
                rmse_log=rmse_log, d1=d1, d2=d2, d3=d3)


# ----------------------------------------------------------------------
@torch.no_grad()
def run_eval(cfg: dict, ckpt: Optional[str], input_size: int,
             min_depth: float, max_depth: float,
             split_name: str, device: torch.device,
             garg: bool = True,
             median_scale: bool = False,
             limit: int = 0) -> Dict[str, float]:
    # -------- model --------
    if ckpt:
        print(f"[eval] loading checkpoint: {ckpt}")
        module = VideoDepthLightningModule.load_from_checkpoint(
            ckpt, cfg=cfg, strict=False, map_location="cpu"
        )
    else:
        print("[eval] no checkpoint → using raw pretrained model (baseline)")
        module = VideoDepthLightningModule(cfg=cfg)
    module.to(device).eval()

    # -------- data list --------
    data_cfg = cfg["data"]
    data_root = Path(data_cfg["data_root"])
    split_file = Path(data_cfg["split_dir"]) / data_cfg.get(split_name, "eigen_test_files.txt")
    samples = _parse_split_file(split_file, allow_no_depth=False)
    if limit > 0:
        samples = samples[:limit]
    print(f"[eval] {len(samples)} samples from {split_file}")

    # -------- accumulate metrics --------
    agg = {k: 0.0 for k in ("abs_rel", "sq_rel", "rmse", "rmse_log",
                            "d1", "d2", "d3")}
    n_valid = 0
    n_seen = 0

    for rgb_rel, dep_rel, _focal in samples:
        n_seen += 1
        rgb_path = data_root / "raw" / rgb_rel
        if not rgb_path.is_file():
            continue
        dep_path = _resolve_depth_path(data_root, dep_rel)
        if dep_path is None:
            continue

        img = Image.open(rgb_path).convert("RGB")
        gt = load_depth_gt(dep_path)       # (H, W) metres, sparse

        Hg, Wg = gt.shape
        # RGB and GT typically share resolution in KITTI; if not, resize RGB.
        if img.size != (Wg, Hg):
            img = img.resize((Wg, Hg), Image.BICUBIC)

        x, _ = preprocess_rgb(img, input_size, device)
        pred = module(x).float()           # (1, H_in, W_in) — module resizes back already
        # `VideoDepthLightningModule.forward` already resizes pred back to input
        # H/W, but here input was resized; we need to go back to GT resolution.
        pred = F.interpolate(pred.unsqueeze(1), size=(Hg, Wg),
                             mode="bilinear", align_corners=False).squeeze(1)
        pred = pred.squeeze(0).cpu().numpy()   # (Hg, Wg)

        # mask: valid GT pixels
        mask = (gt > min_depth) & (gt < max_depth)
        if garg:
            mask &= garg_mask(Hg, Wg)
        if mask.sum() == 0:
            continue

        pred = np.clip(pred, min_depth, max_depth)

        if median_scale:
            # optional: scale-align prediction (for relative-depth baselines)
            scale = np.median(gt[mask]) / np.median(pred[mask])
            pred = pred * scale
            pred = np.clip(pred, min_depth, max_depth)

        m = kitti_metrics(pred, gt, mask)
        for k in agg:
            agg[k] += m[k]
        n_valid += 1

        if n_valid % 100 == 0:
            print(f"  processed {n_valid}/{len(samples)}  "
                  f"(running abs_rel≈{agg['abs_rel']/n_valid:.4f})")

    if n_valid == 0:
        raise RuntimeError("No samples evaluated — check paths.")
    final = {k: v / n_valid for k, v in agg.items()}
    final["_n_valid"] = n_valid
    final["_n_seen"] = n_seen
    final["_input_size"] = input_size
    final["_garg_crop"] = garg
    final["_median_scaled"] = median_scale
    final["_checkpoint"] = ckpt or "pretrained"
    return final


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default="",
                   help="leave empty to evaluate the raw pretrained model")
    p.add_argument("--split", type=str, default="test_split",
                   choices=["train_split", "val_split", "test_split"])
    p.add_argument("--input-size", type=int, default=518,
                   help="short-side resize target (official=518)")
    p.add_argument("--min-depth", type=float, default=1e-3)
    p.add_argument("--max-depth", type=float, default=80.0)
    p.add_argument("--no-garg", action="store_true",
                   help="disable Garg crop (default: on)")
    p.add_argument("--median-scale", action="store_true",
                   help="median-scale prediction to GT (for relative-depth)")
    p.add_argument("--metrics-json", type=str, default="",
                   help="where to save the resulting metrics JSON")
    p.add_argument("--limit", type=int, default=0,
                   help="if >0, evaluate only the first N samples (for smoke tests)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device={device}")

    res = run_eval(
        cfg=cfg,
        ckpt=args.checkpoint or None,
        input_size=args.input_size,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        split_name=args.split,
        device=device,
        garg=not args.no_garg,
        median_scale=args.median_scale,
        limit=args.limit,
    )

    print("\n================ KITTI (official Eigen protocol) ================")
    for k, arrow in [("abs_rel", "↓"), ("sq_rel", "↓"), ("rmse", "↓"),
                     ("rmse_log", "↓"), ("d1", "↑"), ("d2", "↑"), ("d3", "↑")]:
        print(f"  {k:<10s} {arrow}  {res[k]:.4f}")
    print(f"  (n={res['_n_valid']}, input_size={res['_input_size']}, "
          f"garg={res['_garg_crop']}, median_scaled={res['_median_scaled']})")
    print("=================================================================\n")

    if args.metrics_json:
        Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
        out = {f"test/{k}": v for k, v in res.items()
               if not k.startswith("_") and isinstance(v, (int, float))}
        out.update({k: v for k, v in res.items() if k.startswith("_")})
        with open(args.metrics_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[eval] wrote → {args.metrics_json}")


if __name__ == "__main__":
    main()
