"""
Standalone evaluation script.

Example
-------
    python evaluate.py --config configs/kitti.yaml \
        --checkpoint logs/depth_anything_v3/kitti_eigen_vitl/checkpoints/last.ckpt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import yaml
import pytorch_lightning as pl

from src import KITTIDataModule, VideoDepthLightningModule


METRIC_DIRS = {
    "abs_rel": "↓",
    "sq_rel":  "↓",
    "rmse":    "↓",
    "rmse_log":"↓",
    "d1":      "↑",
    "d2":      "↑",
    "d3":      "↑",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a .ckpt file saved during training")
    p.add_argument("--split", type=str, default="test", choices=["val", "test"])
    return p.parse_args()


def main() -> Dict[str, float]:
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    pl.seed_everything(42, workers=True)

    data_cfg = cfg["data"]
    dm = KITTIDataModule(
        data_root=data_cfg["data_root"],
        split_dir=data_cfg["split_dir"],
        train_split=data_cfg.get("train_split", "eigen_train_files.txt"),
        val_split=data_cfg.get("val_split", "eigen_val_files.txt"),
        test_split=data_cfg.get("test_split", "eigen_test_files.txt"),
        input_height=int(data_cfg["input_height"]),
        input_width=int(data_cfg["input_width"]),
        batch_size=1,
        num_workers=int(data_cfg.get("num_workers", 4)),
        min_depth=float(cfg["model"].get("min_depth", 0.001)),
        max_depth=float(cfg["model"].get("max_depth", 80.0)),
    )

    print(f"[eval] Loading checkpoint: {args.checkpoint}")
    model = VideoDepthLightningModule.load_from_checkpoint(
        args.checkpoint, cfg=cfg, strict=False
    )

    hw = cfg["hardware"]
    trainer = pl.Trainer(
        accelerator=hw.get("accelerator", "gpu"),
        devices=hw.get("devices", 1),
        precision=hw.get("precision", "16-mixed"),
        logger=False,
    )

    if args.split == "val":
        results = trainer.validate(model, datamodule=dm)
    else:
        results = trainer.test(model, datamodule=dm)

    if not results:
        print("[eval] No results returned.")
        return {}
    res = results[0]

    # pretty-print the 7 standard metrics
    print("\n================ KITTI depth metrics ================")
    prefix = "val" if args.split == "val" else "test"
    out: Dict[str, float] = {}
    for k, arrow in METRIC_DIRS.items():
        key = f"{prefix}/{k}"
        if key in res:
            v = float(res[key])
            out[k] = v
            print(f"  {k:<10s} {arrow}  {v:.4f}")
    print("=====================================================\n")
    return out


if __name__ == "__main__":
    main()
