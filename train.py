"""
Training entry-point for Video Depth Anything V3 on KITTI (Eigen split).

Usage
-----
    python train.py --config configs/kitti.yaml
    python train.py --config configs/kitti.yaml --batch_size 8 --lr 5e-5
    python train.py --config configs/kitti.yaml --resume path/to/last.ckpt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger

from src import KITTIDataModule, VideoDepthLightningModule


# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Video Depth Anything V3 on KITTI")
    p.add_argument("--config", type=str, required=True,
                   help="Path to a YAML config file (e.g. configs/kitti.yaml)")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a .ckpt file to resume training from")
    p.add_argument("--devices", type=str, default=None,
                   help="Override hardware.devices (e.g. '1' or '0,1')")
    p.add_argument("--batch_size", type=int, default=None,
                   help="Override training.batch_size")
    p.add_argument("--lr", type=float, default=None,
                   help="Override training.lr")
    p.add_argument("--max_epochs", type=int, default=None,
                   help="Override training.max_epochs")
    p.add_argument("--experiment_name", type=str, default=None,
                   help="Override logging.experiment_name (sub-folder under logs/<project>/)")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.max_epochs is not None:
        cfg["training"]["max_epochs"] = args.max_epochs
    if args.experiment_name is not None:
        cfg["logging"]["experiment_name"] = args.experiment_name
    if args.devices is not None:
        # allow '1' -> int, or '0,1' -> list[int]
        d = args.devices
        if "," in d:
            cfg["hardware"]["devices"] = [int(x) for x in d.split(",") if x]
        else:
            try:
                cfg["hardware"]["devices"] = int(d)
            except ValueError:
                cfg["hardware"]["devices"] = d
    return cfg


# ----------------------------------------------------------------------
def build_callbacks(cfg: dict, ckpt_dir: Path):
    log_cfg = cfg["logging"]
    monitor = log_cfg.get("monitor_metric", "val/abs_rel")
    mode = log_cfg.get("monitor_mode", "min")

    monitor_safe = monitor.replace("/", "_")
    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="epoch{epoch:02d}-" + monitor_safe + "{" + monitor + ":.4f}",
        auto_insert_metric_name=False,
        monitor=monitor,
        mode=mode,
        save_top_k=int(log_cfg.get("save_top_k", 3)),
        save_last=True,
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    prog_cb = RichProgressBar()
    return [ckpt_cb, lr_cb, prog_cb]


def build_trainer(cfg: dict, callbacks, logger) -> pl.Trainer:
    tr = cfg["training"]
    hw = cfg["hardware"]
    return pl.Trainer(
        max_epochs=int(tr["max_epochs"]),
        accelerator=hw.get("accelerator", "gpu"),
        devices=hw.get("devices", 1),
        precision=hw.get("precision", "16-mixed"),
        strategy=hw.get("strategy", "auto"),
        gradient_clip_val=float(tr.get("gradient_clip_val", 0.0)) or None,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=20,
        benchmark=True,
    )


# ----------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)

    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("high")

    log_cfg = cfg["logging"]
    log_dir = Path(log_cfg.get("log_dir", "logs"))
    exp_name = log_cfg.get("experiment_name", "kitti")
    proj_name = log_cfg.get("project_name", "depth_anything_v3")

    logger = TensorBoardLogger(save_dir=str(log_dir), name=proj_name, version=exp_name)
    ckpt_dir = Path(logger.log_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- data + model
    data_cfg = cfg["data"]
    dm = KITTIDataModule(
        data_root=data_cfg["data_root"],
        split_dir=data_cfg["split_dir"],
        train_split=data_cfg.get("train_split", "eigen_train_files.txt"),
        val_split=data_cfg.get("val_split", "eigen_val_files.txt"),
        test_split=data_cfg.get("test_split", "eigen_test_files.txt"),
        input_height=int(data_cfg["input_height"]),
        input_width=int(data_cfg["input_width"]),
        batch_size=int(cfg["training"]["batch_size"]),
        num_workers=int(data_cfg.get("num_workers", 8)),
        min_depth=float(cfg["model"].get("min_depth", 0.001)),
        max_depth=float(cfg["model"].get("max_depth", 80.0)),
    )

    model = VideoDepthLightningModule(cfg)

    # ---- callbacks + trainer
    callbacks = build_callbacks(cfg, ckpt_dir)
    trainer = build_trainer(cfg, callbacks, logger)

    trainer.fit(model, datamodule=dm, ckpt_path=args.resume)

    # final test using the best checkpoint tracked by ModelCheckpoint
    try:
        results = trainer.test(model, datamodule=dm, ckpt_path="best")
        # Dump to a JSON sidecar so scripts/record_run.py can pick it up.
        if results:
            import json
            out_json = Path(logger.log_dir) / "test_metrics.json"
            flat = {k: float(v) for k, v in results[0].items()}
            best_ckpt = ""
            for cb in callbacks:
                if isinstance(cb, ModelCheckpoint):
                    best_ckpt = cb.best_model_path or ""
                    break
            flat["_best_ckpt"] = best_ckpt
            flat["_log_dir"] = str(logger.log_dir)
            with out_json.open("w") as f:
                json.dump(flat, f, indent=2)
            print(f"[train] Wrote test metrics → {out_json}")
    except Exception as e:
        print(f"[train] Skipping final test: {e}")


if __name__ == "__main__":
    main()
