"""
LightningModule wrapping Video Depth Anything (V3) for KITTI metric-depth.

Model import
------------
The Video-Depth-Anything repository is **not** a pip package, so we prepend the
configured `vda_repo_path` to `sys.path` at runtime.  We first try:

    from video_depth_anything.video_depth import VideoDepthAnything

and fall back to Depth-Anything V2:

    from depth_anything_v2.dpt import DepthAnythingV2

The `VideoDepthAnything` forward signature expects a 5-D tensor
`(B, T, C, H, W)` and returns `(B, T, H, W)`.  For single-frame training we
feed `T = 1` and squeeze the result to `(B, H, W)`.  DepthAnythingV2 takes
`(B, C, H, W)` and returns `(B, H, W)` or `(B, 1, H, W)` — both are handled.

H and W must be multiples of 14 for DINOv2.  The forward() method resizes the
input to the nearest multiple of 14, runs the model, and resizes the output
back to the original resolution.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from .losses import CombinedDepthLoss


# ============================================================
# Video Depth Anything model configs (aligned with run.py)
# ============================================================
MODEL_CONFIGS: Dict[str, Dict] = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


# ------------------------------------------------------------------
# dynamic import
# ------------------------------------------------------------------
def _add_repo_to_path(vda_repo_path: str) -> None:
    repo = str(Path(vda_repo_path).expanduser().resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)


def load_vda_model(config: dict, vda_repo_path: str) -> Tuple[nn.Module, str]:
    """Instantiate the V3 model and (optionally) load pretrained weights.

    Returns (model, kind) where kind is 'video_depth_anything' or
    'depth_anything_v2'.
    """
    _add_repo_to_path(vda_repo_path)

    encoder = config["encoder"]
    if encoder not in MODEL_CONFIGS:
        raise ValueError(f"Unknown encoder '{encoder}'")
    cfg = dict(MODEL_CONFIGS[encoder])
    metric = bool(config.get("metric_depth", True))

    kind: str
    try:
        from video_depth_anything.video_depth import VideoDepthAnything  # type: ignore

        # The constructor signature (per run.py):
        #   VideoDepthAnything(encoder=..., features=..., out_channels=..., metric=bool)
        try:
            model = VideoDepthAnything(**cfg, metric=metric)
        except TypeError:
            # older revisions without `metric` kwarg
            model = VideoDepthAnything(**cfg)
        kind = "video_depth_anything"
    except Exception as e_video:
        print(f"[model] VideoDepthAnything import failed ({e_video}); "
              f"falling back to DepthAnythingV2.")
        try:
            from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore
            v2_cfg = dict(cfg)
            if metric:
                v2_cfg["max_depth"] = float(config.get("max_depth", 80.0))
            model = DepthAnythingV2(**v2_cfg)
            kind = "depth_anything_v2"
        except Exception as e_v2:
            raise ImportError(
                "Neither video_depth_anything.video_depth.VideoDepthAnything "
                "nor depth_anything_v2.dpt.DepthAnythingV2 could be imported. "
                f"video error: {e_video}; v2 error: {e_v2}"
            )

    # -------------------- load pretrained weights --------------------
    ckpt = config.get("pretrained_ckpt", None)
    if ckpt:
        ckpt_p = Path(ckpt).expanduser()
        if ckpt_p.is_file():
            state = torch.load(str(ckpt_p), map_location="cpu")
            if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
                state = state["model"]
            if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
                state = state["state_dict"]
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"[model] Loaded '{ckpt_p.name}':  "
                  f"missing={len(missing)}  unexpected={len(unexpected)}")
            if missing:
                print(f"[model] first missing: {missing[:5]}")
            if unexpected:
                print(f"[model] first unexpected: {unexpected[:5]}")
        else:
            print(f"[model] WARNING: pretrained_ckpt '{ckpt_p}' not found, "
                  "starting from random init.")

    return model, kind


# ============================================================
# Lightning module
# ============================================================
class VideoDepthLightningModule(pl.LightningModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        model_cfg = cfg["model"]
        self.min_depth: float = float(model_cfg.get("min_depth", 1e-3))
        self.max_depth: float = float(model_cfg.get("max_depth", 80.0))

        # build model
        self.model, self.model_kind = load_vda_model(
            model_cfg, cfg["vda_repo_path"]
        )

        # loss
        loss_cfg = cfg["loss"]
        self.criterion = CombinedDepthLoss(
            silog_weight=loss_cfg.get("silog_weight", 1.0),
            l1_weight=loss_cfg.get("l1_weight", 0.1),
            grad_weight=loss_cfg.get("grad_weight", 0.5),
            silog_variance_focus=loss_cfg.get("silog_variance_focus", 0.85),
        )

        # freeze
        self._unfreeze_epoch = int(model_cfg.get("freeze_backbone_until_epoch", -1))
        if bool(model_cfg.get("freeze_backbone", False)):
            self._freeze_backbone()

        # For flagging once we've unfrozen.
        self._backbone_unfrozen = not bool(model_cfg.get("freeze_backbone", False))

    # --------------------------------------------------------------
    # freeze helpers
    # --------------------------------------------------------------
    _BACKBONE_KEYWORDS = ("pretrained", "encoder", "patch_embed", "blocks")

    def _is_backbone_param(self, name: str) -> bool:
        lname = name.lower()
        return any(kw in lname for kw in self._BACKBONE_KEYWORDS)

    def _freeze_backbone(self) -> None:
        n_frozen = 0
        for name, p in self.model.named_parameters():
            if self._is_backbone_param(name):
                p.requires_grad = False
                n_frozen += 1
        print(f"[model] Froze {n_frozen} backbone parameters.")

    def _unfreeze_backbone(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = True
        print("[model] Unfroze all parameters.")

    def on_train_epoch_start(self) -> None:
        if (
            not self._backbone_unfrozen
            and self._unfreeze_epoch >= 0
            and self.current_epoch >= self._unfreeze_epoch
        ):
            self._unfreeze_backbone()
            self._backbone_unfrozen = True

    # --------------------------------------------------------------
    # forward
    # --------------------------------------------------------------
    @staticmethod
    def _round_to_multiple(x: int, m: int = 14) -> int:
        return max(m, int(round(x / m)) * m)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, H, W) -> depth: (B, H, W) in metres."""
        B, C, H, W = images.shape
        Hp = self._round_to_multiple(H, 14)
        Wp = self._round_to_multiple(W, 14)

        if Hp != H or Wp != W:
            x = F.interpolate(
                images, size=(Hp, Wp), mode="bilinear", align_corners=False
            )
        else:
            x = images

        if self.model_kind == "video_depth_anything":
            # expects (B, T, C, H, W)
            x5 = x.unsqueeze(1)
            out = self.model(x5)  # (B, T, H, W)
            # handle either (B,T,H,W) or (B,T,1,H,W)
            if out.dim() == 5 and out.size(2) == 1:
                out = out.squeeze(2)
            if out.dim() == 4:
                # (B, T, H, W) -> take T=0
                out = out[:, 0]
        else:
            # DepthAnythingV2: (B, C, H, W) -> (B, 1, H, W) or (B, H, W)
            out = self.model(x)
            if out.dim() == 4 and out.size(1) == 1:
                out = out.squeeze(1)

        # resize back to original H, W
        if out.shape[-2:] != (H, W):
            out = F.interpolate(
                out.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
            ).squeeze(1)

        return out  # (B, H, W)

    # --------------------------------------------------------------
    # steps
    # --------------------------------------------------------------
    def _shared_step(self, batch: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images = batch["image"]           # (B, 3, H, W)
        gt = batch["depth"]               # (B, H, W)
        mask = batch["mask"]              # (B, H, W)

        pred = self(images)               # (B, H, W)

        # resize pred to match gt if needed (should already match).
        if pred.shape[-2:] != gt.shape[-2:]:
            pred = F.interpolate(
                pred.unsqueeze(1),
                size=gt.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        pred = pred.clamp(min=self.min_depth, max=self.max_depth)
        return pred, gt, mask

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        pred, gt, mask = self._shared_step(batch)
        losses = self.criterion(pred, gt, mask)

        bs = batch["image"].size(0)
        self.log("train/loss", losses["total"], prog_bar=True, batch_size=bs)
        self.log("train/silog", losses["silog"], batch_size=bs)
        self.log("train/l1", losses["l1"], batch_size=bs)
        self.log("train/grad", losses["grad"], batch_size=bs)
        return losses["total"]

    # --------------------------------------------------------------
    # validation / test
    # --------------------------------------------------------------
    @staticmethod
    def _compute_metrics(
        pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Standard 7-metric KITTI depth evaluation (mask-aware)."""
        eps = 1e-7
        mask = mask.bool()
        if mask.sum() == 0:
            z = pred.new_zeros(())
            return {k: z for k in ("abs_rel", "sq_rel", "rmse", "rmse_log",
                                    "d1", "d2", "d3")}

        p = pred[mask].clamp(min=eps)
        g = gt[mask].clamp(min=eps)

        thresh = torch.max(p / g, g / p)
        d1 = (thresh < 1.25).float().mean()
        d2 = (thresh < 1.25 ** 2).float().mean()
        d3 = (thresh < 1.25 ** 3).float().mean()

        abs_rel = ((p - g).abs() / g).mean()
        sq_rel = (((p - g) ** 2) / g).mean()
        rmse = torch.sqrt(((p - g) ** 2).mean())
        rmse_log = torch.sqrt(((torch.log(p) - torch.log(g)) ** 2).mean())

        return {
            "abs_rel": abs_rel,
            "sq_rel": sq_rel,
            "rmse": rmse,
            "rmse_log": rmse_log,
            "d1": d1,
            "d2": d2,
            "d3": d3,
        }

    def _eval_step(self, batch: dict, prefix: str) -> Dict[str, torch.Tensor]:
        pred, gt, mask = self._shared_step(batch)
        losses = self.criterion(pred, gt, mask)
        metrics = self._compute_metrics(pred, gt, mask)

        bs = batch["image"].size(0)
        self.log(f"{prefix}/loss", losses["total"], prog_bar=(prefix == "val"),
                 batch_size=bs, sync_dist=True)
        for k, v in metrics.items():
            self.log(f"{prefix}/{k}", v,
                     prog_bar=(prefix == "val" and k == "abs_rel"),
                     batch_size=bs, sync_dist=True)
        return metrics

    def validation_step(self, batch: dict, batch_idx: int):
        return self._eval_step(batch, "val")

    def test_step(self, batch: dict, batch_idx: int):
        return self._eval_step(batch, "test")

    # --------------------------------------------------------------
    # optimizers
    # --------------------------------------------------------------
    def configure_optimizers(self):
        tr = self.cfg["training"]
        lr = float(tr["lr"])
        backbone_lr_factor = float(tr.get("backbone_lr_factor", 0.1))
        weight_decay = float(tr.get("weight_decay", 0.0))
        opt_name = str(tr.get("optimizer", "adamw")).lower()

        backbone_params: List[nn.Parameter] = []
        other_params: List[nn.Parameter] = []
        for name, p in self.model.named_parameters():
            if self._is_backbone_param(name):
                backbone_params.append(p)
            else:
                other_params.append(p)

        param_groups = [
            {"params": other_params, "lr": lr, "name": "decoder"},
            {"params": backbone_params, "lr": lr * backbone_lr_factor, "name": "backbone"},
        ]

        if opt_name == "adamw":
            optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
        elif opt_name == "adam":
            optimizer = torch.optim.Adam(param_groups, lr=lr, weight_decay=weight_decay)
        elif opt_name == "sgd":
            optimizer = torch.optim.SGD(
                param_groups, lr=lr, momentum=0.9, weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer '{opt_name}'")

        scheduler_name = str(tr.get("scheduler", "cosine")).lower()
        if scheduler_name == "none":
            return optimizer

        max_epochs = int(tr.get("max_epochs", 25))
        warmup_epochs = int(tr.get("warmup_epochs", 0))

        if warmup_epochs > 0:
            def warmup_lambda(epoch: int) -> float:
                # linear 0 -> 1 over warmup_epochs
                return (epoch + 1) / max(1, warmup_epochs)

            warmup = LambdaLR(optimizer, lr_lambda=warmup_lambda)
            cosine = CosineAnnealingLR(
                optimizer, T_max=max(1, max_epochs - warmup_epochs), eta_min=lr * 0.01
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = CosineAnnealingLR(
                optimizer, T_max=max(1, max_epochs), eta_min=lr * 0.01
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
