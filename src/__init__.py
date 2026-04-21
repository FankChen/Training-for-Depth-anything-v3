"""Training package for Video Depth Anything V3 on KITTI (Eigen split)."""

from .losses import (
    ScaleInvariantLoss,
    L1DepthLoss,
    GradientMatchingLoss,
    CombinedDepthLoss,
)
from .datamodule import KITTIDepthDataset, KITTIDataModule
from .model import VideoDepthLightningModule, load_vda_model, MODEL_CONFIGS

__all__ = [
    "ScaleInvariantLoss",
    "L1DepthLoss",
    "GradientMatchingLoss",
    "CombinedDepthLoss",
    "KITTIDepthDataset",
    "KITTIDataModule",
    "VideoDepthLightningModule",
    "load_vda_model",
    "MODEL_CONFIGS",
]
