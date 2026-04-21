"""
KITTI (Eigen split) depth DataModule for Video Depth Anything V3.

Split-file format (BTS):
    <rgb_rel_path> <depth_rel_path> <focal_length>
    e.g.
    2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000005.png \
        2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02/0000000005.png \
        721.5377

Some test-split lines may contain `None` for the depth path; those samples are
kept only when `allow_no_depth=True` (not used by train/val).

Directory layout expected under `data_root`:
    {data_root}/raw/<rgb_rel_path>
    {data_root}/depth/train/<depth_rel_path>   (tried first)
    {data_root}/depth/val/<depth_rel_path>     (fallback)
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF

import pytorch_lightning as pl


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _parse_split_file(split_path: Path, allow_no_depth: bool = False) -> List[Tuple[str, Optional[str], float]]:
    samples: List[Tuple[str, Optional[str], float]] = []
    with open(split_path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) < 3:
                continue
            rgb_rel, dep_rel, focal = parts[0], parts[1], float(parts[2])
            if dep_rel == "None" or dep_rel == "none":
                if not allow_no_depth:
                    continue
                dep_rel = None  # type: ignore[assignment]
            samples.append((rgb_rel, dep_rel, focal))
    return samples


def _resolve_depth_path(data_root: Path, dep_rel: str) -> Optional[Path]:
    """Try depth/train then depth/val. Return None if neither exists."""
    for sub in ("train", "val"):
        p = data_root / "depth" / sub / dep_rel
        if p.is_file():
            return p
    return None


# ----------------------------------------------------------------------
# dataset
# ----------------------------------------------------------------------
class KITTIDepthDataset(Dataset):
    """KITTI (Eigen) depth dataset.

    Returns a dict ``{"image": (3,H,W) float, "depth": (H,W) float,
                      "mask": (H,W) bool, "focal": float}``.
    """

    def __init__(
        self,
        data_root: str,
        split_file: str,
        input_height: int = 352,
        input_width: int = 1216,
        min_depth: float = 0.001,
        max_depth: float = 80.0,
        mode: str = "train",  # train | val | test
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.input_h = input_height
        self.input_w = input_width
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.mode = mode

        self.samples = _parse_split_file(Path(split_file), allow_no_depth=False)
        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples parsed from {split_file}")

        self.color_jitter = T.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        )
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------
    def _load_rgb(self, rgb_rel: str) -> Image.Image:
        path = self.data_root / "raw" / rgb_rel
        return Image.open(path).convert("RGB")

    def _load_depth(self, dep_rel: str) -> np.ndarray:
        path = _resolve_depth_path(self.data_root, dep_rel)
        if path is None:
            raise FileNotFoundError(
                f"Depth GT not found in {self.data_root}/depth/[train|val]/{dep_rel}"
            )
        # 16-bit PNG; depth_in_m = value / 256
        d = np.asarray(Image.open(path), dtype=np.int32).astype(np.float32) / 256.0
        return d

    # ------------------------------------------------------------------
    def _train_augment(
        self, img: Image.Image, depth: np.ndarray
    ) -> Tuple[Image.Image, np.ndarray]:
        W, H = img.size
        th, tw = self.input_h, self.input_w

        # If the image is smaller than the crop, resize up while preserving ratio.
        if H < th or W < tw:
            scale = max(th / H, tw / W)
            new_w, new_h = int(round(W * scale)) + 1, int(round(H * scale)) + 1
            img = img.resize((new_w, new_h), Image.BICUBIC)
            depth = np.array(
                Image.fromarray(depth).resize((new_w, new_h), Image.NEAREST),
                dtype=np.float32,
            )
            W, H = img.size

        # Random crop
        top = random.randint(0, H - th)
        left = random.randint(0, W - tw)
        img = img.crop((left, top, left + tw, top + th))
        depth = depth[top : top + th, left : left + tw]

        # Random horizontal flip
        if random.random() < 0.5:
            img = TF.hflip(img)
            depth = depth[:, ::-1].copy()

        # Colour jitter
        img = self.color_jitter(img)
        return img, depth

    def _val_crop(
        self, img: Image.Image, depth: np.ndarray
    ) -> Tuple[Image.Image, np.ndarray]:
        """KB-crop style: take the bottom-centre input_h x input_w region."""
        W, H = img.size
        th, tw = self.input_h, self.input_w

        if H < th or W < tw:
            scale = max(th / H, tw / W)
            new_w, new_h = int(round(W * scale)) + 1, int(round(H * scale)) + 1
            img = img.resize((new_w, new_h), Image.BICUBIC)
            depth = np.array(
                Image.fromarray(depth).resize((new_w, new_h), Image.NEAREST),
                dtype=np.float32,
            )
            W, H = img.size

        top = H - th
        left = (W - tw) // 2
        img = img.crop((left, top, left + tw, top + th))
        depth = depth[top : top + th, left : left + tw]
        return img, depth

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        rgb_rel, dep_rel, focal = self.samples[idx]
        img = self._load_rgb(rgb_rel)
        depth = self._load_depth(dep_rel)  # (H, W) float32, metres

        # the depth GT in KITTI is the same size as the RGB
        if depth.shape[0] != img.size[1] or depth.shape[1] != img.size[0]:
            # rare mismatch: resize depth to match RGB (nearest preserves holes).
            depth = np.array(
                Image.fromarray(depth).resize(img.size, Image.NEAREST),
                dtype=np.float32,
            )

        if self.mode == "train":
            img, depth = self._train_augment(img, depth)
        else:
            img, depth = self._val_crop(img, depth)

        # to tensor
        img_t = TF.to_tensor(img)            # (3, H, W)  in [0,1]
        img_t = self.normalize(img_t)
        depth_t = torch.from_numpy(np.ascontiguousarray(depth)).float()  # (H, W)

        mask = (depth_t > self.min_depth) & (depth_t < self.max_depth)

        return {
            "image": img_t,
            "depth": depth_t,
            "mask": mask,
            "focal": torch.tensor(focal, dtype=torch.float32),
        }


# ----------------------------------------------------------------------
# datamodule
# ----------------------------------------------------------------------
class KITTIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        split_dir: str,
        train_split: str = "eigen_train_files.txt",
        val_split: str = "eigen_val_files.txt",
        test_split: str = "eigen_test_files.txt",
        input_height: int = 352,
        input_width: int = 1216,
        batch_size: int = 4,
        num_workers: int = 8,
        min_depth: float = 0.001,
        max_depth: float = 80.0,
    ):
        super().__init__()
        self.save_hyperparameters()

    # ------------------------------------------------------------------
    def _make_ds(self, split_name: str, mode: str) -> KITTIDepthDataset:
        return KITTIDepthDataset(
            data_root=self.hparams.data_root,
            split_file=str(Path(self.hparams.split_dir) / split_name),
            input_height=self.hparams.input_height,
            input_width=self.hparams.input_width,
            min_depth=self.hparams.min_depth,
            max_depth=self.hparams.max_depth,
            mode=mode,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_ds = self._make_ds(self.hparams.train_split, "train")
            self.val_ds = self._make_ds(self.hparams.val_split, "val")
        if stage in (None, "validate"):
            if not hasattr(self, "val_ds"):
                self.val_ds = self._make_ds(self.hparams.val_split, "val")
        if stage in (None, "test"):
            self.test_ds = self._make_ds(self.hparams.test_split, "test")

    # ------------------------------------------------------------------
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )
