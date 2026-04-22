"""Quick smoke test: dataset sample + model forward with a tiny input.
Run from the project root:
    /home/izi2sgh/MYDATA/quanjie/liren/depth-v3/envs/depth-v3/bin/python scripts/smoke_test.py
"""
import sys, os, yaml, torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import KITTIDepthDataset, VideoDepthLightningModule

cfg_path = ROOT / "configs" / "kitti.yaml"
cfg = yaml.safe_load(open(cfg_path))

# ---- dataset: load 1 training sample ----
ds = KITTIDepthDataset(
    data_root=cfg["data"]["data_root"],
    split_file=str(Path(cfg["data"]["split_dir"]) / cfg["data"]["train_split"]),
    input_height=cfg["data"]["input_height"],
    input_width=cfg["data"]["input_width"],
    min_depth=cfg["model"]["min_depth"],
    max_depth=cfg["model"]["max_depth"],
    mode="train",
)
print(f"[dataset] len={len(ds)}")
sample = ds[0]
print(f"[dataset] image={tuple(sample['image'].shape)} "
      f"depth={tuple(sample['depth'].shape)} "
      f"mask-valid={int(sample['mask'].sum())}/{sample['mask'].numel()} "
      f"depth-range=[{float(sample['depth'][sample['mask']].min()):.2f},"
      f"{float(sample['depth'][sample['mask']].max()):.2f}] m")

# ---- model: load + small forward ----
print("\n[model] building...")
lm = VideoDepthLightningModule(cfg)
lm.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[model] moving to {device}")
lm = lm.to(device)
dtype = torch.float16 if device.type == "cuda" else torch.float32

# Feed a dummy 1x3x224x224 (multiple of 14) input.
x = torch.randn(1, 3, 224, 224, device=device, dtype=dtype)
if dtype == torch.float16:
    lm = lm.half()
with torch.no_grad():
    y = lm(x)
print(f"[model] forward OK: in={tuple(x.shape)}  out={tuple(y.shape)} "
      f"range=[{float(y.min()):.3f},{float(y.max()):.3f}]")
print("\nSMOKE TEST PASSED")
