"""Diagnose the prediction vs GT value range mismatch."""
import sys, yaml, torch
import torch.nn.functional as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datamodule import KITTIDepthDataset
from src.model import load_vda_model

cfg = yaml.safe_load(open(ROOT / "configs" / "kitti.yaml"))

ds = KITTIDepthDataset(
    data_root=cfg["data"]["data_root"],
    split_file=str(Path(cfg["data"]["split_dir"]) / cfg["data"]["val_split"]),
    input_height=cfg["data"]["input_height"],
    input_width=cfg["data"]["input_width"],
    min_depth=0.001, max_depth=80.0, mode="val",
)

sample = ds[0]
gt = sample["depth"]
mask = sample["mask"]

print(f"GT stats (valid): min={gt[mask].min():.2f}  max={gt[mask].max():.2f}  "
      f"mean={gt[mask].mean():.2f}  median={gt[mask].median():.2f}")

H_MODEL, W_MODEL = 14*16, 14*16
img_small = F.interpolate(sample["image"].unsqueeze(0), size=(H_MODEL, W_MODEL),
                          mode="bilinear", align_corners=False)

for ckpt_name in ["video_depth_anything_vitl.pth", "metric_video_depth_anything_vitl.pth"]:
    print(f"\n{'='*60}")
    print(f"Checkpoint: {ckpt_name}")
    print(f"{'='*60}")

    test_cfg = dict(cfg["model"])
    test_cfg["pretrained_ckpt"] = str(
        Path("/home/izi2sgh/MYDATA/quanjie/liren/depth_baselines/Video-Depth-Anything/checkpoints") / ckpt_name
    )

    model, kind = load_vda_model(test_cfg, cfg["vda_repo_path"])
    model = model.eval()

    with torch.no_grad():
        x = img_small.unsqueeze(1)
        out = model(x)
        if out.dim() == 4:
            out = out[:, 0]
        if out.dim() == 4 and out.size(1) == 1:
            out = out.squeeze(1)

    raw = out[0].float()
    print(f"Raw output: min={raw.min():.6f}  max={raw.max():.6f}  "
          f"mean={raw.mean():.6f}  median={raw.median():.6f}  std={raw.std():.6f}")

    clamped = raw.clamp(0.001, 80.0)
    print(f"After clamp [0.001,80]: min={clamped.min():.4f}  max={clamped.max():.4f}  "
          f"mean={clamped.mean():.4f}")

    inv = 1.0 / raw.clamp(min=1e-6)
    print(f"1/output (inverse):     min={inv.min():.4f}  max={inv.max():.4f}  "
          f"mean={inv.mean():.4f}  median={inv.median():.4f}")

    pct_at_80 = (raw > 80.0).float().mean() * 100
    pct_below_1 = (raw < 1.0).float().mean() * 100
    print(f"Pct raw > 80: {pct_at_80:.1f}%   Pct raw < 1: {pct_below_1:.1f}%")

    del model
    torch.cuda.empty_cache()
