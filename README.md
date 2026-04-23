# Training-for-Depth-Anything-V3

PyTorch Lightning training framework for fine-tuning **Video Depth Anything**
(a.k.a. Depth Anything V3) on **KITTI metric depth** (Eigen split).

```
training-for-depth-anything-v3/
├── train.py
├── evaluate.py
├── configs/
│   └── kitti.yaml
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── datamodule.py
│   └── losses.py
├── requirements.txt
└── README.md
```

---

## 1. Install

```bash
conda create -n vda_v3 python=3.10 -y
conda activate vda_v3
pip install -r requirements.txt
# and an appropriate torch build for your CUDA version, e.g.
# pip install torch==2.1.2+cu121 torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 2. Clone the Video-Depth-Anything repo

The model architecture is **not** re-implemented here — we import it directly
from the official repository.

```bash
git clone https://github.com/DepthAnything/Video-Depth-Anything.git
cd Video-Depth-Anything
bash get_weights.sh       # downloads video_depth_anything_{vits,vitb,vitl}.pth
```

Then edit `configs/kitti.yaml`:

```yaml
vda_repo_path: /abs/path/to/Video-Depth-Anything
model:
  pretrained_ckpt: /abs/path/to/Video-Depth-Anything/checkpoints/video_depth_anything_vitl.pth
```

## 3. Prepare KITTI

Expected layout:

```
{data_root}/
├── raw/                       # KITTI raw sequences
│   └── 2011_09_26/.../image_02/data/*.png
├── depth/
│   ├── train/…/proj_depth/groundtruth/image_02/*.png
│   └── val/…
└── splits/
    ├── eigen_train_files.txt
    ├── eigen_val_files.txt
    └── eigen_test_files.txt
```

The split files are the ones from BTS:
<https://github.com/cleinc/bts/tree/master/train_test_inputs>.
Each line is `rgb_rel_path depth_rel_path focal_length`.

Set `data.data_root` and `data.split_dir` in `configs/kitti.yaml`.

## 4. Train

```bash
python train.py --config configs/kitti.yaml
# override on the CLI:
python train.py --config configs/kitti.yaml --batch_size 8 --lr 5e-5 --devices 0,1
# resume:
python train.py --config configs/kitti.yaml --resume logs/.../last.ckpt
```

TensorBoard:

```bash
tensorboard --logdir logs/
```

## 5. Evaluate

```bash
python evaluate.py --config configs/kitti.yaml \
    --checkpoint logs/depth_anything_v3/kitti_eigen_vitl/checkpoints/last.ckpt
```

Prints the 7 standard KITTI depth metrics:

| metric   | direction |
|----------|-----------|
| abs_rel  | ↓         |
| sq_rel   | ↓         |
| rmse     | ↓         |
| rmse_log | ↓         |
| d1/d2/d3 | ↑         |

---

## Notes

* **Backbone freezing.** `model.freeze_backbone: true` freezes all parameters
  whose name contains `pretrained`, `encoder`, `patch_embed` or `blocks`. They
  are re-enabled at `freeze_backbone_until_epoch` (set to `-1` to keep them
  frozen forever).
* **DINOv2 patch constraint.** DINOv2 requires `H % 14 == 0` and `W % 14 == 0`.
  The Lightning module resizes any input to the nearest multiple of 14
  internally and resizes the prediction back, so `input_height: 352` /
  `input_width: 1216` (standard KITTI crop) work transparently.
* **Forward-shape compatibility.** `VideoDepthLightningModule.forward` handles
  both `VideoDepthAnything` (expects `(B,T,C,H,W)`, returns `(B,T,H,W)`) and
  the `DepthAnythingV2` fallback (expects `(B,C,H,W)`, returns `(B,H,W)` or
  `(B,1,H,W)`).
* **Mixed precision.** `hardware.precision: 16-mixed` is on by default.
* **Sparse GT.** KITTI depth is sparse LiDAR; all losses and metrics use the
  validity mask `(min_depth < d < max_depth)`.
