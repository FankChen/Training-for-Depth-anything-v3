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

---

# 中文说明（Bosch 集群实验记录）

## 一、本次任务的目的

在 Bosch 内部 LSF 集群（`batch_b200` / `batch_h200` 队列）上，用 **KITTI Eigen split** 对官方 **Video-Depth-Anything V3 (ViT-L, metric)** 预训练权重做 **微调 (fine-tuning)**，并系统性地对比：

1. **原始预训练模型**（不微调）在 KITTI test 上的指标（baseline）
2. **多个不同学习率**下的微调结果（因 DA3 预训练已经很好，重点扫描小 lr）
3. 把所有实验结果**自动汇总到仓库根目录的 `experiments.xlsx`**，每个实验一行，任务名称以 Bosch 项目号 **`BH-000425-08-05`** 开头

目标：找到最适合 KITTI 的微调学习率，量化"微调是否真的带来提升"。

---

## 二、仓库结构（中文注释）

```
training-for-depth-anything-v3/
├── configs/kitti.yaml              # 单一超参来源
├── src/                            # 模型 / 数据 / loss
│   ├── model.py                    # VideoDepthLightningModule
│   ├── datamodule.py               # KITTI BTS split 读取 + 增强
│   └── losses.py                   # SILog + L1 + Gradient
├── train.py                        # 训练入口（带 --lr / --max_epochs / --experiment_name 覆盖）
├── evaluate.py                     # 评估入口（--checkpoint 空则评估原始预训练模型）
├── splits/
│   ├── eigen_train_files.txt       # 23157 条，来自 BTS
│   └── eigen_test_files.txt        # 696 条，同时充当 val
├── kitti_root/                     # 本地软链接目录 (.gitignore)
├── scripts/
│   ├── train.bsub                  # 单次训练（首次运行用，固定到 H200 reservation）
│   ├── train_sweep.bsub            # 通用参数化模板，读 TAG/LR/MAX_EPOCHS
│   ├── submit_sweep.sh             # 一键提交多 lr sweep 到公共队列 batch_b200
│   ├── eval_baseline.bsub          # 原始预训练模型基线评估
│   ├── record_run.py               # 带 filelock 的 Excel 追加/更新工具
│   └── record_run1.sh              # 回填首次 lr=1e-4 运行
├── experiments.xlsx                # 所有实验的总表（跟着 git 一起维护）
└── requirements.txt
```

`experiments.xlsx` 的列（sheet 名 `runs`）：

```
run_name | project_code | stage | status | job_id | submitted_at | finished_at
encoder | freeze_backbone | freeze_until | lr | backbone_lr_factor | batch_size
max_epochs | precision | input_hw | dataset_split | ckpt
abs_rel | sq_rel | rmse | rmse_log | d1 | d2 | d3 | log_dir | notes
```

---

## 三、操作方法

### 3.1 环境

```bash
# 集群上已建好 project-local conda env
/home/izi2sgh/MYDATA/quanjie/liren/depth-v3/envs/depth-v3
# 关键依赖：torch 2.11+cu130, pytorch-lightning 2.6, openpyxl, filelock
```

### 3.2 评估原始预训练模型（baseline）

```bash
bsub < scripts/eval_baseline.bsub
# 自动写 logs/baseline/baseline_metrics.json 和 experiments.xlsx
```

### 3.3 提交学习率 sweep（公共队列 batch_b200，**不占用预留节点**）

默认扫 5 个 lr：`5e-5 3e-5 1e-5 5e-6 1e-6`，每个 25 epoch：

```bash
./scripts/submit_sweep.sh
```

自定义：

```bash
LRS="2e-5 7e-6"          ./scripts/submit_sweep.sh   # 只扫两个 lr
MAX_EPOCHS=10 LRS="3e-5"  ./scripts/submit_sweep.sh   # 跑得短一点做 pilot
```

⚠️ **师兄要求**：不要提交到 `rng-dl01-w24n01`（H200 预留节点，只有 4 张卡），因此 `scripts/train_sweep.bsub` 和 `scripts/eval_baseline.bsub` **都不带 `-U` 和 `-m`**，LSF 会自动分配公共 B200 / H200 卡。

### 3.4 每个任务里发生了什么

`scripts/train_sweep.bsub` 的流程：

1. 起跑时调 `record_run.py` 在 xlsx 写一行 `status=running`
2. 跑 `python train.py --lr $LR --max_epochs $N --experiment_name kitti_eigen_vitl_<tag>`
3. 训练完自动 `trainer.test(ckpt_path="best")`，写出 `logs/.../test_metrics.json`
4. 再调一次 `record_run.py --update`，把 7 个 KITTI 指标 + best checkpoint 路径 + `status=done/failed` 写回**同一行**

多任务并发写 xlsx 用 `filelock`（`experiments.xlsx.lock`）保护，安全。

### 3.5 手动记录（特殊情况）

首次 lr=1e-4 的训练是用旧 bsub 跑的（当时还没 xlsx 集成），跑完后用：

```bash
./scripts/record_run1.sh
```

一次性把结果回填进 xlsx。

### 3.6 查看结果

```bash
# 作业状态
bjobs -u $USER -w | grep VDA-V3-KITTI

# 某个任务实时日志
tail -f jobs/vda_v3_kitti.<JOBID>.stdout

# TensorBoard（本地 + 端口转发，或集群内）
tensorboard --logdir logs/depth_anything_v3/

# 看 Excel
python -c "from openpyxl import load_workbook; \
w=load_workbook('experiments.xlsx')['runs']; \
[print(list(r)) for r in w.iter_rows(values_only=True)]"
```

---

## 四、当前实验结果（截至 README 更新时刻）

| Run | lr | epoch | test/abs_rel ↓ | test/rmse ↓ | test/d1 ↑ | 说明 |
|---|---|---|---|---|---|---|
| **baseline** | — | 0 | **0.1302** | 4.11 | 0.8544 | 原始预训练，不微调 |
| lr=1e-4 (run1) | 1e-4 | 25 | **0.0503** | 2.26 | 0.9756 | 首次成功运行（H200 预留节点）|
| lr=5e-5 … 1e-6 | sweep | 进行中 | 待补 | — | — | 5 个任务在 batch_b200 跑 |

**结论（初步）**：微调相比预训练 abs_rel 下降约 **61%**（0.13 → 0.05）。

Sweep 跑完后，xlsx 会自动填满；我会用同样的流程 commit + push 更新本 README 的表格和 `experiments.xlsx`。

---

## 五、后续操作（TODO）

- [ ] 等所有 sweep 任务 `DONE` 后，检查每行 `experiments.xlsx` 是否都有 7 个指标；缺的用 `scripts/record_run.py --update` 补
- [ ] 在本 README 的"当前实验结果"表格里填入 sweep 最终数据，对每个 lr 写一句 takeaway
- [ ] 选出最佳 lr 后再跑一次 **更长 epoch** (50) 验证是否继续下降
- [ ] 考虑扩展实验维度：
  - `backbone_lr_factor` 扫描（当前固定 0.1）
  - 冻结策略：`freeze_backbone_until_epoch` = -1 / 0 / 5 / 10 对比
  - 输入分辨率：`350×1218` vs 更大 crop
  - 增加 NYU / DDAD 数据集验证泛化
- [ ] 所有 `.bsub` 的 `#BSUB -M`（内存）偏大，可以压到 20 GB 节省资源
- [ ] `train.py` 加 `--seed` 覆盖做不同种子 repeat

---

## 六、Git 协作约定

- `experiments.xlsx` **跟着 git 一起提交**（作为实验档案），但 `.lock` 文件忽略
- `kitti_root/`、`logs/`、`jobs/*.stdout` 都在 `.gitignore` 里
- commit message 建议格式：
  ```
  Run <tag>: abs_rel=X.XXXX (best at epoch Y) — <short note>
  ```
- 实验结束后请**手动撤销**提供过的 GitHub PAT（Settings → Developer settings → Tokens → Revoke）

