# Training: ST-GCN & EfficientGCN on MediaPipe keypoints

This folder is a self-contained snapshot of the scripts used to train two
skeleton-based action-recognition models **from scratch** on MediaPipe/
BlazePose keypoints extracted by this repo (`../pose_detection.py`):

- **ST-GCN** (Yan et al., AAAI-18) — model code reused from `mmskeleton`
- **EfficientGCN-B0** (Song et al., T-PAMI 2022) — model code reused from `effgcn_cam`

Both scripts bypass those repos' own CUDA/mmcv-heavy training frameworks and
import only their pure-PyTorch model definitions directly from source files —
runs fine on CPU. Extracted keypoints, built datasets, and trained checkpoints
are **not** included here; everything below regenerates them.

## Files

| File | Purpose |
|---|---|
| `build_dataset.py` | Converts HumanPoseAction keypoint exports into a train/val skeleton dataset |
| `mp_to_mmskeleton.py` | Joint-layout conversion: MediaPipe's 33 landmarks → openpose-18 / coco-17 / ntu-rgb+d-25 |
| `labels.py`, `kinetics_label_name.txt` | Label-name lookups (for `--from-ntu-filename` and inference reporting) |
| `train_stgcn.py` | Train ST-GCN from scratch |
| `train_effgcn.py` | Train EfficientGCN-B0 from scratch |
| `stgcn_infer.py` | Shared mmskeleton-loading utilities + standalone single-clip inference |

## Prerequisites

1. **Python env**: `torch==2.2.2` (CPU build is fine), `numpy==1.26.4`
   (**numpy must stay <2** — the mmskeleton code paths break on numpy 2),
   plus whatever `../pose_detection.py` needs (`mediapipe`, `opencv-python`,
   `scipy` — see `../requirements.txt`).

2. **Two external repos**, cloned as siblings of this `train/` folder:

   ```
   HumanPoseAction/
   ├── train/          <- you are here
   ├── mmskeleton/      <- clone here
   └── effgcn_cam/      <- clone here
   ```

   ```bash
   git clone https://github.com/open-mmlab/mmskeleton ../mmskeleton
   git clone <effgcn_cam repo url> ../effgcn_cam
   ```

   Only used as a source of pure-PyTorch model code (`ST_GCN_18`, `Graph`,
   `SkeletonLoader`, `EfficientGCN`) — nothing from their own training
   frameworks is imported. If you keep clones elsewhere instead, point at
   them with environment variables:

   ```bash
   export MMSKELETON_ROOT=/path/to/mmskeleton
   export EFFGCN_ROOT=/path/to/effgcn_cam
   ```

## Step 1 — extract keypoints

Run this repo's own pose extraction on your source videos (see the top-level
`../README.md` / `pose_detection.py`). Output: one JSON per clip, e.g. in
`../output/`.

## Step 2 — build a train/val dataset

```bash
python build_dataset.py <keypoints_dir> <dataset_dir> \
    --layout coco --from-ntu-filename --val-split 0.2
```

- `--layout coco` (17 joints) for ST-GCN, `--layout ntu-rgb+d` (25 joints)
  for EfficientGCN — EfficientGCN's graph only ships an `ntu` variant.
- `--from-ntu-filename` derives the action label from NTU-style filenames
  (`...A0xx...`); use `--annotation <file.json>` instead for a custom
  category-annotation JSON if your clips aren't NTU-named.
- `--val-split` holds out **whole subjects**, not random clips, so
  validation never leaks near-duplicate takes of the same person.
- `--only-actions 8 9 23 27 43` restricts to a class subset (remapped to a
  contiguous `0..N-1` range); `--max-per-class N` caps clips/class for
  controlled dataset-size comparisons.

## Step 3 — train

```bash
# ST-GCN
python train_stgcn.py <dataset_dir> --epochs 60 --work-dir work_dir_stgcn

# EfficientGCN-B0
python train_effgcn.py <dataset_dir> --epochs 60 --device cpu --work-dir work_dir_eff
```

Both write `best.pth` / `latest.pth` and a `history.json` (per-epoch train
loss, val top-1, val top-5) into `--work-dir`. Run `--help` on either script
for the full flag list (window size, LR schedule, dropout, batch size, ...).

## Step 4 — evaluate a checkpoint

```bash
python train_stgcn.py <dataset_dir> --evaluate work_dir_stgcn/best.pth
python train_effgcn.py <dataset_dir> --evaluate work_dir_eff/best.pth
```

## What to expect

EfficientGCN-B0 (0.32M params) reaches ~90%+ val top-1 on a 20-class,
~2,700-clip MediaPipe dataset within ~30 epochs. ST-GCN (3M params — the
original AAAI-18 design, sized for the 40k+-clip Kinect datasets it was
published on) overfits hard at this data scale: train loss falls toward
zero while val accuracy plateaus around 30–35%, regardless of dropout or
weight decay. Treat ST-GCN results from this pipeline as a baseline/
comparison point, not a tuned model — closing that gap needs either a much
larger training set or a smaller ST-GCN variant, not more epochs.

## Not included here

- Extracted keypoints / built datasets — regenerate via Steps 1–2
- Trained checkpoints — regenerate via Step 3
- The `mmskeleton` / `effgcn_cam` source repos — clone separately, see Prerequisites
