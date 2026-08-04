# Training: ST-GCN & EfficientGCN on MediaPipe keypoints

This folder is a self-contained snapshot of the scripts used to train two
skeleton-based action-recognition models **from scratch** on MediaPipe/
BlazePose keypoints extracted by this repo (`../pose_detection.py`):

- **ST-GCN** (Yan et al., AAAI-18) — model code reused from `mmskeleton`
- **EfficientGCN-B0** (Song et al., T-PAMI 2022) — model code reused from `effgcn_cam`

Both scripts bypass those repos' own mmcv/thop/pynvml-heavy training
frameworks and import only their pure-PyTorch model definitions directly from
source files. Training runs on CPU or CUDA (auto-detected, or forced with
`--device`). Extracted keypoints, built datasets, and trained checkpoints are
**not** included here; everything below regenerates them.

## Files

| File | Purpose |
|---|---|
| `build_dataset.py` | Converts HumanPoseAction keypoint exports into a train/val skeleton dataset |
| `mp_to_mmskeleton.py` | Joint-layout conversion: MediaPipe's 33 landmarks → openpose-18 / coco-17 / ntu-rgb+d-25 |
| `labels.py`, `kinetics_label_name.txt` | Label-name lookups (for `--from-ntu-filename` and inference reporting) |
| `train_stgcn.py` | Train ST-GCN from scratch |
| `train_effgcn.py` | Train EfficientGCN-B0 from scratch |
| `stgcn_infer.py` | Shared mmskeleton-loading utilities + standalone single-clip inference |
| `requirements.txt` | Python dependencies for everything in this folder |

## Prerequisites

1. **Python env**:

   ```bash
   pip install -r requirements.txt
   ```

   Installs `torch==2.2.2` + `numpy==1.26.4` (**numpy must stay <2** — the
   mmskeleton code paths break on numpy 2). The plain PyPI `torch` build has
   CUDA support built in: all three scripts (`train_stgcn.py`,
   `train_effgcn.py`, `stgcn_infer.py`) take `--device auto|cpu|cuda`
   (`auto`, the default, uses a GPU if one is available and falls back to
   CPU otherwise — no separate GPU/CPU setup needed). See the comments in
   `requirements.txt` if you need to pin a specific CUDA toolkit version.

   Extraction (Step 1) needs its own separate deps (`mediapipe`,
   `opencv-python`, `scipy`) — see `../requirements.txt`; not required just
   to run training on keypoints you already have.

2. **Two external repos**, cloned as siblings of the `HumanPoseAction/` repo
   itself (two levels up from this file, *not* inside `HumanPoseAction/`):

   ```
   workspace/
   ├── HumanPoseAction/
   │   └── train/       <- you are here
   ├── mmskeleton/       <- clone here
   └── effgcn_cam/       <- clone here
   ```

   ```bash
   git clone https://github.com/open-mmlab/mmskeleton ../../mmskeleton
   git clone https://github.com/attention-eq-everything/effgcn_cam.git ../../effgcn_cam
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
# ST-GCN (--device defaults to auto: uses a GPU if available, else CPU)
python train_stgcn.py <dataset_dir> --epochs 60 --work-dir work_dir_stgcn

# EfficientGCN-B0
python train_effgcn.py <dataset_dir> --epochs 60 --work-dir work_dir_eff

# force a specific device on either script:
python train_stgcn.py <dataset_dir> --device cuda ...
python train_stgcn.py <dataset_dir> --device cpu ...
```

Both write `best.pth` / `latest.pth` and a `history.json` (per-epoch train
loss, val top-1, val top-5) into `--work-dir`. Run `--help` on either script
for the full flag list (window size, LR schedule, dropout, batch size, ...).

## Step 4 — evaluate a checkpoint

```bash
python train_stgcn.py <dataset_dir> --evaluate work_dir_stgcn/best.pth
python train_effgcn.py <dataset_dir> --evaluate work_dir_eff/best.pth
```

## How the mmskeleton loading trick works (`train_stgcn.py`, lines ~54-75)

mmskeleton is an old research repo: importing it normally (`import mmskeleton`)
walks through its package `__init__.py` files, which pull in `mmcv`/`mmdet`/
`pycocotools` — a stack that doesn't install cleanly on a modern Python. But
the two pieces `train_stgcn.py` actually needs, `SkeletonLoader` (dataset
loading) and the preprocessing functions in `skeleton_process.py`
(`normalize_by_resolution`, `mask_by_visibility`, `pad_zero`, `random_crop`,
`simulate_camera_moving`), are themselves pure numpy/torch — it's only the
*package `__init__.py` files* standing in the way, not the code itself. So
instead of `import mmskeleton...`, that block:

1. **Loads each file directly from disk** with `importlib`, via
   `stgcn_infer._load_module_from_file()` (`stgcn_infer.py` does the same
   trick for the `ST_GCN_18` model itself) — this executes the file as a
   module without ever running the package `__init__.py` that would trigger
   the mmcv-chain import.
2. **Stubs out fake empty packages in `sys.modules`** for
   `mmskeleton`, `mmskeleton.deprecated`, `mmskeleton.deprecated.datasets`,
   `mmskeleton.deprecated.datasets.utils` *before* loading
   `skeleton_process.py`. That file contains a real
   `from mmskeleton.deprecated.datasets.utils import skeleton` — normally
   this would either fail (no such importable package here) or cascade into
   the same heavy imports. Registering hollow placeholder modules under
   those exact dotted names first means Python's import system finds
   *something* already there and resolves the reference immediately.
3. **Reattaches the real loaded module onto the stub**
   (`sys.modules["mmskeleton.deprecated.datasets.utils"].skeleton = _aaai18_utils`)
   so that when `skeleton_process.py` does `skeleton.some_function(...)` at
   call time, it's actually hitting the real file loaded in step 1 — the stub
   package is just scaffolding to satisfy the import statement, not a dead
   end.

Net effect: the exact same `SkeletonLoader` and preprocessing code mmskeleton
ships is reused byte-for-byte, with zero code copied or reimplemented, and
without installing `mmcv`/`mmdet`/`pycocotools` at all.

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
