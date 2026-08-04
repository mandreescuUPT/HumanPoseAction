"""
train_stgcn.py
==============

Path B: train ST-GCN **from scratch on MediaPipe keypoints** -- no Kinect
domain gap, no legacy mmcv/mmdet stack, CPU-friendly.

This replaces mmskeleton's `processor.recognition.train` (which hard-requires
mmcv.runner + MMDataParallel + CUDA) with a plain PyTorch loop, while reusing
mmskeleton's own building blocks, loaded directly from its source files:

    ST_GCN_18            mmskeleton/models/backbones/st_gcn_aaai18.py
    SkeletonLoader       mmskeleton/datasets/skeleton/loader.py
    preprocessing        mmskeleton/datasets/skeleton/skeleton_process.py
                         (normalize_by_resolution, mask_by_visibility,
                          pad_zero, random_crop, simulate_camera_moving, ...)

The training recipe follows configs/recognition/st_gcn/humanpose_coco/train.yaml
(itself modeled on the kinetics-from-openpose config): coco-17 layout,
pad_zero(150) -> random_crop(150) -> simulate_camera_moving for train,
pad_zero(300) for val, SGD 0.1 with steps, but sized down for a small dataset.

Usage
-----
    # 1. build the dataset from extracted keypoints (see extract_all.py):
    python build_dataset.py ../HumanPoseAction/output ./data/humanpose_coco \
        --layout coco --from-ntu-filename --val-split 0.2

    # 2. train:
    python train_stgcn.py ./data/humanpose_coco --epochs 60

    # 3. evaluate a checkpoint on the val split:
    python train_stgcn.py ./data/humanpose_coco --evaluate work_dir/best.pth
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import stgcn_infer  # reuse the mmcv-free module loaders

# mmskeleton's loader.py / skeleton_process.py are pure numpy+torch but can't be
# imported through the package __init__ (it chains into pycocotools/mmcv), so
# load them file-directly, same as stgcn_infer does for the model.
_MMSK = Path(stgcn_infer._MMSKELETON_ROOT)
_loader_mod = stgcn_infer._load_module_from_file(
    "mmsk_skeleton_loader", str(_MMSK / "mmskeleton/datasets/skeleton/loader.py"))
SkeletonLoader = _loader_mod.SkeletonLoader

# skeleton_process imports one symbol from mmskeleton.deprecated...utils, which
# is itself pure -- pre-load it under the expected package names.
import types
for pkg in ("mmskeleton", "mmskeleton.deprecated", "mmskeleton.deprecated.datasets",
            "mmskeleton.deprecated.datasets.utils"):
    if pkg not in sys.modules:
        m = types.ModuleType(pkg)
        m.__path__ = []
        sys.modules[pkg] = m
_aaai18_utils = stgcn_infer._load_module_from_file(
    "mmskeleton.deprecated.datasets.utils.skeleton",
    str(_MMSK / "mmskeleton/deprecated/datasets/utils/skeleton.py"))
sys.modules["mmskeleton.deprecated.datasets.utils"].skeleton = _aaai18_utils
sp = stgcn_infer._load_module_from_file(
    "mmsk_skeleton_process",
    str(_MMSK / "mmskeleton/datasets/skeleton/skeleton_process.py"))


class Pipeline(torch.utils.data.Dataset):
    """mmskeleton DataPipeline equivalent: SkeletonLoader + processing stages."""

    def __init__(self, data_dir, train, window, num_keypoints, repeat=1):
        self.source = SkeletonLoader(str(data_dir), num_track=1, repeat=repeat,
                                     num_keypoints=num_keypoints)
        self.train = train
        self.window = window

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        data = self.source[index]
        data = sp.normalize_by_resolution(data)
        data = sp.mask_by_visibility(data)
        data = sp.pad_zero(data, size=self.window)
        data = sp.random_crop(data, size=self.window)
        if self.train:
            data = sp.simulate_camera_moving(data)
        data = sp.transpose(data, order=[0, 2, 1, 3])
        x, y = sp.to_tuple(data)
        return torch.from_numpy(np.ascontiguousarray(x)), int(y)


def infer_classes(data_dirs):
    """Scan dataset JSONs for the set of category ids -> (num_class, ids)."""
    ids = set()
    for d in data_dirs:
        for f in Path(d).glob("*.json"):
            with open(f, "r", encoding="utf-8") as fh:
                ids.add(json.load(fh)["category_id"])
    return (max(ids) + 1 if ids else 0), sorted(ids)


def build_model(num_class, layout="coco", dropout=0.0):
    ST_GCN_18 = stgcn_infer.load_stgcn_model_class()
    return ST_GCN_18(
        in_channels=3,
        num_class=num_class,
        edge_importance_weighting=True,
        graph_cfg={"layout": layout, "strategy": "spatial"},
        dropout=dropout,
    )


def evaluate(model, loader, device="cpu"):
    model.eval()
    hits1 = hits5 = total = 0
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            _, top5 = out.topk(min(5, out.size(1)), dim=1)
            y = y.to(device).view(-1, 1)
            hits1 += (top5[:, :1] == y).any(1).sum().item()
            hits5 += (top5 == y).any(1).sum().item()
            total += y.numel()
    return hits1 / max(total, 1), hits5 / max(total, 1)


def pick_device(requested):
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(data_root, epochs, batch_size, lr, window, val_window, layout,
          work_dir, workers, lr_steps, repeat, device="auto", dropout=0.0,
          weight_decay=1e-4):
    train_dir = Path(data_root) / "train"
    val_dir = Path(data_root) / "val"
    if not train_dir.is_dir():
        raise SystemExit(f"[train] no train/ dir in {data_root} -- run "
                        "build_dataset.py first (see module docstring)")
    has_val = val_dir.is_dir() and any(val_dir.glob("*.json"))

    num_class, ids = infer_classes([train_dir] + ([val_dir] if has_val else []))
    n_kpts = 17 if layout == "coco" else (18 if layout == "openpose" else 25)
    print(f"[train] classes: {num_class} (ids {ids[0]}..{ids[-1]}, "
         f"{len(ids)} distinct)   layout={layout} V={n_kpts}")

    # If build_dataset.py wrote a label_map.json (for an --only-actions subset),
    # embed it in every checkpoint so the model is self-describing: downstream
    # tools (run_pipeline.py --custom-model) can print real action names and
    # score against NTU ground truth without needing the dataset dir.
    label_map = None
    lm_path = Path(data_root) / "label_map.json"
    if lm_path.exists():
        label_map = json.loads(lm_path.read_text())
        print(f"[train] embedding label_map ({len(label_map)} classes) "
              "into checkpoints")

    train_set = Pipeline(train_dir, train=True, window=window,
                         num_keypoints=n_kpts, repeat=repeat)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=workers,
        drop_last=True)
    val_loader = None
    if has_val:
        # A larger window than training's, matching mmskeleton's own recipe:
        # large enough that random_crop is a no-op for ~any real clip, so
        # val is evaluated on the *whole* sequence, deterministically --
        # reusing the small training window here would silently evaluate a
        # different random crop every epoch instead.
        val_set = Pipeline(val_dir, train=False, window=val_window,
                           num_keypoints=n_kpts)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                                 num_workers=workers)
    print(f"[train] clips: train={len(train_set)} (repeat={repeat}) "
         f"val={len(val_set) if val_loader else 0}")

    device = pick_device(device)
    print(f"[train] device: {device}"
         + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    model = build_model(num_class, layout, dropout=dropout).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                nesterov=True, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=lr_steps,
                                                     gamma=0.1)

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    best_top1 = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        t0, running, batches = time.time(), 0.0, 0
        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x.to(device))
            loss = loss_fn(out, torch.as_tensor(y).to(device))
            loss.backward()
            optimizer.step()
            running += loss.item()
            batches += 1
        scheduler.step()
        train_loss = running / max(batches, 1)

        msg = (f"[epoch {epoch:3d}/{epochs}] loss={train_loss:.4f} "
              f"lr={scheduler.get_last_lr()[0]:g} "
              f"({time.time() - t0:.0f}s)")
        entry = {"epoch": epoch, "train_loss": round(train_loss, 4)}
        if val_loader:
            top1, top5 = evaluate(model, val_loader, device)
            entry.update(val_top1=round(top1, 4), val_top5=round(top5, 4))
            msg += f"  val top-1={100 * top1:.1f}%  top-5={100 * top5:.1f}%"
            if top1 > best_top1:
                best_top1 = top1
                torch.save({"state_dict": model.state_dict(),
                            "num_class": num_class, "layout": layout,
                            "epoch": epoch, "val_top1": top1,
                            "label_map": label_map},
                           work_dir / "best.pth")
                msg += "  *best*"
        print(msg, flush=True)
        history.append(entry)
        torch.save({"state_dict": model.state_dict(), "num_class": num_class,
                    "layout": layout, "epoch": epoch, "label_map": label_map},
                   work_dir / "latest.pth")
        (work_dir / "history.json").write_text(json.dumps(history, indent=2))

    print(f"\n[train] done. best val top-1: {100 * best_top1:.1f}%  "
         f"checkpoints in {work_dir}")


def evaluate_checkpoint(data_root, ckpt_path, batch_size, val_window, layout,
                        workers, device="auto"):
    device = pick_device(device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    layout = ckpt.get("layout", layout)
    n_kpts = 17 if layout == "coco" else (18 if layout == "openpose" else 25)
    val_dir = Path(data_root) / "val"
    val_set = Pipeline(val_dir, train=False, window=val_window, num_keypoints=n_kpts)
    loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                         num_workers=workers)
    model = build_model(ckpt["num_class"], layout).to(device)
    model.load_state_dict(ckpt["state_dict"])
    top1, top5 = evaluate(model, loader, device)
    print(f"[eval] {ckpt_path} (epoch {ckpt.get('epoch')}) on {len(val_set)} "
         f"val clips:  top-1={100 * top1:.2f}%  top-5={100 * top5:.2f}%")


def _cli():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data_root", help="dataset dir with train/ and val/ "
                                     "(from build_dataset.py)")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--lr-steps", type=int, nargs="+", default=[40, 50])
    p.add_argument("--window", type=int, default=150,
                   help="training temporal window (frames); random_crop makes "
                        "this an augmentation, so smaller than val-window is "
                        "intentional")
    p.add_argument("--val-window", type=int, default=300,
                   help="val/eval temporal window; kept large so random_crop "
                        "is a no-op and evaluation covers the full clip "
                        "deterministically (default: 300, matching mmskeleton)")
    p.add_argument("--layout", default="coco",
                   choices=["coco", "openpose", "ntu-rgb+d"])
    p.add_argument("--repeat", type=int, default=1,
                   help="repeat the train set N times per epoch (mmskeleton "
                        "used 20 for tiny datasets)")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--work-dir", default=str(_HERE / "work_dir"))
    p.add_argument("--device", default="auto",
                   help="auto | cpu | cuda (default: cuda if available)")
    p.add_argument("--dropout", type=float, default=0.0,
                   help="dropout inside each st_gcn_block's TCN (default 0, "
                        "matching mmskeleton's own default -- raise this, "
                        "e.g. 0.5, to fight overfitting on small datasets)")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--evaluate", metavar="CKPT",
                   help="skip training; evaluate this checkpoint on val/")
    args = p.parse_args()

    if args.evaluate:
        evaluate_checkpoint(args.data_root, args.evaluate, args.batch_size,
                            args.val_window, args.layout, args.workers,
                            args.device)
    else:
        train(args.data_root, args.epochs, args.batch_size, args.lr,
              args.window, args.val_window, args.layout, args.work_dir,
              args.workers, args.lr_steps, args.repeat, args.device,
              args.dropout, args.weight_decay)


if __name__ == "__main__":
    _cli()
