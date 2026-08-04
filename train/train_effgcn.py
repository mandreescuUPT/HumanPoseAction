"""
train_effgcn.py
===============

Path B, EfficientGCN edition: train **EfficientGCN** (Song et al., T-PAMI 2022)
from scratch on MediaPipe keypoints — the same recipe and tooling as
`train_stgcn.py`, just a stronger model.

Like `train_stgcn.py` reuses mmskeleton's `ST_GCN_18`, this reuses the
EfficientGCN repo's own model code, loaded directly from source:

    src/model/            EfficientGCN, layers, attentions, activations
    src/dataset/graphs.py Graph  (the `ntu` 25-joint graph: A, connect_joint, parts)
    src/dataset/ntu_feeder multi_input()  (joint / velocity / bone branches)

...and it bypasses the repo's own trainer (`main.py`/`initializer.py`), which
hard-requires thop/pynvml/tensorboardX + CUDA. Only the pure torch/numpy model
subtree is touched.

Input contract. EfficientGCN wants (N, I, C, T, V, M):
  I = input branches (J/V/B), C = 2*raw_channels (multi_input doubles them),
  V = 25 joints (ntu layout), M = persons. So the dataset must be built with
  the **ntu-rgb+d** layout, which mp_to_mmskeleton derives from MediaPipe's 33
  landmarks and which matches EfficientGCN's `ntu` graph exactly.

Usage
-----
    # 1. build an ntu-layout dataset (subject-holdout split; subset if wanted):
    python build_dataset.py ../HumanPoseAction/output ./data/eff5 \
        --layout ntu-rgb+d --from-ntu-filename --val-split 0.25 \
        --only-actions 8 9 23 27 43

    # 2. train:
    python train_effgcn.py ./data/eff5 --epochs 60 --device cpu

    # 3. evaluate a checkpoint:
    python train_effgcn.py ./data/eff5 --evaluate work_dir_eff/best.pth
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
# Location of the effgcn_cam repo: EFFGCN_ROOT env var if set, else
# ../../effgcn_cam relative to this file -- i.e. clone it as a sibling of
# the HumanPoseAction/ repo itself, not inside it (see stgcn_infer.py's
# _MMSKELETON_ROOT comment for the same layout, mmskeleton's counterpart).
EFFGCN_ROOT = Path(os.environ["EFFGCN_ROOT"]) if os.environ.get("EFFGCN_ROOT") \
    else _HERE.parent.parent / "effgcn_cam"

# effgcn_cam must be importable as the top-level package root, because the repo
# refers to its own modules as `src.model.layers...` (see nets.py import_class).
sys.path.insert(0, str(EFFGCN_ROOT))
sys.path.insert(0, str(_HERE))

import src.model as effgcn_factory                 # noqa: E402  (create())
from src.dataset.graphs import Graph               # noqa: E402

# EfficientGCN-B0 / SG config, copied from configs/2001.yaml (model_args).
B0_MODEL_ARGS = dict(
    stem_channel=64,
    block_args=[[48, 1, 0.5], [24, 1, 0.5], [64, 2, 1], [128, 2, 1]],
    fusion_stage=2,
    act_type="swish",
    att_type="stja",
    layer_type="SG",
    drop_prob=0.25,
    kernel_size=[5, 2],
    scale_args=[1.2, 1.35],
    expand_ratio=0,
    reduct_ratio=2,
    bias=True,
    edge=True,
)
RAW_CHANNELS = 3          # (x, y, score) per joint, as build_dataset emits
INPUTS = "JVB"            # joint / velocity / bone branches


# ── data ────────────────────────────────────────────────────────────────────────

def _multi_input(data, conn):
    """joint / velocity / bone branches from raw (C, T, V, M).
    Faithful port of NTU_Feeder.multi_input (src/dataset/ntu_feeder.py)."""
    C, T, V, M = data.shape
    joint = np.zeros((C * 2, T, V, M), dtype=np.float32)
    velocity = np.zeros((C * 2, T, V, M), dtype=np.float32)
    bone = np.zeros((C * 2, T, V, M), dtype=np.float32)
    joint[:C] = data
    for i in range(V):
        joint[C:, :, i, :] = data[:, :, i, :] - data[:, :, 1, :]
    for i in range(T - 2):
        velocity[:C, i, :, :] = data[:, i + 1, :, :] - data[:, i, :, :]
        velocity[C:, i, :, :] = data[:, i + 2, :, :] - data[:, i, :, :]
    for i in range(len(conn)):
        bone[:C, :, i, :] = data[:, :, i, :] - data[:, :, conn[i], :]
    bone_length = np.sqrt((bone[:C] ** 2).sum(0)) + 1e-4
    for i in range(C):
        bone[C + i] = np.arccos(np.clip(bone[i] / bone_length, -1.0, 1.0))
    branches = {"J": joint, "V": velocity, "B": bone}
    return np.stack([branches[k] for k in INPUTS], axis=0)  # (I, C*2, T, V, M)


class EffFeeder(torch.utils.data.Dataset):
    """Reads build_dataset ntu-layout skeleton JSONs -> EfficientGCN branches."""

    def __init__(self, data_dir, conn, window, num_keypoints=25):
        self.files = sorted(glob.glob(str(Path(data_dir) / "*.json")))
        self.conn = conn
        self.window = window
        self.V = num_keypoints

    def __len__(self):
        return len(self.files)

    def _raw(self, path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        T = d["info"]["num_frame"]
        raw = np.zeros((RAW_CHANNELS, T, self.V, 1), dtype=np.float32)
        for a in d["annotations"]:
            t = a["frame_index"]
            if t >= T:
                continue
            kp = np.asarray(a["keypoints"], dtype=np.float32)      # (V, 3)
            raw[:, t, :, 0] = kp[:, :RAW_CHANNELS].T
        return raw, int(d["category_id"])

    def __getitem__(self, idx):
        raw, cat = self._raw(self.files[idx])
        # pad/crop the temporal axis to a fixed window (T)
        C, T, V, M = raw.shape
        if T < self.window:
            pad = np.zeros((C, self.window, V, M), dtype=np.float32)
            pad[:, :T] = raw
            raw = pad
        elif T > self.window:
            raw = raw[:, :self.window]
        x = _multi_input(raw, self.conn)                          # (I, C*2, T, V, M)
        return torch.from_numpy(np.ascontiguousarray(x)), cat


def infer_classes(dirs):
    ids = set()
    for d in dirs:
        for f in Path(d).glob("*.json"):
            with open(f, "r", encoding="utf-8") as fh:
                ids.add(json.load(fh)["category_id"])
    return (max(ids) + 1 if ids else 0), sorted(ids)


# ── model ─────────────────────────────────────────────────────────────────────

def build_model(num_class, window, num_track=1):
    graph = Graph("ntu")
    V = graph.num_node
    data_shape = [len(INPUTS), RAW_CHANNELS * 2, window, V, num_track]
    args = dict(B0_MODEL_ARGS)
    model = effgcn_factory.create(
        "EfficientGCN-B0",
        data_shape=data_shape,
        num_class=num_class,
        A=torch.Tensor(graph.A),
        parts=graph.parts,
        **args,
    )
    return model, graph


def evaluate(model, loader, device):
    model.eval()
    hits1 = hits5 = total = 0
    with torch.no_grad():
        for x, y in loader:
            out, _ = model(x.to(device))            # forward returns (logits, feat)
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


# ── train ─────────────────────────────────────────────────────────────────────

def train(data_root, epochs, batch_size, lr, window, work_dir, workers,
          lr_steps, device="auto"):
    train_dir, val_dir = Path(data_root) / "train", Path(data_root) / "val"
    if not train_dir.is_dir():
        raise SystemExit(f"[train] no train/ dir in {data_root} — run "
                        "build_dataset.py --layout ntu-rgb+d first")
    has_val = val_dir.is_dir() and any(val_dir.glob("*.json"))

    num_class, ids = infer_classes([train_dir] + ([val_dir] if has_val else []))
    print(f"[train] EfficientGCN-B0  classes={num_class} (ids {ids[0]}..{ids[-1]})"
          f"  inputs={INPUTS}  window={window}")

    label_map = None
    lm = Path(data_root) / "label_map.json"
    if lm.exists():
        label_map = json.loads(lm.read_text())

    device = pick_device(device)
    model, graph = build_model(num_class, window)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] device={device}  params={n_params/1e6:.2f}M  V={graph.num_node}")

    pin_memory = device.type == "cuda"
    train_set = EffFeeder(train_dir, graph.connect_joint, window)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=workers,
        drop_last=True, pin_memory=pin_memory)
    val_loader = None
    if has_val:
        val_set = EffFeeder(val_dir, graph.connect_joint, window)
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=batch_size, num_workers=workers,
            pin_memory=pin_memory)
    print(f"[train] clips: train={len(train_set)} val={len(val_set) if val_loader else 0}")

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                nesterov=True, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=lr_steps, gamma=0.1)

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    best_top1, history = -1.0, []

    for epoch in range(1, epochs + 1):
        model.train()
        t0, running, batches = time.time(), 0.0, 0
        for x, y in train_loader:
            optimizer.zero_grad()
            out, _ = model(x.to(device))
            loss = loss_fn(out, torch.as_tensor(y).to(device))
            loss.backward()
            optimizer.step()
            running += loss.item()
            batches += 1
        scheduler.step()
        train_loss = running / max(batches, 1)

        msg = (f"[epoch {epoch:3d}/{epochs}] loss={train_loss:.4f} "
              f"lr={scheduler.get_last_lr()[0]:g} ({time.time()-t0:.0f}s)")
        entry = {"epoch": epoch, "train_loss": round(train_loss, 4)}
        if val_loader:
            top1, top5 = evaluate(model, val_loader, device)
            entry.update(val_top1=round(top1, 4), val_top5=round(top5, 4))
            msg += f"  val top-1={100*top1:.1f}%  top-5={100*top5:.1f}%"
            if top1 > best_top1:
                best_top1 = top1
                torch.save({"state_dict": model.state_dict(),
                            "num_class": num_class, "arch": "EfficientGCN-B0",
                            "layout": "ntu-rgb+d", "inputs": INPUTS,
                            "window": window, "epoch": epoch, "val_top1": top1,
                            "label_map": label_map}, work_dir / "best.pth")
                msg += "  *best*"
        print(msg, flush=True)
        history.append(entry)
        torch.save({"state_dict": model.state_dict(), "num_class": num_class,
                    "arch": "EfficientGCN-B0", "layout": "ntu-rgb+d",
                    "inputs": INPUTS, "window": window, "epoch": epoch,
                    "label_map": label_map}, work_dir / "latest.pth")
        (work_dir / "history.json").write_text(json.dumps(history, indent=2))

    print(f"\n[train] done. best val top-1: {100*best_top1:.1f}%  -> {work_dir}")


def evaluate_checkpoint(data_root, ckpt_path, batch_size, workers, device="auto"):
    device = pick_device(device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    window = ckpt["window"]
    model, graph = build_model(ckpt["num_class"], window)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    val_set = EffFeeder(Path(data_root) / "val", graph.connect_joint, window)
    loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                         num_workers=workers,
                                         pin_memory=(device.type == "cuda"))
    top1, top5 = evaluate(model, loader, device)
    print(f"[eval] {ckpt_path} (epoch {ckpt.get('epoch')}) on {len(val_set)} "
          f"val clips:  top-1={100*top1:.2f}%  top-5={100*top5:.2f}%")


def _cli():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data_root", help="dataset dir with train/ + val/ "
                                     "(build_dataset.py --layout ntu-rgb+d)")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--lr-steps", type=int, nargs="+", default=[40, 50])
    p.add_argument("--window", type=int, default=128,
                   help="temporal frames fed to the model (pad/crop, default 128)")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--work-dir", default=str(_HERE / "work_dir_eff"))
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p.add_argument("--evaluate", metavar="CKPT",
                   help="skip training; evaluate this checkpoint on val/")
    args = p.parse_args()

    if args.evaluate:
        evaluate_checkpoint(args.data_root, args.evaluate, args.batch_size,
                            args.workers, args.device)
    else:
        train(args.data_root, args.epochs, args.batch_size, args.lr, args.window,
              args.work_dir, args.workers, args.lr_steps, args.device)


if __name__ == "__main__":
    _cli()
