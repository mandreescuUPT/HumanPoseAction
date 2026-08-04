"""
stgcn_infer.py
==============

Run mmskeleton's ST-GCN on keypoints produced by HumanPoseAction, using the
pretrained checkpoints shipped in `mmskeleton/checkpoints/`.

The full mmskeleton training framework depends on an ancient mmcv/mmdet stack that
will not install on modern Python.  The *model itself*, however, is pure PyTorch:
`mmskeleton/models/backbones/st_gcn_aaai18.py` only needs `torch` plus the two ops
modules `graph.py` and `gconv_origin.py`.  This script loads exactly those three
files directly (bypassing the package `__init__`, which imports mmcv), rebuilds the
`(N, C, T, V, M)` input tensor the way `SkeletonLoader` + the data pipeline would,
loads the matching checkpoint, and prints the top-k predicted actions.

Usage
-----
    # Input can be a HumanPoseAction export OR an already-converted skeleton JSON.
    python stgcn_infer.py <keypoints.json> --checkpoint <path.pth> --layout openpose

    # Convenience: point at the repo checkpoints by name
    python stgcn_infer.py <keypoints.json> --model kinetics
    python stgcn_infer.py <keypoints.json> --model ntu-xsub
"""

import argparse
import importlib.util
import json
import os
import sys
import types

import numpy as np
import torch

import mp_to_mmskeleton
import labels as label_maps

# Location of the mmskeleton repo: MMSKELETON_ROOT env var if set, else
# ../../mmskeleton relative to this file -- i.e. clone mmskeleton as a
# sibling of the HumanPoseAction/ repo itself, not inside it:
#   workspace/
#   ├── HumanPoseAction/train/   <- this file lives here
#   ├── mmskeleton/               <- expected here
#   └── effgcn_cam/
_HERE = os.path.dirname(os.path.abspath(__file__))
_MMSKELETON_ROOT = os.environ.get("MMSKELETON_ROOT") or os.path.normpath(
    os.path.join(_HERE, "..", "..", "mmskeleton"))

# Friendly names -> (checkpoint filename, graph layout, label set).
MODELS = {
    "kinetics":  ("st_gcn.kinetics-6fa43f73.pth",  "openpose",   "openpose"),
    "ntu-xsub":  ("st_gcn.ntu-xsub-300b57d4.pth",  "ntu-rgb+d",  "ntu-rgb+d"),
    "ntu-xview": ("st_gcn.ntu-xview-9ba67746.pth",  "ntu-rgb+d",  "ntu-rgb+d"),
}
NUM_CLASS = {"openpose": 400, "ntu-rgb+d": 60, "coco": 3}


def _load_module_from_file(mod_name, file_path):
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def load_stgcn_model_class():
    """Import ST_GCN_18 without triggering mmskeleton's mmcv-dependent __init__."""
    ops_dir = os.path.join(_MMSKELETON_ROOT, "mmskeleton", "ops", "st_gcn")
    backbone = os.path.join(_MMSKELETON_ROOT, "mmskeleton", "models",
                            "backbones", "st_gcn_aaai18.py")

    # Pre-register lightweight stub packages so that
    # `from mmskeleton.ops.st_gcn import Graph, ConvTemporalGraphical`
    # inside st_gcn_aaai18.py resolves to the two real op files below and does
    # NOT execute the real (mmcv-importing) package __init__ files.
    for pkg in ("mmskeleton", "mmskeleton.ops", "mmskeleton.models",
                "mmskeleton.models.backbones"):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = []  # mark as package
            sys.modules[pkg] = m

    graph_mod = _load_module_from_file(
        "mmskeleton.ops.st_gcn._graph", os.path.join(ops_dir, "graph.py"))
    gconv_mod = _load_module_from_file(
        "mmskeleton.ops.st_gcn._gconv", os.path.join(ops_dir, "gconv_origin.py"))

    st_gcn_ops = types.ModuleType("mmskeleton.ops.st_gcn")
    st_gcn_ops.__path__ = []
    st_gcn_ops.Graph = graph_mod.Graph
    st_gcn_ops.ConvTemporalGraphical = gconv_mod.ConvTemporalGraphical
    sys.modules["mmskeleton.ops.st_gcn"] = st_gcn_ops

    backbone_mod = _load_module_from_file(
        "mmskeleton.models.backbones._stgcn18", backbone)
    return backbone_mod.ST_GCN_18


def load_skeleton_json(path, layout):
    """Load either a HumanPoseAction export or a converted skeleton JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "frames" in data and "metadata" in data:  # raw HumanPoseAction export
        print("[infer] detected HumanPoseAction export -> converting to layout "
              "'%s'" % layout)
        data = mp_to_mmskeleton.convert(data, layout=layout, category_id=-1)
    elif data.get("info", {}).get("layout") not in (None, layout):
        print("[infer] WARNING: skeleton JSON layout '%s' != requested '%s'"
              % (data["info"].get("layout"), layout))
    return data


def build_input_tensor(skeleton, num_track=2, max_frames=None):
    """Reproduce SkeletonLoader + normalize_by_resolution + mask_by_visibility,
    returning an (N=1, C=3, T, V, M) float tensor ready for ST_GCN_18."""
    info = skeleton["info"]
    V = info["num_keypoints"]
    T = info["num_frame"]
    C = len(info["keypoint_channels"])  # 3: x, y, score
    res = info.get("resolution", [1, 1])

    # (C, V, T, M) as SkeletonLoader builds it.
    data = np.zeros((C, V, T, num_track), dtype=np.float32)
    for a in skeleton["annotations"]:
        pid = a["id"] if a.get("person_id") is None else a["person_id"]
        t = a["frame_index"]
        if pid < num_track and t < T:
            data[:, :, t, pid] = np.array(a["keypoints"], dtype=np.float32).T

    # normalize_by_resolution: x/W-0.5, y/H-0.5  (res=[1,1] => pure centering)
    data[0] = data[0] / res[0] - 0.5
    data[1] = data[1] / res[1] - 0.5
    # mask_by_visibility: zero x,y where score channel == 0
    invisible = data[2] == 0
    data[0][invisible] = 0
    data[1][invisible] = 0

    # transpose to (C, T, V, M) as the data pipeline's transpose [0,2,1,3] does.
    data = data.transpose(0, 2, 1, 3)

    if max_frames and data.shape[1] > max_frames:
        data = data[:, :max_frames]

    tensor = torch.from_numpy(np.ascontiguousarray(data)).unsqueeze(0)  # add N
    return tensor


def pick_device(requested):
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint, layout, device="cpu"):
    """Build ST_GCN_18 for `layout` and load `checkpoint` into it (eval mode)."""
    ST_GCN_18 = load_stgcn_model_class()
    model = ST_GCN_18(
        in_channels=3,
        num_class=NUM_CLASS[layout],
        edge_importance_weighting=True,
        graph_cfg={"layout": layout, "strategy": "spatial"},
    )

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[infer] (info) missing keys not in checkpoint: %d" % len(missing))
    if unexpected:
        print("[infer] (info) unexpected keys in checkpoint: %d" % len(unexpected))
    model.to(device)
    model.eval()
    return model


def predict(keypoints_path, checkpoint, layout, num_track=2, max_frames=None,
            model=None, device="cpu"):
    """Full softmax over the clip.

    Returns (probs, info) where probs is a (num_class,) float list and info is
    the skeleton's `info` dict plus a `num_detected` count.  Pass `model` (from
    `load_model`) to reuse an already-loaded checkpoint across clips.
    """
    skeleton = load_skeleton_json(keypoints_path, layout)
    n_detected = len(skeleton["annotations"])
    print("[infer] clip='%s'  layout=%s  V=%d  frames=%d (detected=%d)"
          % (skeleton["info"].get("video_name"), layout,
             skeleton["info"]["num_keypoints"], skeleton["info"]["num_frame"],
             n_detected))
    if n_detected == 0:
        raise SystemExit("[infer] ERROR: no detected frames in input.")

    x = build_input_tensor(skeleton, num_track=num_track, max_frames=max_frames)
    x = x.to(device)
    print("[infer] model input tensor (N,C,T,V,M) = %s" % (tuple(x.shape),))

    if model is None:
        model = load_model(checkpoint, layout, device=device)
    with torch.no_grad():
        logits = model(x)                       # (1, num_class)
        probs = torch.softmax(logits, dim=1)[0]

    info = dict(skeleton["info"])
    info["num_detected"] = n_detected
    return probs.tolist(), info


def run(keypoints_path, checkpoint, layout, label_set, topk=5,
        num_track=2, max_frames=None, device="cpu"):
    probs, _info = predict(keypoints_path, checkpoint, layout,
                           num_track=num_track, max_frames=max_frames,
                           device=device)
    probs = torch.tensor(probs)

    names = label_maps.labels_for(label_set)
    k = min(topk, probs.numel())
    top_p, top_i = torch.topk(probs, k)
    print("\n[infer] Top-%d predictions:" % k)
    for rank, (p, idx) in enumerate(zip(top_p.tolist(), top_i.tolist()), 1):
        name = names[idx] if names and idx < len(names) else "class_%d" % idx
        print("  %d. %-40s  %5.1f%%  (id=%d)" % (rank, name, 100 * p, idx))
    return top_i[0].item(), names[top_i[0].item()] if names else None


def _cli():
    p = argparse.ArgumentParser(
        description="Run pretrained mmskeleton ST-GCN on HumanPoseAction keypoints.")
    p.add_argument("keypoints", help="HumanPoseAction export or converted skeleton JSON")
    p.add_argument("--model", choices=list(MODELS),
                   help="use a repo checkpoint by name (sets checkpoint+layout)")
    p.add_argument("--checkpoint", help="explicit .pth path (overrides --model)")
    p.add_argument("--layout", choices=["openpose", "coco", "ntu-rgb+d"],
                   help="graph layout (inferred from --model if omitted)")
    p.add_argument("--label-set", choices=["openpose", "ntu-rgb+d", "coco"],
                   help="which label names to print (defaults to layout)")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--num-track", type=int, default=2,
                   help="M dimension / persons (kinetics=2, ntu=2)")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--device", default="auto",
                   help="auto | cpu | cuda (default: cuda if available)")
    args = p.parse_args()

    layout = args.layout
    checkpoint = args.checkpoint
    label_set = args.label_set
    if args.model:
        ckpt_name, m_layout, m_labels = MODELS[args.model]
        checkpoint = checkpoint or os.path.join(
            _MMSKELETON_ROOT, "checkpoints", ckpt_name)
        layout = layout or m_layout
        label_set = label_set or m_labels
    if not checkpoint or not layout:
        p.error("provide --model, or both --checkpoint and --layout")
    label_set = label_set or layout

    run(args.keypoints, checkpoint, layout, label_set,
        topk=args.topk, num_track=args.num_track, max_frames=args.max_frames,
        device=pick_device(args.device))


if __name__ == "__main__":
    _cli()
