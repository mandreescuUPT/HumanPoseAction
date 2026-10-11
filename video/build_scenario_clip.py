"""
build_scenario_clip.py
=======================

Fully standalone: builds a synthetic multi-action test clip AND recognises
composite activities in it, in one file, with no Python imports from any
other file in this pipeline -- no recognize_composite.py, no
compare_inference.py, no mp_to_mmskeleton.py, no labels.py, no run_pipeline.py.

    python build_scenario_clip.py --actions 15 8 9 --run
    python build_scenario_clip.py --actions "take off jacket" "sitting down" --run
    python build_scenario_clip.py --preset medical --run

Actions can be given as NTU numbers, as the action's own name (quote the ones
with spaces), or as a single --preset keyword standing in for a whole known
scenario (medical, illness, fall_recover, arrival, fetch_dispose,
point_retrieve, pickpocket, altercation, greeting). Give exactly one of
--actions / --preset.

WHAT'S COPIED IN vs. WHAT CANNOT BE AVOIDED
--------------------------------------------
Copied in below (frozen, not imported): the NTU action-name table (from
labels.py), the 9 preset scenarios (from recognize_composite.py), the
MediaPipe -> NTU-25 skeleton conversion (from mp_to_mmskeleton.py), and the
sliding-window / smoothing / segmentation / composite-matching logic (from
recognize_composite.py + compare_inference.py). None of those five files are
imported or called. If their originals are ever edited, this file will not
notice -- update both places by hand.

What is NOT copied in, and cannot reasonably be: EfficientGCN's own model
code (attention modules, depthwise-separable conv blocks, the graph
definition) in `effgcn_cam/`. That is the actual trained neural network, not
pipeline glue -- there is no "simpler" standalone version of a model's own
architecture, so this file imports it the same way every trainer in this
project does. `torch` and `numpy` are third-party libraries, same category as
any language runtime -- not project files either.

This only runs real recognition for EfficientGCN checkpoints (the project's
best-performing model, and this script's default). ST-GCN/CTR-GCN/TA-GCN/etc.
checkpoints come from two OTHER external repos with their own full model
source; supporting all of them standalone would mean vendoring three
third-party neural-network codebases into one convenience script, which is
not sensible to maintain. Use compare_inference.py for those.

This exists so composite sequences can be tested without filming anything: you
name the primitives in order, it finds real NTU clips for them, splices them
into one timeline with a KNOWN ground truth, and (with --run) scores a trained
model against that ground truth immediately.

Splicing happens at the keypoints level by default -- MediaPipe is essentially
per-frame, so concatenating extractions is equivalent to extracting a
concatenation, minus the minutes of re-extraction. Pass --video to also write a
real .mp4 (needs cv2, which lives in the MediaPipe interpreter, not the torch
one), for when you want to eyeball the footage.

Clips are drawn from ONE (setup, camera, person, replication) group where
possible, so it is the same actor filmed from the same angle throughout and the
splice reads as one continuous recording rather than a cut between strangers.
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
IPCEI_ROOT = _HERE.parent
HPA_OUTPUT = IPCEI_ROOT / "HumanPoseAction" / "output"

NTU_RE = re.compile(r"S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})")
VIDEO_ROOTS = ["nturgb+d_rgb", "nturgb+d_rgb_1", "nturgb+d_rgb_2", "ntu_folositor"]

# The interpreter that has mediapipe+opencv, for --video's frame splicing.
DEFAULT_EXTRACT_PYTHON = (
    r"C:\Users\alext\AppData\Local\Programs\Python\Python311\python.exe"
)

# The actual trained model -- EfficientGCN only (see module docstring). This
# file lives in HumanPoseAction/, but the trained checkpoints live in
# pipeline/models/ -- that's where train_effgcn.py writes them, not wherever
# this script happens to sit, so the path is anchored at IPCEI_ROOT, not _HERE.
DEFAULT_MODEL = str(IPCEI_ROOT / "pipeline" / "models" / "effgcn_94_LATEST" / "best.pth")

# EfficientGCN's own source tree (not this project's -- the model itself).
# Same convention train_effgcn.py uses: an EFFGCN_ROOT env var if set, else a
# sibling of the IPCEI root (effgcn_cam sits next to pipeline/, HumanPoseAction/, etc).
EFFGCN_ROOT = (Path(os.environ["EFFGCN_ROOT"]) if os.environ.get("EFFGCN_ROOT")
               else IPCEI_ROOT / "effgcn_cam")

# ── copied from labels.py (NTU_60) ──────────────────────────────────────────
# Official A001..A060 order (index i == class i, i.e. label i+1).
NTU_60 = [
    "drink water", "eat meal/snack", "brushing teeth", "brushing hair", "drop",
    "pickup", "throw", "sitting down", "standing up", "clapping",
    "reading", "writing", "tear up paper", "wear jacket", "take off jacket",
    "wear a shoe", "take off a shoe", "wear on glasses", "take off glasses",
    "put on a hat/cap", "take off a hat/cap", "cheer up", "hand waving",
    "kicking something", "reach into pocket", "hopping (one foot jumping)",
    "jump up", "make a phone call/answer phone", "playing with phone/tablet",
    "typing on a keyboard", "pointing to something with finger",
    "taking a selfie", "check time (from watch)", "rub two hands together",
    "nod head/bow", "shake head", "wipe face", "salute",
    "put the palms together", "cross hands in front (say stop)",
    "sneeze/cough", "staggering", "falling", "touch head (headache)",
    "touch chest (stomachache/heart pain)", "touch back (backache)",
    "touch neck (neckache)", "nausea or vomiting condition",
    "use a fan (with hand or paper)/feeling warm", "punching/slapping other person",
    "kicking other person", "pushing other person",
    "pat on back of other person", "point finger at the other person",
    "hugging other person", "giving something to other person",
    "touch other person's pocket", "handshaking",
    "walking towards each other", "walking apart from each other",
]

# ── copied from recognize_composite.py (DEFAULT_COMPOSITES) ────────────────
DEFAULT_COMPOSITES = {
    "medical_collapse": {
        "label": "Medical collapse",
        "reliability": "strong",
        "steps": [
            ["staggering"],
            ["touch chest (stomachache/heart pain)", "touch head (headache)"],
            ["falling"],
        ],
    },
    "illness_episode": {
        "label": "Illness episode",
        "reliability": "strong",
        "steps": [
            ["touch chest (stomachache/heart pain)"],
            ["staggering"],
            ["sitting down"],
            ["nausea or vomiting condition"],
        ],
    },
    "fall_and_recover": {
        "label": "Fall and recover",
        "reliability": "strong",
        "steps": [["staggering"], ["falling"], ["standing up"]],
    },
    "arrival_routine": {
        "label": "Arrival routine",
        "reliability": "strong",
        "steps": [["take off jacket"], ["sitting down"]],
    },
    "fetch_and_dispose": {
        "label": "Fetch and dispose",
        "reliability": "strong",
        "steps": [["pickup"], ["throw", "drop"]],
    },
    "point_and_retrieve": {
        "label": "Point and retrieve",
        "reliability": "strong",
        "steps": [
            ["pointing to something with finger"],
            ["pickup"],
            ["drop"],
        ],
    },
    "pickpocketing": {
        "label": "Pickpocketing",
        "reliability": "weak (two-person classes)",
        "steps": [
            ["walking towards each other"],
            ["touch other person's pocket"],
            ["walking apart from each other"],
        ],
    },
    "altercation": {
        "label": "Physical altercation",
        "reliability": "weak (two-person classes)",
        "steps": [
            ["walking towards each other"],
            ["pushing other person"],
            ["punching/slapping other person"],
            ["kicking other person"],
            ["falling"],
        ],
    },
    "greeting": {
        "label": "Greeting",
        "reliability": "weak (two-person classes)",
        "steps": [
            ["walking towards each other"],
            ["hand waving"],
            ["walking apart from each other"],
        ],
    },
}

# Short, typeable aliases for the scenarios above.
PRESETS = {
    "medical": "medical_collapse",
    "illness": "illness_episode",
    "fall_recover": "fall_and_recover",
    "arrival": "arrival_routine",
    "fetch_dispose": "fetch_and_dispose",
    "point_retrieve": "point_and_retrieve",
    "pickpocket": "pickpocketing",
    "altercation": "altercation",
    "greeting": "greeting",
}

# ── copied from mp_to_mmskeleton.py: the ntu-rgb+d (25-joint) layout only ──
# (COCO/OpenPose dropped -- EfficientGCN only ever consumes ntu-rgb+d.)
NTU_25 = [
    ("LEFT_HIP", "RIGHT_HIP"),                                     # 0  SpineBase
    ("LEFT_HIP", "RIGHT_HIP", "LEFT_SHOULDER", "RIGHT_SHOULDER"),   # 1  SpineMid
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),                           # 2  Neck
    ("LEFT_EAR", "RIGHT_EAR"),                                     # 3  Head
    "LEFT_SHOULDER",                                               # 4  ShoulderLeft
    "LEFT_ELBOW",                                                  # 5  ElbowLeft
    "LEFT_WRIST",                                                  # 6  WristLeft
    ("LEFT_INDEX", "LEFT_PINKY"),                                  # 7  HandLeft
    "RIGHT_SHOULDER",                                              # 8  ShoulderRight
    "RIGHT_ELBOW",                                                 # 9  ElbowRight
    "RIGHT_WRIST",                                                 # 10 WristRight
    ("RIGHT_INDEX", "RIGHT_PINKY"),                                # 11 HandRight
    "LEFT_HIP",                                                    # 12 HipLeft
    "LEFT_KNEE",                                                   # 13 KneeLeft
    "LEFT_ANKLE",                                                  # 14 AnkleLeft
    "LEFT_FOOT_INDEX",                                             # 15 FootLeft
    "RIGHT_HIP",                                                   # 16 HipRight
    "RIGHT_KNEE",                                                  # 17 KneeRight
    "RIGHT_ANKLE",                                                 # 18 AnkleRight
    "RIGHT_FOOT_INDEX",                                            # 19 FootRight
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),                           # 20 SpineShoulder
    "LEFT_INDEX",                                                  # 21 HandTipLeft
    "LEFT_THUMB",                                                  # 22 ThumbLeft
    "RIGHT_INDEX",                                                 # 23 HandTipRight
    "RIGHT_THUMB",                                                 # 24 ThumbRight
]


def _landmark_xy(kp_dict, name):
    """(x, y, visibility) for one MediaPipe landmark, or None if absent."""
    lm = kp_dict.get(name)
    if lm is None:
        return None
    return float(lm["x_norm"]), float(lm["y_norm"]), float(lm.get("visibility", 1.0))


def _resolve_joint(kp_dict, recipe):
    """One NTU_25 slot (direct copy or averaged proxy) -> [x, y, score]."""
    names = (recipe,) if isinstance(recipe, str) else recipe
    pts = [p for n in names if (p := _landmark_xy(kp_dict, n)) and p[2] > 0.0]
    if not pts:
        return [0.0, 0.0, 0.0]
    x = sum(p[0] for p in pts) / len(pts)
    y = sum(p[1] for p in pts) / len(pts)
    s = sum(p[2] for p in pts) / len(pts)
    return [round(x, 6), round(y, 6), round(s, 6)]


def keypoints_to_ntu25(hpa_json):
    """HumanPoseAction keypoints dict -> raw (C=3, T, V=25, M=1) joint array.

    Equivalent to mp_to_mmskeleton.convert(..., layout="ntu-rgb+d") followed
    by compare_inference's _raw_from_skeleton(), merged into one step since
    nothing downstream needs the intermediate {info, annotations} JSON shape.
    """
    import numpy as np
    frames = hpa_json.get("frames", [])
    T = len(frames)
    raw = np.zeros((3, T, 25, 1), dtype=np.float32)
    for t, frame in enumerate(frames):
        if not frame.get("detected") or not frame.get("keypoints"):
            continue
        kp = frame["keypoints"]
        for v, recipe in enumerate(NTU_25):
            raw[:, t, v, 0] = _resolve_joint(kp, recipe)
    return raw


# ── EfficientGCN model loading (the one unavoidable external dependency) ───
# Copied from train_effgcn.py / compare_inference.py's effgcn branch.

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
RAW_CHANNELS = 3
INPUTS = "JVB"  # joint / velocity / bone branches


def _multi_input(data, conn):
    """joint / velocity / bone branches from raw (C, T, V, M).
    Faithful port of EfficientGCN's NTU_Feeder.multi_input."""
    import numpy as np
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
    return np.stack([branches[k] for k in INPUTS], axis=0)


def load_effgcn(ckpt_path, device):
    """Load an EfficientGCN-B0 checkpoint -> a ready-to-run state dict.

    Requires effgcn_cam/ on disk (EFFGCN_ROOT above) -- that is the model's
    own source, not pipeline glue; see the module docstring.
    """
    import torch
    if not EFFGCN_ROOT.is_dir():
        raise SystemExit(
            f"[scenario] EfficientGCN source not found at {EFFGCN_ROOT}\n"
            "           (clone it there, or point EFFGCN_ROOT at it)")
    sys.path.insert(0, str(EFFGCN_ROOT))
    import src.model as effgcn_factory          # noqa: E402  (create())
    from src.dataset.graphs import Graph        # noqa: E402

    dev = torch.device(device if device != "auto"
                       else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch")
    # Plain mmskeleton ST-GCN checkpoints (train_stgcn.py) carry NO 'arch' key
    # at all -- a missing key is just as much "not EfficientGCN" as a wrong
    # one, so both must be rejected here, not just an explicit mismatch.
    if not arch or not str(arch).startswith("EfficientGCN"):
        found = arch or "stgcn (mmskeleton, no 'arch' field -- layout=%r)" % ckpt.get("layout")
        raise SystemExit(
            f"[scenario] {ckpt_path} is a {found} checkpoint, not "
            "EfficientGCN -- this standalone script only runs EfficientGCN "
            "checkpoints (see the module docstring). Use compare_inference.py "
            "for other architectures.")

    graph = Graph("ntu")
    window = ckpt.get("window") or 128
    data_shape = [len(INPUTS), RAW_CHANNELS * 2, window, graph.num_node, 1]
    model = effgcn_factory.create(
        "EfficientGCN-B0", data_shape=data_shape, num_class=ckpt["num_class"],
        A=torch.Tensor(graph.A), parts=graph.parts, **B0_MODEL_ARGS)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(dev)

    return {"model": model, "graph": graph, "dev": dev, "ckpt": ckpt,
            "window": window, "label_map": ckpt.get("label_map") or {}}


# ── stage 2: sliding-window classification ──────────────────────────────────

def sliding_probs(st, raw, stride):
    """Run the model over overlapping windows -> (starts, probs (W, num_class)).

    Window centres sweep the whole clip (including negative starts and starts
    past the end, zero-padded), so the opening/closing frames of the clip get
    windows centred on them too -- otherwise an action living right at an edge
    never gets a window of its own.
    """
    import numpy as np
    import torch
    window = st["window"]
    n_frames = raw.shape[1]
    half = window // 2
    first, last = -half, max(n_frames - half, -half)
    starts = list(range(first, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)

    C, _, V, M = raw.shape
    out = []
    with torch.no_grad():
        for s in starts:
            s0, s1 = max(0, s), min(n_frames, s + window)
            seg = np.zeros((C, window, V, M), dtype=np.float32)
            if s1 > s0:
                seg[:, s0 - s:s1 - s] = raw[:, s0:s1]
            x = _multi_input(seg, st["graph"].connect_joint)
            x = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).to(st["dev"])
            logits, _ = st["model"](x)
            out.append(torch.softmax(logits, dim=1)[0].cpu().numpy())
    return starts, np.stack(out), window, n_frames


# ── stage 3: smoothing + segmentation (verbatim from recognize_composite.py) ─

def smooth(probs, k):
    """Moving average over the probability vectors of k adjacent windows.

    Averaging the distributions (rather than voting on argmax labels) keeps a
    window that is split 45/55 between two actions from casting a full vote
    for the wrong one.
    """
    import numpy as np
    if k <= 1 or len(probs) <= 1:
        return probs
    pad = k // 2
    padded = np.pad(probs, ((pad, pad), (0, 0)), mode="edge")
    return np.stack([padded[i:i + k].mean(axis=0) for i in range(len(probs))])


def segment(starts, probs, labels, window, fps, min_conf, min_windows,
            stride, n_frames):
    """Collapse consecutive same-label windows into primitive segments.

    Boundaries come from each window's CENTRE, not its full extent, because
    adjacent windows overlap heavily -- using full extents would make
    neighbouring segments overlap by a whole window.
    """
    import numpy as np
    top = probs.argmax(axis=1)
    conf = probs.max(axis=1)

    runs, cur = [], None
    for i, (lab, c) in enumerate(zip(top, conf)):
        lab = int(lab)
        if c < min_conf:
            lab = None
        if cur and cur["lab"] == lab:
            cur["idx"].append(i)
        else:
            cur = {"lab": lab, "idx": [i]}
            runs.append(cur)

    segs = []
    for r in runs:
        if r["lab"] is None or len(r["idx"]) < min_windows:
            continue
        first, last = r["idx"][0], r["idx"][-1]
        half = window // 2
        c0, c1 = starts[first] + half, starts[last] + half
        f0 = max(0, c0 - stride // 2)
        f1 = min(n_frames, c1 + stride // 2)
        segs.append({
            "label": labels.get(str(r["lab"]), {}).get("label", f"class_{r['lab']}"),
            "class_id": r["lab"],
            "start_frame": int(f0), "end_frame": int(f1),
            "start_s": round(f0 / fps, 2), "end_s": round(f1 / fps, 2),
            "confidence": round(float(np.mean([conf[i] for i in r["idx"]])), 4),
            "n_windows": len(r["idx"]),
        })
    return segs


# ── stage 4: composite matching (verbatim from recognize_composite.py) ─────

def match_composite(segments, steps):
    """Greedy ordered-subsequence match, gaps allowed: a step that finds no
    match is skipped (it counts against coverage) without failing later steps."""
    matched, si = [], 0
    for step in steps:
        hit = None
        j = si
        while j < len(segments):
            if segments[j]["label"] in step:
                hit = j
                break
            j += 1
        if hit is not None:
            matched.append(hit)
            si = hit + 1
    return matched


def score_composites(segments, composites):
    import numpy as np
    results = []
    for key, spec in composites.items():
        steps = spec["steps"]
        matched = match_composite(segments, steps)
        coverage = len(matched) / len(steps) if steps else 0.0
        mean_conf = float(np.mean([segments[i]["confidence"] for i in matched])) if matched else 0.0
        results.append({
            "key": key, "label": spec.get("label", key),
            "reliability": spec.get("reliability", ""),
            "coverage": round(coverage, 3), "steps_total": len(steps),
            "steps_matched": len(matched), "mean_confidence": round(mean_conf, 4),
            "score": round(coverage * mean_conf, 4),
            "matched_segments": [
                {"step": n, "label": segments[i]["label"],
                 "start_s": segments[i]["start_s"], "end_s": segments[i]["end_s"]}
                for n, i in enumerate(matched)
            ],
        })
    results.sort(key=lambda r: (r["score"], r["coverage"]), reverse=True)
    return results


def print_composite_report(rep, min_coverage):
    print("\n" + "=" * 78)
    print(f"  COMPOSITE ACTIVITY - {rep['clip_id']}")
    print(f"  {rep['clip']['num_frame']} frames @ {rep['clip']['fps']:g} fps "
          f"({rep['clip']['duration_s']:g}s)   "
          f"{rep['stage2']['n_windows']} windows of {rep['stage2']['window']} "
          f"(stride {rep['stage2']['stride']})")
    print("=" * 78)

    print("\n  PRIMITIVE TIMELINE")
    print("  " + "-" * 74)
    if not rep["segments"]:
        print("    (nothing passed the confidence/duration filters)")
    for s in rep["segments"]:
        bar = f"{s['start_s']:>6.1f}s - {s['end_s']:>6.1f}s"
        print(f"    {bar}  {s['label'][:38]:<38} {100*s['confidence']:>4.0f}%"
              f"  x{s['n_windows']}")

    print("\n  COMPOSITE MATCHES")
    print("  " + "-" * 74)
    print(f"    {'activity':<26}{'steps':>8}{'cover':>8}{'conf':>7}{'score':>8}  note")
    for r in rep["composites"]:
        flag = " " if r["coverage"] >= min_coverage else "."
        print(f"  {flag} {r['label'][:26]:<26}"
              f"{r['steps_matched']}/{r['steps_total']:<6}"
              f"{100*r['coverage']:>7.0f}%{100*r['mean_confidence']:>6.0f}%"
              f"{r['score']:>8.3f}  {r['reliability']}")

    best = rep["best"]
    print("\n" + "=" * 78)
    if best:
        print(f"  -> {best['label'].upper()}  "
              f"({best['steps_matched']}/{best['steps_total']} primitives, "
              f"score {best['score']:.3f})")
        for m in best["matched_segments"]:
            print(f"       {m['start_s']:>6.1f}s  {m['label']}")
    else:
        print(f"  -> no composite reached the {100*min_coverage:.0f}% coverage "
              "threshold; see the primitive timeline above")
    print("=" * 78)


def recognize(st, keypoints_path, fps, stride, smooth_k, min_conf, min_windows,
             min_coverage, composites, results_path, clip_id):
    """Run stage 2-4 on an already-built keypoints JSON and print + save a report."""
    with open(keypoints_path, "r", encoding="utf-8") as f:
        hpa = json.load(f)
    raw = keypoints_to_ntu25(hpa)

    starts, probs, window, n_frames = sliding_probs(st, raw, stride)
    print(f"[scenario] stage 2: {len(starts)} windows over {n_frames} frames")

    probs = smooth(probs, smooth_k)
    min_windows_eff = min(min_windows, len(starts))
    if min_windows_eff != min_windows:
        print(f"[scenario] only {len(starts)} window(s) -- relaxing "
              f"--min-windows {min_windows} -> {min_windows_eff}")
    segments = segment(starts, probs, st["label_map"], window, fps,
                       min_conf, min_windows_eff, stride, n_frames)
    print(f"[scenario] stage 3: {len(segments)} primitive segment(s)")

    scored = score_composites(segments, composites)
    best = next((r for r in scored if r["coverage"] >= min_coverage), None)

    rep = {
        "clip_id": clip_id,
        "keypoints_json": str(Path(keypoints_path).resolve()),
        "model": {"arch": "EfficientGCN-B0",
                  "num_class": st["ckpt"]["num_class"], "layout": "ntu-rgb+d"},
        "clip": {"num_frame": n_frames, "fps": fps,
                 "duration_s": round(n_frames / fps, 2)},
        "stage2": {"window": window, "stride": stride, "n_windows": len(starts)},
        "stage3": {"smooth": smooth_k, "min_conf": min_conf,
                   "min_windows": min_windows_eff},
        "segments": segments, "composites": scored, "best": best,
    }
    print_composite_report(rep, min_coverage)

    out = Path(results_path) if results_path else (
        _HERE / "results" / f"{clip_id}_composite.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"\n[scenario] report -> {out}")
    return rep


# ── action-name resolution ──────────────────────────────────────────────────

def label_to_action_map():
    """{lowercased NTU action name: 1-based action id}."""
    return {name.lower(): i + 1 for i, name in enumerate(NTU_60)}


def resolve_action_token(tok, label_to_action):
    """One --actions entry -> a 1-based NTU action id.

    Accepts a plain number (43 -> A043) or the action's own name, matched
    case-insensitively (e.g. "falling", "touch head (headache)", or
    "take_off_jacket" with underscores for spaces).
    """
    if tok.lstrip("-").isdigit():
        return int(tok)
    key = tok.lower().replace("_", " ")
    if key in label_to_action:
        return label_to_action[key]
    raise SystemExit(
        "[scenario] unknown action %r -- give an NTU action number or one of "
        "the names in NTU_60" % tok)


def resolve_preset(name, label_to_action):
    """Expand a --preset keyword into its ordered NTU action list.

    A step with alternatives (e.g. touch chest OR touch head) uses its first
    alternative to produce one concrete sequence; the other options are
    returned as notes so that choice is never silent.
    """
    key = PRESETS[name]
    comp = DEFAULT_COMPOSITES[key]
    actions, notes = [], []
    for step in comp["steps"]:
        label = step[0]
        if label not in label_to_action:
            raise SystemExit(
                "[scenario] preset %r step %r has no matching NTU action"
                % (name, label))
        actions.append(label_to_action[label])
        if len(step) > 1:
            notes.append("%r chosen -- composite also allows: %s"
                        % (label, ", ".join(step[1:])))
    return actions, comp["label"], notes


# ── clip discovery, selection and splicing ──────────────────────────────────

def index_clips(output_dir):
    """{(S,C,P,R): {action_int: path}} over every extracted NTU keypoints file."""
    idx = defaultdict(dict)
    for f in Path(output_dir).glob("*_keypoints_full_body.json"):
        m = NTU_RE.search(f.name)
        if m:
            s, c, p, r, a = m.groups()
            idx[(s, c, p, r)][int(a)] = f
    return idx


def pick_clips(actions, idx, seed):
    """Choose one keypoints file per requested action.

    Prefers a single (setup, camera, person, replication) group containing every
    requested action, so the splice is one actor from one angle. Falls back to
    per-action choice -- still pinned to the most common group where it can --
    when no group covers the whole sequence.
    """
    rng = random.Random(seed)
    wanted = set(actions)

    full = [g for g, acts in idx.items() if wanted <= set(acts)]
    if full:
        g = rng.choice(sorted(full))
        return ([(a, idx[g][a]) for a in actions],
                f"S{g[0]}C{g[1]}P{g[2]}R{g[3]}", len(full))

    ranked = sorted(idx.items(), key=lambda kv: -len(wanted & set(kv[1])))
    best = ranked[0][0] if ranked else None
    chosen = []
    for a in actions:
        if best and a in idx[best]:
            chosen.append((a, idx[best][a]))
            continue
        cands = [p for acts in idx.values() for k, p in acts.items() if k == a]
        if not cands:
            raise SystemExit(f"[scenario] no extracted clip for action A{a:03d}")
        chosen.append((a, rng.choice(sorted(cands))))
    return chosen, "mixed", 0


def splice_keypoints(chosen, out_path, fps=30.0):
    """Concatenate the clips' frames into one keypoints JSON.

    Every frame is re-stamped with its own sequential index. (Getting this wrong
    -- advancing the counter per clip instead of per frame -- silently collapses
    the whole timeline onto a handful of indices and the model then sees a
    nearly empty array, so it is worth being explicit about.)
    """
    frames, plan, meta = [], [], None
    for action, path in chosen:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if meta is None:
            meta = dict(d["metadata"])
        start = len(frames)
        for fr in d["frames"]:
            g = dict(fr)
            g["frame_id"] = len(frames)
            g["timestamp_s"] = round(len(frames) / fps, 4)
            frames.append(g)
        plan.append({"action": action, "clip": Path(path).name,
                     "start_frame": start, "end_frame": len(frames),
                     "start_s": round(start / fps, 2),
                     "end_s": round(len(frames) / fps, 2)})

    meta.update({
        "source": "build_scenario_clip.py splice of " + ", ".join(
            f"A{a:03d}" for a, _ in chosen),
        "source_file": out_path.stem.replace("_keypoints_full_body", ""),
        "total_frames_processed": len(frames),
        "total_detections": sum(1 for f in frames if f.get("detected")),
        "source_fps": fps,
    })
    out_path.write_text(json.dumps({"metadata": meta, "frames": frames}),
                        encoding="utf-8")
    return plan, len(frames)


def find_video(clip_name):
    """Locate the .avi behind a keypoints filename, across the dataset roots."""
    stem = clip_name.replace("_keypoints_full_body.json", "")
    for root in VIDEO_ROOTS:
        base = IPCEI_ROOT / root
        if not base.is_dir():
            continue
        hits = list(base.glob(f"**/{stem}.avi"))
        if hits:
            return hits[0]
    return None


def splice_video(chosen, out_path, extract_python):
    """Concatenate the source .avi files into one .mp4 (needs cv2, a different
    interpreter -- unavoidable for the same reason torch/numpy are: it's a
    third-party library, run in a separate process, not a project import)."""
    vids = []
    for _, kp in chosen:
        v = find_video(Path(kp).name)
        if v is None:
            print(f"[scenario] no source video for {Path(kp).name} -- skipping "
                  "video output")
            return None
        vids.append(str(v))

    script = r'''
import cv2, sys, json
paths = json.loads(sys.argv[1]); out = sys.argv[2]
caps = [cv2.VideoCapture(p) for p in paths]
w = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = caps[0].get(cv2.CAP_PROP_FPS) or 30.0
wr = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
n = 0
for c in caps:
    while True:
        ok, fr = c.read()
        if not ok: break
        if (fr.shape[1], fr.shape[0]) != (w, h):
            fr = cv2.resize(fr, (w, h))
        wr.write(fr); n += 1
    c.release()
wr.release()
print("FRAMES=%d" % n)
'''
    r = subprocess.run([extract_python, "-c", script, json.dumps(vids),
                        str(out_path)], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[scenario] video splice failed:\n{r.stderr[-400:]}")
        return None
    return out_path


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--actions", nargs="+", metavar="A",
                   help="actions IN ORDER -- NTU numbers (15 8 9) or action "
                        "names (\"take off jacket\" \"sitting down\"); give "
                        "exactly one of --actions / --preset")
    p.add_argument("--preset", choices=sorted(PRESETS), metavar="NAME",
                   help="a named scenario instead of spelling out --actions "
                        "(choices: %s)" % ", ".join(sorted(PRESETS)))
    p.add_argument("--name", help="output basename (default: seq_A015_A008_... "
                                 "or the preset name)")
    p.add_argument("--seed", type=int, default=None,
                   help="clip-picking seed. Omitted = a fresh random draw every "
                        "run; the seed used is printed so any run can be "
                        "reproduced by passing it back in")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--video", action="store_true",
                   help="also write a real concatenated .mp4 (needs cv2)")
    p.add_argument("--extract-python", default=DEFAULT_EXTRACT_PYTHON,
                   help="interpreter with mediapipe+opencv, for --video")
    p.add_argument("--run", action="store_true",
                   help="recognise composite activity in the spliced clip "
                        "(runs the EfficientGCN model in-process -- no "
                        "subprocess, no other script)")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="EfficientGCN checkpoint (default: %(default)s)")
    p.add_argument("--device", default="cpu", help="cpu | cuda | auto")
    p.add_argument("--composites", help="JSON file of composite definitions "
                                       "(default: the 9 built-in scenarios)")
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--smooth", type=int, default=3)
    p.add_argument("--min-conf", type=float, default=0.5)
    p.add_argument("--min-windows", type=int, default=2)
    p.add_argument("--min-coverage", type=float, default=0.6)
    p.add_argument("--results", help="where to write the recognition report "
                                    "JSON (default: results/<name>_composite.json)")
    args = p.parse_args()
    if bool(args.actions) == bool(args.preset):
        p.error("give exactly one of --actions or --preset")

    label_to_action = label_to_action_map()
    if args.preset:
        actions, preset_label, notes = resolve_preset(args.preset, label_to_action)
        print(f"[scenario] preset {args.preset!r} -> {preset_label}: "
              + " -> ".join(f"A{a:03d} {NTU_60[a-1]}" for a in actions))
        for n in notes:
            print(f"[scenario]   {n}")
    else:
        actions = [resolve_action_token(a, label_to_action) for a in args.actions]

    idx = index_clips(HPA_OUTPUT)
    if not idx:
        raise SystemExit(f"[scenario] no extracted NTU keypoints in {HPA_OUTPUT}")

    seed = args.seed if args.seed is not None else random.randrange(1 << 30)
    chosen, group, n_cand = pick_clips(actions, idx, seed)
    name = args.name or args.preset or "seq_" + "_".join(f"A{a:03d}" for a in actions)
    out_kp = HPA_OUTPUT / f"{name}_keypoints_full_body.json"

    plan, n_frames = splice_keypoints(chosen, out_kp, args.fps)

    variety = (f"{n_cand} group(s) had all {len(actions)} actions"
               if n_cand else "no single group had them all -- mixed sources")
    print(f"[scenario] group={group}  seed={seed}  {len(chosen)} clip(s)  "
          f"{n_frames} frames ({n_frames/args.fps:.1f}s)")
    print(f"[scenario] {variety}")
    print(f"[scenario] GROUND TRUTH timeline:")
    for step in plan:
        a = step["action"]
        nm = NTU_60[a - 1] if a - 1 < len(NTU_60) else "?"
        dur = step["end_s"] - step["start_s"]
        warn = "   <-- shorter than the 128f window" if dur < 4.3 else ""
        src = step["clip"].replace("_rgb_keypoints_full_body.json", "")
        print(f"    {step['start_s']:>6.1f}s - {step['end_s']:>6.1f}s  "
              f"A{a:03d} {nm[:30]:<30} ({dur:.1f}s)  {src}{warn}")
    print(f"[scenario] keypoints -> {out_kp}")
    print(f"[scenario] reproduce this exact draw with:  --seed {seed}")
    (out_kp.with_name(f"{name}_truth.json")).write_text(
        json.dumps({"group": group, "seed": seed, "actions": actions,
                    "preset": args.preset, "plan": plan, "n_frames": n_frames,
                    "fps": args.fps}, indent=2),
        encoding="utf-8")

    if args.video:
        out_v = _HERE / "sequences" / f"{name}_vid.mp4"
        out_v.parent.mkdir(parents=True, exist_ok=True)
        if splice_video(chosen, out_v, args.extract_python):
            print(f"[scenario] video -> {out_v}")

    if args.run:
        composites = DEFAULT_COMPOSITES
        if args.composites:
            composites = json.loads(Path(args.composites).read_text(encoding="utf-8"))
        print(f"\n[scenario] stage 2: loading {Path(args.model).name} ...")
        st = load_effgcn(args.model, args.device)
        print(f"[scenario] EfficientGCN-B0  {st['ckpt']['num_class']} classes")
        recognize(st, out_kp, args.fps, args.stride, args.smooth, args.min_conf,
                 args.min_windows, args.min_coverage, composites, args.results, name)


if __name__ == "__main__":
    main()
