"""
run_pipeline.py
===============

One command, both stages: **video in, prediction-vs-ground-truth report out**.

    python run_pipeline.py ../nturgb+d_rgb/S001C001P001R001A009_rgb.avi
    python run_pipeline.py ../nturgb+d_rgb/*.avi --model all      # benchmark

Stage 1  video -> MediaPipe keypoints   (HumanPoseAction/pose_detection.py)
Stage 2  keypoints -> ST-GCN action     (stgcn_infer.py + pretrained checkpoint)
Report   ground truth vs each model's top-k -> results/<clip>.json
                                            -> results/summary.json (aggregate)

The two stages need different dependencies (stage 1 needs mediapipe+opencv,
stage 2 needs torch+numpy) and in this workspace they do not live in the same
interpreter.  Stage 2 runs in-process, so run *this* script with the torch env
(`../env/Scripts/python.exe`); stage 1 is shelled out to DEFAULT_EXTRACT_PYTHON
below (the interpreter confirmed to have mediapipe+opencv), or to whatever you
pass via `--extract-python`.

Stage 1 is skipped when the keypoints JSON already exists, so re-running a clip
is cheap and clips that were extracted earlier work without mediapipe present.

Ground truth is read from the NTU RGB+D filename (`...A009_rgb.avi` -> action
009 -> "standing up").  Non-NTU filenames still run; the report just records the
ground truth as unknown and skips the correctness fields.

Scoring caveat: only the NTU checkpoints (`ntu-xsub`, `ntu-xview`) predict into
the same 60-class space the ground truth lives in.  `kinetics` predicts 400
Kinetics classes, so its prediction is reported but left unscored -- its class
ids mean something different and cannot be compared to an NTU action id.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
IPCEI_ROOT = _HERE.parent
HPA_ROOT = IPCEI_ROOT / "HumanPoseAction"

# Interpreter used for stage 1 (needs mediapipe+opencv). Override per-run with
# --extract-python. This is the machine's system Python 3.11, the one confirmed
# to have both packages installed.
DEFAULT_EXTRACT_PYTHON = (
    r"C:\Users\alext\AppData\Local\Programs\Python\Python311\python.exe"
)

sys.path.insert(0, str(_HERE))

import HumanPoseAction.inter_pipeline.models.build_dataset as build_dataset          # ntu_category_from_name
import HumanPoseAction.inter_pipeline.models.labels as label_maps

ALL_MODELS = ["ntu-xsub", "ntu-xview", "kinetics"]


# ── stage 1: video -> keypoints ────────────────────────────────────────────────

def keypoints_path_for(video, output_dir, mode="body"):
    """Mirror pose_detection.py's own output naming (`run()`, near its end)."""
    return Path(output_dir) / f"{Path(video).stem}_keypoints_full_{mode}.json"


def extract_keypoints(video, output_dir, mode="body", max_size=600,
                      confidence=0.5, extract_python=None, force=False):
    """Run HumanPoseAction on `video`; return the keypoints JSON path."""
    out_json = keypoints_path_for(video, output_dir, mode)

    if out_json.exists() and not force:
        print(f"[stage1] keypoints already exist, skipping extraction: {out_json}")
        return out_json

    py = extract_python or DEFAULT_EXTRACT_PYTHON
    if not Path(py).exists():
        raise SystemExit(
            f"[stage1] ERROR: extract-python interpreter not found: {py}\n"
            "         Pass a valid one with --extract-python <path-to-python.exe>\n"
            "         (it must have mediapipe+opencv installed)."
        )
    print(f"[stage1] extracting keypoints with: {py}")

    cmd = [
        py, "pose_detection.py",
        "--input", str(Path(video).resolve()),
        "--mode", mode,
        "--output", str(Path(output_dir).resolve()),
        "--no-display",
        "--max-size", str(max_size),
        "--confidence", str(confidence),
    ]
    # cwd=HPA_ROOT: pose_detection.py imports its sibling packages (utils, config,
    # detector) by relative name, so it only resolves from the repo root.
    proc = subprocess.run(cmd, cwd=str(HPA_ROOT))
    if proc.returncode != 0:
        raise SystemExit(f"[stage1] ERROR: pose_detection.py failed "
                         f"(exit {proc.returncode})")
    if not out_json.exists():
        raise SystemExit(f"[stage1] ERROR: expected output not written: {out_json}")

    print(f"[stage1] keypoints -> {out_json}")
    return out_json


# ── ground truth ───────────────────────────────────────────────────────────────

def ground_truth_for(video):
    """NTU action id + name parsed out of the filename."""
    action_id = build_dataset.ntu_category_from_name(Path(video).name)
    if action_id is None or not (0 <= action_id < len(label_maps.NTU_60)):
        return {"known": False, "source": None, "action_id": None, "label": None}
    return {
        "known": True,
        "source": "ntu-filename",
        "action_id": action_id,
        "label": label_maps.NTU_60[action_id],
    }


# ── report ─────────────────────────────────────────────────────────────────────

def score_model(probs, gt, label_set, topk):
    """One model's ranked predictions, scored against `gt` when comparable."""
    names = label_maps.labels_for(label_set) or []
    ranked = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)

    def name_of(idx):
        return names[idx] if idx < len(names) else f"class_{idx}"

    top = [
        {"rank": r, "action_id": i, "label": name_of(i), "prob": round(probs[i], 6)}
        for r, i in enumerate(ranked[:topk], 1)
    ]

    # A kinetics checkpoint predicts into a different 400-class space, so its
    # class ids carry no relation to an NTU ground-truth id.
    comparable = gt["known"] and label_set == "ntu-rgb+d"
    correctness = None
    if comparable:
        gt_rank = ranked.index(gt["action_id"]) + 1
        correctness = {
            "top1": gt_rank == 1,
            "topk": gt_rank <= topk,
            "k": topk,
            "gt_rank": gt_rank,
            "gt_prob": round(probs[gt["action_id"]], 6),
        }

    return {
        "label_set": label_set,
        "comparable_to_ground_truth": comparable,
        "prediction": top[0] if top else None,
        "topk": top,
        "correctness": correctness,
    }


def load_custom_model(ckpt_path):
    """Load a checkpoint trained by train_stgcn.py (self-describing: carries its
    own num_class, layout, and label_map). Returns a dict ready for scoring."""
    import HumanPoseAction.inter_pipeline.models.stgcn_infer as stgcn_infer
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    layout = ckpt["layout"]
    num_class = ckpt["num_class"]
    label_map = ckpt.get("label_map")  # {str(class_id): {ntu_action_id, label}}
    ST_GCN_18 = stgcn_infer.load_stgcn_model_class()
    model = ST_GCN_18(
        in_channels=3,
        num_class=num_class,
        edge_importance_weighting=True,
        graph_cfg={"layout": layout, "strategy": "spatial"},
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return {"model": model, "layout": layout, "num_class": num_class,
            "label_map": label_map, "val_top1": ckpt.get("val_top1")}


def score_custom(probs, gt, label_map, topk):
    """Score a custom model's output against NTU ground truth.

    The model only knows the actions in label_map (a subset of NTU-60). A clip
    whose true action is outside that subset can never be predicted correctly,
    so it's marked out-of-vocabulary rather than counted as a plain miss.
    """
    ranked = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)

    def entry_for(class_id, rank):
        meta = (label_map or {}).get(str(class_id), {})
        return {"rank": rank, "class_id": class_id,
                "label": meta.get("label", f"class_{class_id}"),
                "ntu_action_id": meta.get("ntu_action_id"),
                "prob": round(probs[class_id], 6)}

    top = [entry_for(i, r) for r, i in enumerate(ranked[:topk], 1)]

    # Set of 0-based NTU action ids this model can output.
    vocab = {m["ntu_action_id"] - 1 for m in (label_map or {}).values()
             if m.get("ntu_action_id") is not None}
    in_vocab = gt["known"] and gt["action_id"] in vocab

    correctness, unscored_reason = None, None
    if not gt["known"]:
        unscored_reason = "ground truth unknown (non-NTU filename)"
    elif not in_vocab:
        unscored_reason = "ground truth action not among the model's classes"
    else:
        # rank of the model class whose NTU id == the ground-truth action
        gt_class = next(int(c) for c, m in label_map.items()
                        if m.get("ntu_action_id") == gt["action_id"] + 1)
        gt_rank = ranked.index(gt_class) + 1
        correctness = {
            "top1": gt_rank == 1,
            "topk": gt_rank <= topk,
            "k": topk,
            "gt_rank": gt_rank,
            "gt_prob": round(probs[gt_class], 6),
        }

    return {
        "label_set": "custom",
        "comparable_to_ground_truth": in_vocab,
        "unscored_reason": unscored_reason,
        "prediction": top[0] if top else None,
        "topk": top,
        "correctness": correctness,
    }


def print_summary(rep):
    gt = rep["ground_truth"]
    print("\n" + "=" * 74)
    print(f"  clip:          {rep['clip_id']}")
    print(f"  frames:        {rep['clip']['num_detected']}/{rep['clip']['num_frame']} detected")
    if gt["known"]:
        print(f"  ground truth:  {gt['label']}  (A{gt['action_id'] + 1:03d})")
    else:
        print("  ground truth:  unknown (filename is not NTU-style)")
    for name, res in rep["models"].items():
        corr = res["correctness"]
        if corr:
            verdict = "CORRECT" if corr["top1"] else f"miss (GT #{corr['gt_rank']})"
        else:
            verdict = "unscored: " + res.get("unscored_reason",
                                              "different label space")
        print("-" * 74)
        print(f"  {name}  [{res['layout']}]  ->  {verdict}")
        for entry in res["topk"]:
            # Pretrained entries carry a 0-based NTU 'action_id'; custom-model
            # entries carry a 1-based 'ntu_action_id'. Normalize to 0-based to
            # compare against the ground truth.
            if "action_id" in entry:
                entry_ntu = entry["action_id"]
            elif entry.get("ntu_action_id") is not None:
                entry_ntu = entry["ntu_action_id"] - 1
            else:
                entry_ntu = None
            marker = " <-- ground truth" if (
                gt["known"] and entry_ntu == gt["action_id"]) else ""
            print(f"    {entry['rank']:>2}. {entry['label'][:44]:<44} "
                  f"{100 * entry['prob']:>5.1f}%{marker}")
    print("=" * 74)


# ── main ───────────────────────────────────────────────────────────────────────

def analyze(video, models, output_dir, results_dir, topk=5, mode="body",
            max_size=600, confidence=0.5, extract_python=None, force=False,
            num_track=2, max_frames=None, model_cache=None, custom_models=None):
    import HumanPoseAction.inter_pipeline.models.stgcn_infer as stgcn_infer  # deferred: pulls torch, and stage 1 must be able to fail first

    model_cache = {} if model_cache is None else model_cache
    custom_models = custom_models or []

    keypoints_json = extract_keypoints(
        video, output_dir, mode=mode, max_size=max_size, confidence=confidence,
        extract_python=extract_python, force=force,
    )

    gt = ground_truth_for(video)
    results, info = {}, {}

    for name in models:
        ckpt_name, layout, label_set = stgcn_infer.MODELS[name]
        checkpoint = Path(stgcn_infer._MMSKELETON_ROOT) / "checkpoints" / ckpt_name
        if not checkpoint.exists():
            raise SystemExit(f"[main] ERROR: checkpoint not found: {checkpoint}")

        if name not in model_cache:
            model_cache[name] = stgcn_infer.load_model(str(checkpoint), layout)

        probs, info = stgcn_infer.predict(
            str(keypoints_json), str(checkpoint), layout,
            num_track=num_track, max_frames=max_frames, model=model_cache[name],
        )
        results[name] = score_model(probs, gt, label_set, topk)
        results[name]["checkpoint"] = ckpt_name
        results[name]["layout"] = layout

    for ckpt_path in custom_models:
        name = Path(ckpt_path).stem
        if name not in model_cache:
            model_cache[name] = load_custom_model(ckpt_path)
        spec = model_cache[name]
        probs, info = stgcn_infer.predict(
            str(keypoints_json), ckpt_path, spec["layout"],
            num_track=num_track, max_frames=max_frames, model=spec["model"],
        )
        results[name] = score_custom(probs, gt, spec["label_map"], topk)
        results[name]["checkpoint"] = Path(ckpt_path).name
        results[name]["layout"] = spec["layout"]

    rep = {
        "clip_id": Path(video).stem,
        "video": str(Path(video).resolve()),
        "keypoints_json": str(Path(keypoints_json).resolve()),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "clip": {
            "num_frame": info.get("num_frame"),
            "num_detected": info.get("num_detected"),
        },
        "ground_truth": gt,
        "models": results,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"{rep['clip_id']}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)

    print_summary(rep)
    print(f"\n[main] report -> {out}")
    return rep


def write_aggregate(reports, models, results_dir, topk):
    """Per-model accuracy across every scoreable clip -> results/summary.json."""
    per_model = {}
    for name in models:
        scored = [r for r in reports if r["models"][name]["correctness"]]
        if not scored:
            per_model[name] = {"scored_clips": 0, "note": "no clips comparable "
                               "to ground truth (different label space)"}
            continue
        top1 = sum(r["models"][name]["correctness"]["top1"] for r in scored)
        topk_hits = sum(r["models"][name]["correctness"]["topk"] for r in scored)
        per_model[name] = {
            "scored_clips": len(scored),
            "top1_correct": top1,
            "top1_accuracy": round(top1 / len(scored), 4),
            f"top{topk}_correct": topk_hits,
            f"top{topk}_accuracy": round(topk_hits / len(scored), 4),
            "mean_gt_rank": round(
                sum(r["models"][name]["correctness"]["gt_rank"] for r in scored)
                / len(scored), 2),
        }

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "num_clips": len(reports),
        "models": models,
        "topk": topk,
        "accuracy": per_model,
        "clips": [
            {
                "clip_id": r["clip_id"],
                "ground_truth": r["ground_truth"]["label"],
                "predictions": {
                    n: {
                        "label": r["models"][n]["prediction"]["label"],
                        "prob": r["models"][n]["prediction"]["prob"],
                        "gt_rank": (r["models"][n]["correctness"]["gt_rank"]
                                    if r["models"][n]["correctness"] else None),
                    }
                    for n in models
                },
            }
            for r in reports
        ],
    }

    out = results_dir / "summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 74)
    print(f"  BENCHMARK: {len(reports)} clips")
    print("-" * 74)
    print(f"  {'model':<11} {'scored':>7} {'top-1':>14} {'top-' + str(topk):>14} {'mean GT rank':>14}")
    for name, acc in per_model.items():
        if not acc.get("scored_clips"):
            print(f"  {name:<11} {'0':>7}   {'unscored (different label space)':>40}")
            continue
        print(f"  {name:<11} {acc['scored_clips']:>7} "
              f"{acc['top1_correct']:>4}/{acc['scored_clips']:<3} "
              f"({100 * acc['top1_accuracy']:>4.0f}%) "
              f"{acc[f'top{topk}_correct']:>4}/{acc['scored_clips']:<3} "
              f"({100 * acc[f'top{topk}_accuracy']:>4.0f}%) "
              f"{acc['mean_gt_rank']:>13}")
    print("=" * 74)
    print(f"\n[main] summary -> {out}")
    return summary


def _cli():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", nargs="+", help="video file(s) to analyze")
    p.add_argument("--model", nargs="+", default=None,
                   choices=ALL_MODELS + ["all"],
                   help="pretrained checkpoints to run; 'all' runs every one "
                        "(default: ntu-xsub, unless only --custom-model is given)")
    p.add_argument("--custom-model", nargs="+", metavar="PATH", default=None,
                   help="path(s) to a model trained by train_stgcn.py "
                        "(e.g. models/humanpose5_coco.pth); scored against NTU "
                        "ground truth, with out-of-vocabulary clips flagged")
    p.add_argument("--output-dir", help="where keypoint JSONs go "
                                        "(default: HumanPoseAction/output)")
    p.add_argument("--results-dir", help="where report JSONs go "
                                         "(default: pipeline/results)")
    p.add_argument("--topk", type=int, default=5,
                   help="how many top predictions to score and print per model "
                        "(default: 5)")
    p.add_argument("--mode", default="body", choices=["body", "face", "hands"],
                   help="HumanPoseAction detection mode (default: body)")
    p.add_argument("--max-size", type=int, default=600,
                   help="longest frame edge fed to MediaPipe (default: 600)")
    p.add_argument("--confidence", type=float, default=0.5)
    p.add_argument("--extract-python", default=None,
                   help="interpreter with mediapipe+opencv for stage 1 "
                        f"(default: {DEFAULT_EXTRACT_PYTHON})")
    p.add_argument("--force", action="store_true",
                   help="re-extract keypoints even if the JSON already exists")
    p.add_argument("--num-track", type=int, default=2)
    p.add_argument("--max-frames", type=int, default=None)
    args = p.parse_args()

    custom_models = args.custom_model or []
    for cm in custom_models:
        if not Path(cm).exists():
            raise SystemExit(f"[main] ERROR: custom model not found: {cm}")

    # Default to ntu-xsub only when neither --model nor --custom-model is given;
    # if the user asked for just a custom model, don't silently add pretrained ones.
    if args.model:
        models = ALL_MODELS if "all" in args.model else list(dict.fromkeys(args.model))
    else:
        models = [] if custom_models else ["ntu-xsub"]

    output_dir = Path(args.output_dir or (HPA_ROOT / "output"))
    results_dir = Path(args.results_dir or (_HERE / "results"))

    for v in args.video:
        if not Path(v).exists():
            raise SystemExit(f"[main] ERROR: video not found: {v}")

    model_cache = {}
    reports = [
        analyze(v, models, output_dir, results_dir, topk=args.topk,
                mode=args.mode, max_size=args.max_size,
                confidence=args.confidence, extract_python=args.extract_python,
                force=args.force, num_track=args.num_track,
                max_frames=args.max_frames, model_cache=model_cache,
                custom_models=custom_models)
        for v in args.video
    ]

    if len(reports) > 1:
        all_names = models + [Path(cm).stem for cm in custom_models]
        write_aggregate(reports, all_names, results_dir, args.topk)


if __name__ == "__main__":
    _cli()
