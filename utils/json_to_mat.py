"""
Convert project JSON keypoint exports to MATLAB .mat files.

Handles three formats produced by this project:
  - single-frame  : top-level "keypoints" dict  (e.g. test_pic_keypoints.json)
  - multi-frame   : top-level "frames" list      (e.g. campus_keypoints_full.json)
  - metrics       : top-level "metrics" list     (e.g. dance2_keypoints_full_metrics.json)

Multi-frame output layout (most useful for MATLAB analysis):
  data.frame_ids     (F,)    frame indices
  data.timestamps    (F,)    seconds
  data.detected      (F,)    uint8 boolean
  data.keypoint_names        cell array, length K
  data.x_norm        (F, K)  normalised x  [0, 1]
  data.y_norm        (F, K)  normalised y  [0, 1]
  data.z_norm        (F, K)  depth (relative)
  data.x_px          (F, K)  pixel x
  data.y_px          (F, K)  pixel y
  data.visibility    (F, K)  MediaPipe visibility score
  data.metadata              struct with source info

Usage:
  python utils/json_to_mat.py output/campus_keypoints_full.json
  python utils/json_to_mat.py output/campus_keypoints_full.json -o my_output.mat
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io


# ── helpers ──────────────────────────────────────────────────────────────────

def _kp_arrays(kp_dict: dict) -> dict:
    """Turn a keypoints dict into parallel numpy arrays, sorted by landmark id."""
    items = sorted(kp_dict.items(), key=lambda x: x[1]["id"])
    names = [name for name, _ in items]
    fields = {f: [] for f in ("x_norm", "y_norm", "z_norm", "x_px", "y_px", "visibility")}
    for _, v in items:
        for f in fields:
            fields[f].append(v.get(f, np.nan))
    return {"names": names, **{f: np.array(v) for f, v in fields.items()}}


def _metadata_struct(meta: dict) -> dict:
    """Convert metadata dict to a scipy-compatible struct (scalar numpy values)."""
    out = {}
    for k, v in meta.items():
        if isinstance(v, bool):
            out[k] = np.uint8(v)
        elif isinstance(v, int):
            out[k] = np.int64(v)
        elif isinstance(v, float):
            out[k] = np.float64(v)
        else:
            out[k] = str(v)
    return out


# ── converters ───────────────────────────────────────────────────────────────

def convert_single_frame(data: dict) -> dict:
    kp = _kp_arrays(data["keypoints"])
    mat = {
        "keypoint_names": np.array(kp["names"], dtype=object),
        "x_norm":         kp["x_norm"],
        "y_norm":         kp["y_norm"],
        "z_norm":         kp["z_norm"],
        "x_px":           kp["x_px"],
        "y_px":           kp["y_px"],
        "visibility":     kp["visibility"],
        "metadata":       _metadata_struct(data.get("metadata", {})),
    }
    return mat


def convert_multi_frame(data: dict) -> dict:
    frames = data["frames"]
    n_frames = len(frames)

    # Determine keypoint list from first detected frame
    ref_frame = next((f for f in frames if f.get("detected") and f.get("keypoints")), None)
    if ref_frame is None:
        sys.exit("No detected frames found in JSON.")

    ref_kp = _kp_arrays(ref_frame["keypoints"])
    kp_names = ref_kp["names"]
    n_kp = len(kp_names)
    fields = ("x_norm", "y_norm", "z_norm", "x_px", "y_px", "visibility")

    frame_ids  = np.zeros(n_frames, dtype=np.int64)
    timestamps = np.zeros(n_frames, dtype=np.float64)
    detected   = np.zeros(n_frames, dtype=np.uint8)
    arrays     = {f: np.full((n_frames, n_kp), np.nan) for f in fields}

    for i, frame in enumerate(frames):
        frame_ids[i]  = frame.get("frame_id", i)
        timestamps[i] = frame.get("timestamp_s", np.nan)
        det = bool(frame.get("detected", False))
        detected[i]   = np.uint8(det)
        if det and frame.get("keypoints"):
            kp = _kp_arrays(frame["keypoints"])
            for f in fields:
                arrays[f][i] = kp[f]

    mat = {
        "frame_ids":      frame_ids,
        "timestamps":     timestamps,
        "detected":       detected,
        "keypoint_names": np.array(kp_names, dtype=object),
        "metadata":       _metadata_struct(data.get("metadata", {})),
    }
    mat.update(arrays)
    return mat


def convert_metrics(data: dict) -> dict:
    metrics = data["metrics"]
    n = len(metrics)

    # Collect all metric fields (excluding frame_id, timestamp_s, detected)
    skip = {"frame_id", "timestamp_s", "detected"}
    metric_keys = [k for k in metrics[0] if k not in skip]

    frame_ids  = np.array([m.get("frame_id", i) for i, m in enumerate(metrics)], dtype=np.int64)
    timestamps = np.array([m.get("timestamp_s", np.nan) for m in metrics], dtype=np.float64)
    detected   = np.array([np.uint8(m.get("detected", False)) for m in metrics], dtype=np.uint8)

    arrays = {}
    for k in metric_keys:
        arrays[k] = np.array(
            [np.nan if m.get(k) is None else m[k] for m in metrics],
            dtype=np.float64,
        )

    mat = {
        "frame_ids":  frame_ids,
        "timestamps": timestamps,
        "detected":   detected,
        "metadata":   _metadata_struct(data.get("metadata", {})),
    }
    mat.update(arrays)
    return mat


# ── detect format ─────────────────────────────────────────────────────────────

def detect_format(data: dict) -> str:
    if "frames" in data:
        return "multi_frame"
    if "metrics" in data:
        return "metrics"
    if "keypoints" in data:
        return "single_frame"
    sys.exit("Unrecognised JSON format: expected 'frames', 'metrics', or 'keypoints' key.")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert keypoint JSON to MATLAB .mat")
    parser.add_argument("input", help="Path to input JSON file")
    parser.add_argument("-o", "--output", help="Output .mat path (default: same name, .mat extension)")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"File not found: {in_path}")

    out_path = Path(args.output) if args.output else in_path.with_suffix(".mat")

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fmt = detect_format(data)
    converters = {
        "single_frame": convert_single_frame,
        "multi_frame":  convert_multi_frame,
        "metrics":      convert_metrics,
    }
    mat = converters[fmt](data)

    scipy.io.savemat(str(out_path), mat)
    print(f"[{fmt}] {in_path.name}  ->  {out_path}")

    if fmt == "multi_frame":
        n_frames = len(mat["frame_ids"])
        n_kp = len(mat["keypoint_names"])
        n_det = int(mat["detected"].sum())
        print(f"  {n_frames} frames, {n_kp} keypoints, {n_det} detected")


if __name__ == "__main__":
    main()
