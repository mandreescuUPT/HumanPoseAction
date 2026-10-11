"""
trim_keypoints.py
=================

Cut the idle lead-in / lead-out off a HumanPoseAction keypoints JSON, keeping
only the part where the body is actually moving.

    python trim_keypoints.py --json clip_keypoints_full_body.json
    python trim_keypoints.py --video my_clip.mp4
    python trim_keypoints.py --batch videos_folder/
    python trim_keypoints.py --batch_json ../HumanPoseAction/output/
    python trim_keypoints.py --json clip.json --dry-run --view-trim

Exactly one input mode, picking what gets trimmed:

    --json PATH         one existing keypoints JSON
    --batch_json FOLDER  every keypoints JSON in FOLDER (*_keypoints_full_body.json)
    --video PATH         one video (.mp4/.avi/.mov/.mkv) -- runs MediaPipe
                         extraction first, then trims the result, then ALSO
                         writes a trimmed .mp4 (the source footage is known
                         exactly, no searching needed)
    --batch FOLDER       every video in FOLDER, same as --video for each one

Output lands in trimmed/ by default; --out-dir overrides that for one run.
--view-trim opens view_trim.py on whatever was just processed, so the trim can
be checked visually without a second command.

Writes two files per clip (never touching the input):

    <name>_trimmed.json   same schema as the input, so every downstream tool
                          (build_dataset.py, recognize_composite.py, ...) reads
                          it unchanged -- only shorter
    <name>_trim.json      what was cut: frame indices, timestamps, seconds
                          removed from each end, and why


WHY MOTION, NOT VALUE-EQUALITY
------------------------------
Keypoints are floats from a vision model: they are never bitwise constant, they
jitter by ~1e-4 even when the subject is perfectly still. So "constant" has to
mean "below a motion threshold", not "equal to the previous frame".

The measure is per-frame DYNAMISM -- the mean absolute displacement of each
keypoint between consecutive frames, in normalised (0-1) coordinates:

    dyn[t] = mean_over_joints( |x[t] - x[t-1]| + |y[t] - y[t-1]| )

Only joints visible in BOTH frames are counted. An occluded joint flickering in
and out would otherwise register as a huge fake displacement and hide the fact
that the subject is standing still.


THE THRESHOLD IS RELATIVE, NOT ABSOLUTE
---------------------------------------
A fall moves far more than a finger point, so no single fixed cutoff serves
both. The default threshold is a fraction of each clip's OWN peak dynamism
(--rel, default 0.20), which self-scales per action. --abs overrides it with a
fixed value for when cross-clip comparability matters more than per-clip fit.


ENDS ONLY, NEVER THE MIDDLE
---------------------------
Trimming scans inward from each end and stops at the first frame above
threshold. A quiet stretch in the MIDDLE is deliberately left alone: mid-action
pauses are part of the action (the stillness between a stagger and a fall), and
removing them would splice together two moments that were never adjacent.


THE HOLD GUARD -- why the naive version breaks
----------------------------------------------
Measured on this dataset: "touch head" has ~51% of its frames quiet at the TAIL,
"pointing" ~41%. That is not dead time. Those actions are *raise the arm, then
HOLD the pose* -- the still part IS the evidence. Cutting it drops touch-head
classification from 10/10 to 4/10.

So before trimming a quiet tail, compare the average POSE during that tail
against the average pose during the quiet head:

    poses similar  ->  the subject returned to rest; the tail is real dead time
    poses differ   ->  the subject is holding something; KEEP the tail

--no-hold-guard disables this and gives the naive behaviour.


A NOTE ON WHERE THIS HELPS
--------------------------
Trimming measurably HURTS inference with a model trained on untrimmed clips
(98% -> 90.5% top-1 on this dataset), because the model learned from clips that
include lead-in and lead-out. The place trimming can pay off is preprocessing a
RETRAIN: trim the training set and the inference input consistently, and there
is no distribution mismatch to pay for.
"""

import argparse
import json
import subprocess
from pathlib import Path

VIS = 0.5           # a joint counts only when at least this visible in both frames
HOLD_DIST = 0.04    # normalised pose distance above which a tail reads as a hold

_HERE = Path(__file__).resolve().parent
IPCEI_ROOT = _HERE.parent
# Where the source footage lives, when a keypoints JSON's own source video
# needs to be searched for (find_source_video(), used by view_trim.py's
# playback and by trim_file() when no source_video is already known). Includes
# both sequence locations: pipeline/sequences (older synthetic sequences) and
# HumanPoseAction/sequences (build_scenario_clip.py's default, now that it
# lives here too).
VIDEO_ROOTS = ["nturgb+d_rgb", "nturgb+d_rgb_1", "nturgb+d_rgb_2",
               "ntu_folositor", "pipeline/sequences", "HumanPoseAction/sequences"]
VIDEO_EXT = (".avi", ".mp4", ".mov", ".mkv")
# The interpreter with mediapipe+opencv, for --video/--batch extraction and the
# cv2 subprocess in write_trimmed_video. Same literal value as run_pipeline.py's
# own DEFAULT_EXTRACT_PYTHON, kept local so this file never needs to import
# run_pipeline just to read one constant (its unrelated top-level imports can
# break this file otherwise -- see extract_keypoints_for_video below).
DEFAULT_EXTRACT_PYTHON = (
    r"C:\Users\alext\AppData\Local\Programs\Python\Python311\python.exe"
)
DEFAULT_OUT_DIR = _HERE / "trimmed"   # where output lands unless --out-dir overrides it


def frame_dynamism(frames):
    """Per-frame mean absolute keypoint displacement, in normalised units."""
    prev, dyn = None, []
    for f in frames:
        kp = f.get("keypoints") or {}
        if not f.get("detected") or not kp:
            dyn.append(0.0)
            prev = None                    # break the chain; no fake jump after a gap
            continue
        cur = {k: (v["x_norm"], v["y_norm"], v.get("visibility", 1.0))
               for k, v in kp.items()}
        if prev is None:
            dyn.append(0.0)
        else:
            d = [abs(cur[k][0] - prev[k][0]) + abs(cur[k][1] - prev[k][1])
                 for k in cur
                 if k in prev and cur[k][2] >= VIS and prev[k][2] >= VIS]
            dyn.append(sum(d) / len(d) if d else 0.0)
        prev = cur
    return dyn


def smooth(v, k):
    """Moving average, so one noisy frame doesn't read as motion."""
    if k <= 1:
        return list(v)
    out = []
    for i in range(len(v)):
        w = v[max(0, i - k // 2): i + k // 2 + 1]
        out.append(sum(w) / len(w))
    return out


def mean_pose(frames, lo, hi):
    """Average (x, y) per joint over frames[lo:hi], visible joints only."""
    acc, n = {}, {}
    for f in frames[lo:hi]:
        if not f.get("detected"):
            continue
        for k, v in (f.get("keypoints") or {}).items():
            if v.get("visibility", 1.0) < VIS:
                continue
            ax, ay = acc.get(k, (0.0, 0.0))
            acc[k] = (ax + v["x_norm"], ay + v["y_norm"])
            n[k] = n.get(k, 0) + 1
    return {k: (acc[k][0] / n[k], acc[k][1] / n[k]) for k in acc if n.get(k)}


def pose_distance(a, b):
    """Mean |a - b| over joints present in both; None when they don't overlap."""
    shared = [k for k in a if k in b]
    if not shared:
        return None
    return sum(abs(a[k][0] - b[k][0]) + abs(a[k][1] - b[k][1])
               for k in shared) / len(shared)


def find_range(frames, rel=0.20, absolute=None, smooth_k=5, margin=3,
               min_frames=32, hold_guard=True):
    """Return (start, end, info). Scans inward from each end only."""
    n = len(frames)
    dyn = smooth(frame_dynamism(frames), smooth_k)
    peak = max(dyn) if dyn else 0.0
    thr = absolute if absolute is not None else peak * rel

    info = {"peak_dynamism": round(peak, 6),
            "threshold": round(thr, 6),
            "threshold_mode": ("absolute" if absolute is not None
                               else str(rel) + " x peak")}

    if peak <= 0 or thr <= 0:
        info["note"] = "no motion detected; nothing trimmed"
        return 0, n, info

    start = 0
    while start < n and dyn[start] < thr:
        start += 1
    end = n
    while end > start and dyn[end - 1] < thr:
        end -= 1

    if start >= end:                       # never crossed the threshold
        info["note"] = "clip never exceeds threshold; nothing trimmed"
        return 0, n, info

    # Hold guard: is the quiet tail a return to rest, or a pose being held?
    if hold_guard and end < n:
        rest = mean_pose(frames, 0, start) if start > 0 else mean_pose(frames, 0, min(5, n))
        tail = mean_pose(frames, end, n)
        d = pose_distance(rest, tail)
        info["tail_pose_distance"] = None if d is None else round(d, 4)
        if d is not None and d > HOLD_DIST:
            info["hold_guard"] = ("tail kept: pose differs from rest by "
                                  + format(d, ".3f") + " (> " + str(HOLD_DIST)
                                  + ") -- a held pose, not dead time")
            end = n
        elif d is not None:
            info["hold_guard"] = ("tail trimmed: pose within " + format(d, ".3f")
                                  + " of rest (<= " + str(HOLD_DIST) + ")")

    # Keep a little lead-in/out so the action doesn't begin mid-stride.
    start = max(0, start - margin)
    end = min(n, end + margin)

    if end - start < min_frames:           # guard against over-trimming
        centre = (start + end) // 2
        start = max(0, centre - min_frames // 2)
        end = min(n, start + min_frames)
        start = max(0, end - min_frames)
        info["note"] = "expanded to --min-frames " + str(min_frames)

    return start, end, info


def find_source_video(kp_path, meta):
    """Locate the footage a keypoints JSON came from.

    Tries the path recorded at extraction time first, then falls back to
    searching the dataset roots by filename stem -- the recorded path is often
    from whichever machine did the extraction, so it won't resolve here.
    """
    src = meta.get("source")
    if src and Path(src).exists():
        return Path(src)

    stem = Path(kp_path).name
    for suffix in ("_keypoints_full_body.json", "_keypoints.json", ".json"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if src:                      # e.g. a Mac path: keep only its filename stem
        stem = Path(src).stem or stem

    for root in VIDEO_ROOTS:
        base = IPCEI_ROOT / root
        if not base.is_dir():
            continue
        for ext in VIDEO_EXT:
            hits = list(base.glob("**/" + stem + ext))
            if hits:
                return hits[0]
    return None


def extract_keypoints_for_video(video, output_dir=None, extract_python=None,
                                force=False):
    """Run MediaPipe pose extraction on raw footage, returning the keypoints
    JSON path that trim_file() can then process -- lets --extract-video take
    an .mp4/.avi/... directly instead of requiring extraction to have already
    happened.

    Same subprocess call run_pipeline.py's own extract_keypoints() makes for
    stage 1 (pose_detection.py, run under the mediapipe+opencv interpreter),
    inlined here rather than imported: `import run_pipeline` pulls in its
    module-level `import build_dataset` too, which is unrelated to extraction
    and breaks this call if build_dataset.py ever moves or is renamed.
    """
    out_dir = Path(output_dir) if output_dir else (
        IPCEI_ROOT / "HumanPoseAction" / "output")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / (Path(video).stem + "_keypoints_full_body.json")

    if out_json.exists() and not force:
        print(f"[trim] keypoints already exist, skipping extraction: {out_json}")
        return out_json

    py = extract_python or DEFAULT_EXTRACT_PYTHON
    if not Path(py).exists():
        raise SystemExit(
            f"[trim] ERROR: extract-python interpreter not found: {py}\n"
            "       Pass a valid one with --extract-python <path-to-python.exe>\n"
            "       (it must have mediapipe+opencv installed).")

    print(f"[trim] extracting keypoints with: {py}")
    cmd = [py, "pose_detection.py",
           "--input", str(Path(video).resolve()),
           "--mode", "body",
           "--output", str(out_dir.resolve()),
           "--no-display", "--max-size", "600", "--confidence", "0.5"]
    # cwd=HPA_ROOT: pose_detection.py imports its sibling packages (utils,
    # config, detector) by relative name, so it only resolves from there.
    hpa_root = IPCEI_ROOT / "HumanPoseAction"
    proc = subprocess.run(cmd, cwd=str(hpa_root))
    if proc.returncode != 0:
        raise SystemExit(f"[trim] ERROR: pose_detection.py failed "
                         f"(exit {proc.returncode})")
    if not out_json.exists():
        raise SystemExit(f"[trim] ERROR: expected output not written: {out_json}")

    print(f"[trim] keypoints -> {out_json}")
    return out_json


def write_trimmed_video(video, start, end, out_path, python_exe=None):
    """Write frames [start, end) of `video` to out_path.

    Shelled out because cv2 lives in the MediaPipe interpreter, not the torch
    one this script usually runs under.
    """
    if python_exe is None:
        python_exe = DEFAULT_EXTRACT_PYTHON

    script = (
        "import cv2, sys\n"
        "src, out, a, b = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])\n"
        "c = cv2.VideoCapture(src)\n"
        "w = int(c.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))\n"
        "fps = c.get(cv2.CAP_PROP_FPS) or 30.0\n"
        "wr = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))\n"
        "i = n = 0\n"
        "while True:\n"
        "    ok, fr = c.read()\n"
        "    if not ok: break\n"
        "    if a <= i < b:\n"
        "        wr.write(fr); n += 1\n"
        "    i += 1\n"
        "    if i >= b: break\n"
        "c.release(); wr.release()\n"
        "print('WROTE=%d' % n)\n"
    )
    r = subprocess.run([str(python_exe), "-c", script, str(video), str(out_path),
                        str(start), str(end)], capture_output=True, text=True)
    if r.returncode != 0 or "WROTE=" not in r.stdout:
        return None, (r.stderr or r.stdout)[-300:]
    return out_path, int(r.stdout.split("WROTE=")[1].split()[0])


def trim_file(path, out_dir=None, dry_run=False, want_video=False,
             source_video=None, **kw):
    """source_video: the exact footage to also trim, when already known (the
    --video/--batch CLI modes extracted from it, so there is nothing to search
    for). Omit it to fall back to find_source_video()'s dataset-root search,
    for callers that only have the keypoints JSON."""
    path = Path(path)
    d = json.loads(path.read_text(encoding="utf-8"))
    frames = d.get("frames", [])
    if not frames:
        print("  " + path.name + ": no frames, skipped")
        return None

    fps = d.get("metadata", {}).get("source_fps") or 30.0
    start, end, info = find_range(frames, **kw)
    n = len(frames)

    report = {
        "source": str(path),
        "fps": fps,
        "original": {"frames": n, "duration_s": round(n / fps, 3)},
        "kept": {"start_frame": start, "end_frame": end,
                 "frames": end - start,
                 "start_s": round(start / fps, 3),
                 "end_s": round(end / fps, 3),
                 "duration_s": round((end - start) / fps, 3)},
        "trimmed": {"head_frames": start,
                    "head_s": round(start / fps, 3),
                    "tail_frames": n - end,
                    "tail_s": round((n - end) / fps, 3),
                    "total_frames": start + (n - end),
                    "percent": round(100.0 * (start + (n - end)) / n, 1)},
        "detection": info,
    }

    pct = report["trimmed"]["percent"]
    print("  {:<46} {:>4}f -> {:>4}f  (-{:>4.1f}%)  keep {:.2f}s-{:.2f}s".format(
        path.name[:46], n, end - start, pct,
        report["kept"]["start_s"], report["kept"]["end_s"]))
    if info.get("hold_guard", "").startswith("tail kept"):
        print("      hold guard: tail preserved (held pose, not dead time)")

    if dry_run:
        return report

    out_dir = Path(out_dir) if out_dir else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = path.name.replace(".json", "")

    kept = []
    for i, f in enumerate(frames[start:end]):
        g = dict(f)
        g["frame_id"] = i                  # re-index: downstream assumes 0..N-1
        g["timestamp_s"] = round(i / fps, 4)
        g["source_frame_id"] = f.get("frame_id", start + i)    # traceability
        kept.append(g)

    meta = dict(d.get("metadata", {}))
    meta.update({"total_frames_processed": len(kept),
                 "total_detections": sum(1 for f in kept if f.get("detected")),
                 "trimmed_from": {"start_frame": start, "end_frame": end,
                                  "original_frames": n}})

    out_json = out_dir / (stem + "_trimmed.json")
    out_json.write_text(json.dumps({"metadata": meta, "frames": kept}),
                        encoding="utf-8")
    report["output"] = str(out_json)

    if want_video:
        vid = Path(source_video) if source_video else find_source_video(
            path, d.get("metadata", {}))
        if vid is None:
            report["video"] = {"error": "source footage not found"}
            print("      video: source footage not found")
        else:
            out_v = out_dir / (stem.replace("_keypoints_full_body", "")
                               + "_trimmed.mp4")
            got, extra = write_trimmed_video(vid, start, end, out_v)
            if got is None:
                report["video"] = {"error": extra}
                print("      video: FAILED -- " + str(extra)[:120])
            else:
                report["video"] = {"source": str(vid), "output": str(out_v),
                                   "frames_written": extra}
                print("      video: " + out_v.name + "  (" + str(extra) + " frames)")

    (out_dir / (stem + "_trim.json")).write_text(json.dumps(report, indent=2),
                                                 encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--json", metavar="PATH", help="one keypoints JSON to trim")
    mode.add_argument("--batch_json", metavar="FOLDER",
                      help="every *_keypoints_full_body.json in FOLDER")
    mode.add_argument("--video", metavar="PATH",
                      help="one video (.mp4/.avi/.mov/.mkv): extract keypoints "
                           "with MediaPipe, trim, and also write a trimmed "
                           "video (the source is known exactly)")
    mode.add_argument("--batch", metavar="FOLDER",
                      help="every video in FOLDER, same handling as --video")
    p.add_argument("--out-dir", metavar="DIR",
                   help="where output is written (default: trimmed/)")
    p.add_argument("--extract-python",
                   help="interpreter with mediapipe+opencv, for --video/--batch "
                        "extraction and --view-trim (default: %s)"
                        % DEFAULT_EXTRACT_PYTHON)
    p.add_argument("--view-trim", action="store_true",
                   help="after processing, open view_trim.py on the file(s) "
                        "just handled so the trim can be checked visually")
    p.add_argument("--rel", type=float, default=0.20,
                   help="threshold as a fraction of the clip's peak (default 0.20)")
    p.add_argument("--abs", dest="absolute", type=float,
                   help="absolute threshold instead of --rel")
    p.add_argument("--smooth", type=int, default=5, metavar="K",
                   help="moving-average window over the motion signal (default 5)")
    p.add_argument("--margin", type=int, default=3,
                   help="frames of lead-in/out to keep (default 3)")
    p.add_argument("--min-frames", type=int, default=32,
                   help="never trim shorter than this (default 32)")
    p.add_argument("--no-hold-guard", action="store_true",
                   help="disable held-pose protection (see module docstring)")
    p.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = p.parse_args()

    py = args.extract_python or DEFAULT_EXTRACT_PYTHON

    # Build the list of keypoints JSON to trim, plus -- for the two video
    # modes only -- the exact source footage each one came from, so trim_file
    # can trim that footage directly instead of searching the dataset roots.
    sources, video_for = [], {}
    if args.json:
        if not Path(args.json).exists():
            p.error("file not found: " + args.json)
        sources = [Path(args.json)]
    elif args.batch_json:
        sources = sorted(Path(args.batch_json).glob("*_keypoints_full_body.json"))
        if not sources:
            p.error("no *_keypoints_full_body.json in " + args.batch_json)
    elif args.video:
        if not Path(args.video).exists():
            p.error("video not found: " + args.video)
        kp = extract_keypoints_for_video(args.video, extract_python=py)
        sources = [kp]
        video_for[str(kp)] = args.video
    else:  # args.batch (folder of videos)
        videos = sorted(v for ext in VIDEO_EXT
                        for v in Path(args.batch).glob("*" + ext))
        if not videos:
            p.error("no video (%s) in %s" % (", ".join(VIDEO_EXT), args.batch))
        print("[trim] extracting keypoints for " + str(len(videos))
              + " video(s) with MediaPipe ...")
        for v in videos:
            kp = extract_keypoints_for_video(v, extract_python=py)
            sources.append(kp)
            video_for[str(kp)] = str(v)

    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_DIR

    kw = dict(rel=args.rel, absolute=args.absolute, smooth_k=args.smooth,
              margin=args.margin, min_frames=args.min_frames,
              hold_guard=not args.no_hold_guard)

    print("[trim] " + str(len(sources)) + " file(s) -> " + str(out_dir)
          + ("  (dry run, nothing written)" if args.dry_run else ""))
    reports = []
    for f in sources:
        src_vid = video_for.get(str(f))
        r = trim_file(f, out_dir, args.dry_run, want_video=bool(src_vid),
                     source_video=src_vid, **kw)
        if r:
            reports.append(r)

    if reports:
        tot = sum(r["trimmed"]["percent"] for r in reports) / len(reports)
        held = sum(1 for r in reports
                   if r["detection"].get("hold_guard", "").startswith("tail kept"))
        print("[trim] mean removed: {:.1f}%   tails preserved by hold guard: {}/{}"
              .format(tot, held, len(reports)))

    if args.view_trim:
        print("[trim] opening view_trim.py ...")
        subprocess.run([py, str(_HERE / "view_trim.py")]
                       + [str(s) for s in sources])


if __name__ == "__main__":
    main()
