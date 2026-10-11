"""
view_trim.py
============

Desktop window showing what trim_keypoints.py does to a clip: the whole-body
motion curve on top, a per-joint motion spectrogram underneath, and the trimmed
spans shaded across both.

    python view_trim.py --batch "../HumanPoseAction/output/S001C001P001R001A0*.json"
    python view_trim.py ../HumanPoseAction/output/S001C001P001R001A008_rgb_keypoints_full_body.json
    python view_trim.py --batch "...A008*.json" --save shots/   # headless PNGs

Keys, once the window is up:

    left / right    previous / next clip
    v               play the trimmed footage (cv2 window; q or Esc closes it)
    f               play the FULL clip instead, so the cut is visible in context
    s               save the current view as a PNG next to the script
    q               quit

RUN IT WITH THE MEDIAPIPE INTERPRETER, not the torch venv -- matplotlib and cv2
live there:

    C:/Users/alext/AppData/Local/Programs/Python/Python311/python.exe view_trim.py ...

The spectrogram is the point of the window: rows are joints grouped head / arms /
torso / legs top to bottom, colour is how far that joint moved between frames.
It answers "which body part moved, and when" -- the curve above only answers
"how much did anything move".
"""

import argparse
import glob
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import video.trim_keypoints as tk        # stdlib-only at import time; safe here

# anatomical top-to-bottom order, so the heatmap reads like a body
GROUPS = [
    ("head",  ["NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR",
               "MOUTH_LEFT", "MOUTH_RIGHT"]),
    ("arms",  ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
               "LEFT_WRIST", "RIGHT_WRIST", "LEFT_INDEX", "RIGHT_INDEX"]),
    ("torso", ["LEFT_HIP", "RIGHT_HIP"]),
    ("legs",  ["LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE",
               "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"]),
]
JOINTS = [j for _, js in GROUPS for j in js]


def per_joint_motion(frames):
    """(n_joints, n_frames) displacement, same visibility rule as the trimmer."""
    import numpy as np
    prev = None
    out = np.zeros((len(JOINTS), len(frames)), dtype=float)
    for t, f in enumerate(frames):
        kp = f.get("keypoints") or {}
        cur = ({k: (v["x_norm"], v["y_norm"], v.get("visibility", 1.0))
                for k, v in kp.items()} if (f.get("detected") and kp) else None)
        if cur is not None and prev is not None:
            for r, j in enumerate(JOINTS):
                a, b = cur.get(j), prev.get(j)
                if a and b and a[2] >= tk.VIS and b[2] >= tk.VIS:
                    out[r, t] = abs(a[0] - b[0]) + abs(a[1] - b[1])
        prev = cur
    return out


def smooth_rows(mat, k=3):
    import numpy as np
    if k <= 1:
        return mat
    ker = np.ones(k) / k
    return np.vstack([np.convolve(r, ker, mode="same") for r in mat])


def load_clip(path):
    """Everything the window needs for one clip, or None if it isn't one.

    A glob over the output folder also catches non-clip JSONs living beside
    the keypoints (trim reports, analyze_keypoints metrics) -- no frames.
    """
    import numpy as np
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    frames = d.get("frames", []) if isinstance(d, dict) else []
    if not frames:
        return None
    fps = d.get("metadata", {}).get("source_fps") or 30.0
    dyn = tk.smooth(tk.frame_dynamism(frames), 5)
    start, end, info = tk.find_range(frames)
    grid = smooth_rows(per_joint_motion(frames), 3)
    peak = max(dyn) if dyn else 0.0
    return {
        "path": Path(path), "name": Path(path).name.replace(
            "_keypoints_full_body.json", ""),
        "meta": d.get("metadata", {}), "frames": frames, "fps": fps,
        "n": len(frames), "dyn": np.asarray(dyn), "grid": grid,
        "peak": peak, "thr": info["threshold"],
        "start": start, "end": end, "info": info,
    }


# ---------------------------------------------------------------- video ----

def play(clip, trimmed_only=True):
    """Play the source footage in a cv2 window; q or Esc closes it."""
    try:
        import cv2
    except ImportError:
        print("[view] cv2 not available in this interpreter -- can't play video")
        return
    vid = tk.find_source_video(clip["path"], clip["meta"])
    if vid is None:
        print("[view] source footage not found for " + clip["name"])
        return

    a, b = (clip["start"], clip["end"]) if trimmed_only else (0, clip["n"])
    cap = cv2.VideoCapture(str(vid))
    fps = cap.get(cv2.CAP_PROP_FPS) or clip["fps"]
    delay = max(1, int(1000 / fps))
    title = clip["name"] + ("  [trimmed]" if trimmed_only else "  [full]")
    print("[view] playing " + title + "  (q/Esc to close)")

    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if a <= i < b:
            if not trimmed_only and (i < clip["start"] or i >= clip["end"]):
                fr = (fr * 0.45).astype(fr.dtype)      # dim what gets cut
                cv2.putText(fr, "TRIMMED", (14, 34), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(fr, "%.2fs  f%d" % (i / fps, i), (14, fr.shape[0] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.imshow(title, fr)
            if cv2.waitKey(delay) & 0xFF in (ord("q"), 27):
                break
        i += 1
        if i >= b:
            break
    cap.release()
    cv2.destroyWindow(title)


# ----------------------------------------------------------------- plot ----

def draw(clip, fig, axes):
    import numpy as np
    ax_top, ax_bot = axes
    ax_top.clear(); ax_bot.clear()

    n, fps = clip["n"], clip["fps"]
    t = np.arange(n) / fps
    s_s, e_s = clip["start"] / fps, clip["end"] / fps
    peak = clip["peak"] or 1e-9

    # --- motion curve, as % of this clip's peak so the threshold is comparable
    pct = clip["dyn"] / peak * 100.0
    ax_top.fill_between(t, pct, color="#2a78d6", alpha=0.12, linewidth=0)
    ax_top.plot(t, pct, color="#2a78d6", lw=1.8)
    ax_top.axhline(clip["thr"] / peak * 100.0, color="#898781", lw=1,
                   ls=(0, (3, 3)), label="threshold (20% of peak)")
    for x in (s_s, e_s):
        if 0 < x < n / fps:
            ax_top.axvline(x, color="#2a78d6", lw=1, alpha=0.6)
    if clip["start"] > 0:
        ax_top.axvspan(0, s_s, color="#c3c2b7", alpha=0.28, linewidth=0)
    if clip["end"] < n:
        ax_top.axvspan(e_s, n / fps, color="#c3c2b7", alpha=0.28, linewidth=0)
    ax_top.set_ylim(0, 105)
    # imshow below pins its x-limits to the extent with no margin; match that
    # exactly or the trim boundary lands on a different pixel in each lane and
    # the two views stop lining up.
    ax_top.set_xlim(0, n / fps)
    ax_top.margins(x=0)
    ax_top.set_ylabel("motion\n(% of peak)", fontsize=8.5)
    ax_top.tick_params(labelsize=8, bottom=False, labelbottom=False)
    ax_top.legend(fontsize=7.5, loc="upper right", frameon=False)
    for sp in ("top", "right"):
        ax_top.spines[sp].set_visible(False)

    # --- per-joint spectrogram
    g = clip["grid"]
    vmax = g.max() or 1e-9
    ax_bot.imshow(g, aspect="auto", origin="upper", cmap="Blues",
                  vmin=0, vmax=vmax, interpolation="nearest",
                  extent=[0, n / fps, len(JOINTS), 0])
    if clip["start"] > 0:
        ax_bot.axvspan(0, s_s, color="#fcfcfb", alpha=0.6, linewidth=0)
    if clip["end"] < n:
        ax_bot.axvspan(e_s, n / fps, color="#fcfcfb", alpha=0.6, linewidth=0)
    for x in (s_s, e_s):
        if 0 < x < n / fps:
            ax_bot.axvline(x, color="#2a78d6", lw=1, alpha=0.75)

    # group bands + labels instead of 25 unreadable joint ticks
    edges, acc = [], 0
    for name, js in GROUPS:
        edges.append((name, acc, acc + len(js)))
        acc += len(js)
    ax_bot.set_yticks([(a + b) / 2 for _, a, b in edges])
    ax_bot.set_yticklabels([name for name, _, _ in edges], fontsize=8.5)
    for _, a, b in edges[:-1]:
        ax_bot.axhline(b, color="#fcfcfb", lw=1.2)
    ax_bot.set_xlim(0, n / fps)
    ax_bot.set_xlabel("seconds", fontsize=8.5)
    ax_bot.tick_params(labelsize=8)

    kept = clip["end"] - clip["start"]
    cut = 100.0 * (n - kept) / n
    guard = clip["info"].get("hold_guard", "")
    badge = ("  |  hold guard: TAIL KEPT" if guard.startswith("tail kept")
             else ("  |  hold guard: tail trimmed" if guard else ""))
    fig.suptitle(
        "%s   %d -> %d frames  (-%.1f%%)   keep %.2fs-%.2fs%s"
        % (clip["name"], n, kept, cut, s_s, e_s, badge),
        fontsize=10.5, x=0.01, ha="left")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", nargs="*", help="keypoints JSON file(s)")
    p.add_argument("--batch", metavar="GLOB", help='e.g. "output/*A008*.json"')
    p.add_argument("--save", metavar="DIR",
                   help="render every clip to PNG in DIR and exit (no window)")
    args = p.parse_args()

    files = list(args.input)
    if args.batch:
        files += sorted(glob.glob(args.batch))
    if not files:
        p.error("give one or more JSON files, or --batch GLOB")

    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("[view] loading %d clip(s) ..." % len(files))
    clips = []
    for f in files:
        c = load_clip(f)
        if c is None:
            print("  skipped (no frames, not a keypoints clip): " + Path(f).name)
        else:
            clips.append(c)
    if not clips:
        p.error("none of the given files contain keypoint frames")

    fig, axes = plt.subplots(
        2, 1, figsize=(11, 6.2), height_ratios=[1, 1.9],
        gridspec_kw=dict(hspace=0.08, left=0.095, right=0.985,
                         top=0.90, bottom=0.085))
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")

    state = {"i": 0}

    def show():
        draw(clips[state["i"]], fig, axes)
        fig.canvas.draw_idle()

    if args.save:
        out = Path(args.save); out.mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(clips):
            state["i"] = i
            draw(c, fig, axes)
            f = out / (c["name"] + "_trim.png")
            fig.savefig(f, dpi=140, facecolor=fig.get_facecolor())
            print("  " + f.name)
        print("[view] wrote %d PNG(s) to %s" % (len(clips), out))
        return

    def on_key(ev):
        if ev.key in ("right", "down", "n"):
            state["i"] = (state["i"] + 1) % len(clips); show()
        elif ev.key in ("left", "up", "p"):
            state["i"] = (state["i"] - 1) % len(clips); show()
        elif ev.key == "v":
            play(clips[state["i"]], trimmed_only=True)
        elif ev.key == "f":
            play(clips[state["i"]], trimmed_only=False)
        elif ev.key == "s":
            c = clips[state["i"]]
            f = _HERE / (c["name"] + "_trim.png")
            fig.savefig(f, dpi=140, facecolor=fig.get_facecolor())
            print("[view] saved " + str(f))
        elif ev.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    show()
    print("[view] keys:  left/right = clip   v = trimmed video   "
          "f = full video   s = save PNG   q = quit")
    plt.show()


if __name__ == "__main__":
    main()
