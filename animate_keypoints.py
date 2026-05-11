"""
Keypoints Animation
===================
Animates body pose from a keypoints_full.json produced by pose_detector.py.

Usage:
  python animate_keypoints.py
  python animate_keypoints.py --input output/keypoints_full.json
  python animate_keypoints.py --fps 25 --min-vis 0.3
  python animate_keypoints.py --save pose_animation.mp4
  python animate_keypoints.py --save pose_animation.gif
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

# BlazePose skeleton connections (index pairs matching MediaPipe's POSE_CONNECTIONS)
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),          # nose → left eye → ear
    (0, 4), (4, 5), (5, 6), (6, 8),          # nose → right eye → ear
    (9, 10),                                   # mouth
    (11, 12),                                  # shoulders
    (11, 13), (13, 15),                        # left arm
    (15, 17), (17, 19), (19, 15), (15, 21),   # left hand
    (12, 14), (14, 16),                        # right arm
    (16, 18), (18, 20), (20, 16), (16, 22),   # right hand
    (11, 23), (12, 24),                        # torso sides
    (23, 24),                                  # hips
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),  # left leg
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),  # right leg
]

# Color per body segment
SEGMENT_COLORS = {
    "face":        "#00e5ff",
    "left_arm":    "#76ff03",
    "right_arm":   "#ff6d00",
    "torso":       "#ffe57f",
    "left_leg":    "#40c4ff",
    "right_leg":   "#ff4081",
}

def connection_color(a, b):
    pair = (min(a, b), max(a, b))
    if pair[0] <= 10:
        return SEGMENT_COLORS["face"]
    if pair in {(11, 13), (13, 15), (15, 17), (17, 19), (19, 15), (15, 21)}:
        return SEGMENT_COLORS["left_arm"]
    if pair in {(12, 14), (14, 16), (16, 18), (18, 20), (20, 16), (16, 22)}:
        return SEGMENT_COLORS["right_arm"]
    if pair in {(11, 12), (11, 23), (12, 24), (23, 24)}:
        return SEGMENT_COLORS["torso"]
    if pair in {(23, 25), (25, 27), (27, 29), (29, 31), (27, 31)}:
        return SEGMENT_COLORS["left_leg"]
    return SEGMENT_COLORS["right_leg"]


def load_frames(json_path: Path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["metadata"], data["frames"]


def frame_to_arrays(frame_data, min_vis: float):
    """Return (ids, xs, ys, vis) arrays for one frame. Filters by visibility."""
    kp = frame_data.get("keypoints") or {}
    ids, xs, ys, vis = [], [], [], []
    for name, v in kp.items():
        if v["visibility"] >= min_vis:
            ids.append(v["id"])
            xs.append(v["x_norm"])
            ys.append(1.0 - v["y_norm"])  # flip Y: image top → plot top
            vis.append(v["visibility"])
    return np.array(ids), np.array(xs), np.array(ys), np.array(vis)


def build_figure():
    fig, ax = plt.subplots(figsize=(6, 7), facecolor="#111111")
    ax.set_facecolor("#111111")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def run(json_path: Path, fps: float, min_vis: float, save_path: Path | None):
    meta, frames = load_frames(json_path)

    source_fps = meta.get("source_fps", 25.0)
    interval_ms = 1000.0 / fps

    fig, ax = build_figure()

    title = ax.set_title("", color="white", fontsize=9, pad=6)

    # Pre-build line objects for each connection
    conn_lines = []
    for a, b in CONNECTIONS:
        color = connection_color(a, b)
        (line,) = ax.plot([], [], "-", color=color, linewidth=1.8, alpha=0.85)
        conn_lines.append((a, b, line))

    scatter = ax.scatter([], [], s=20, c=[], cmap="RdYlGn",
                         vmin=0.0, vmax=1.0, zorder=5)

    no_det_text = ax.text(
        0.5, 0.5, "No detection", color="#ff4444",
        ha="center", va="center", fontsize=14,
        transform=ax.transAxes, visible=False,
    )

    def update(frame_idx):
        fd = frames[frame_idx]
        ids, xs, ys, vis = frame_to_arrays(fd, min_vis)

        detected = fd.get("detected", False) and len(ids) > 0
        no_det_text.set_visible(not detected)

        # Build id → (x, y) lookup
        lookup = {int(i): (x, y) for i, x, y in zip(ids, xs, ys)}

        for a, b, line in conn_lines:
            if a in lookup and b in lookup:
                xa, ya = lookup[a]
                xb, yb = lookup[b]
                line.set_data([xa, xb], [ya, yb])
                line.set_visible(True)
            else:
                line.set_visible(False)

        if len(xs) > 0:
            scatter.set_offsets(np.c_[xs, ys])
            scatter.set_array(vis)
            scatter.set_visible(True)
        else:
            scatter.set_visible(False)

        ts = fd.get("timestamp_s", 0.0)
        title.set_text(
            f"Frame {fd['frame_id']:04d}  |  t = {ts:.2f}s  |  "
            f"{len(ids)}/33 landmarks  |  vis ≥ {min_vis}"
        )
        return [scatter, title, no_det_text] + [l for _, _, l in conn_lines]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=interval_ms,
        blit=True,
    )

    if save_path:
        suffix = save_path.suffix.lower()
        print(f"Saving animation to {save_path} ...")
        if suffix == ".gif":
            writer = animation.PillowWriter(fps=fps)
        else:
            writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
        ani.save(str(save_path), writer=writer)
        print("Done.")
    else:
        plt.tight_layout()
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Animate body pose from keypoints_full.json"
    )
    parser.add_argument("--input", "-i", default="output/keypoints_full.json",
                        help="Path to keypoints JSON (default: output/keypoints_full.json)")
    parser.add_argument("--fps", type=float, default=25.0,
                        help="Animation playback speed in fps (default: 25)")
    parser.add_argument("--min-vis", type=float, default=0.5,
                        help="Hide landmarks below this visibility (default: 0.5)")
    parser.add_argument("--save", default=None,
                        help="Save animation to file (.mp4 or .gif). Omit to display.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        json_path=Path(args.input),
        fps=args.fps,
        min_vis=args.min_vis,
        save_path=Path(args.save) if args.save else None,
    )
