"""
Visualize NTU RGB+D ground truth skeleton overlaid on the RGB video.

Single-clip mode:
    python visualize_ntu_skeleton.py --video <path> --skeleton <path>

Batch mode (all clips for one action in a folder):
    python visualize_ntu_skeleton.py --folder <video-dir> --skeleton-folder <skel-dir> --action A031
    python visualize_ntu_skeleton.py --folder <dir> --action 31   # videos and skeletons in same dir

Controls:
    SPACE  - pause / resume
    q      - quit (batch: exits entirely)
    n      - next frame (while paused)
    (end of clip in batch mode) any key - next clip
"""

import argparse
import os
import sys
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ── NTU RGB+D joint indices and skeleton connections ──────────────────────────

JOINT_NAMES = {
    0:  "SpineBase",    1: "SpineMid",     2: "Neck",         3: "Head",
    4:  "ShoulderL",    5: "ElbowL",       6: "WristL",       7: "HandL",
    8:  "ShoulderR",    9: "ElbowR",       10: "WristR",      11: "HandR",
    12: "HipL",         13: "KneeL",       14: "AnkleL",      15: "FootL",
    16: "HipR",         17: "KneeR",       18: "AnkleR",      19: "FootR",
    20: "SpineShoulder",21: "HandTipL",    22: "ThumbL",      23: "HandTipR",
    24: "ThumbR",
}

BONES = [
    # spine
    (0, 1), (1, 20), (20, 2), (2, 3),
    # left arm
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
    # right arm
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),
    # left leg
    (0, 12), (12, 13), (13, 14), (14, 15),
    # right leg
    (0, 16), (16, 17), (17, 18), (18, 19),
]

SUBJECT_COLORS = [
    {"joint": (0, 255, 120), "bone": (255, 200, 0)},
    {"joint": (0, 180, 255), "bone": (255, 80, 200)},
]


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Joint:
    x: float; y: float; z: float
    depth_x: float; depth_y: float
    color_x: float; color_y: float
    orient_w: float; orient_x: float; orient_y: float; orient_z: float
    tracking_state: int


@dataclass
class Body:
    body_id: int
    joints: List[Joint] = field(default_factory=list)


@dataclass
class Frame:
    bodies: List[Body] = field(default_factory=list)


# ── Skeleton file parser ──────────────────────────────────────────────────────

def parse_skeleton(path: str) -> List[Frame]:
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    idx = 0
    num_frames = int(lines[idx]); idx += 1
    frames = []

    for _ in range(num_frames):
        frame = Frame()
        num_bodies = int(lines[idx]); idx += 1

        for _ in range(num_bodies):
            body_info = lines[idx].split(); idx += 1
            body = Body(body_id=int(body_info[0]))

            num_joints = int(lines[idx]); idx += 1
            for _ in range(num_joints):
                vals = list(map(float, lines[idx].split())); idx += 1
                joint = Joint(
                    x=vals[0],  y=vals[1],  z=vals[2],
                    depth_x=vals[3], depth_y=vals[4],
                    color_x=vals[5], color_y=vals[6],
                    orient_w=vals[7], orient_x=vals[8],
                    orient_y=vals[9], orient_z=vals[10],
                    tracking_state=int(vals[11]),
                )
                body.joints.append(joint)
            frame.bodies.append(body)

        frames.append(frame)

    return frames


# ── Drawing ───────────────────────────────────────────────────────────────────

def joint_pixel(joint: Joint, scale_x: float, scale_y: float):
    px = int(joint.color_x * scale_x)
    py = int(joint.color_y * scale_y)
    return px, py


def draw_skeleton(frame_img: np.ndarray, skeleton_frame: Frame,
                  orig_w: int, orig_h: int) -> np.ndarray:
    h, w = frame_img.shape[:2]
    sx = w / orig_w
    sy = h / orig_h

    for s_idx, body in enumerate(skeleton_frame.bodies):
        if not body.joints:
            continue
        colors = SUBJECT_COLORS[s_idx % len(SUBJECT_COLORS)]
        jc = colors["joint"]
        bc = colors["bone"]

        pixels = [joint_pixel(j, sx, sy) for j in body.joints]

        for (a, b) in BONES:
            if a >= len(pixels) or b >= len(pixels):
                continue
            pa, pb = pixels[a], pixels[b]
            ta = body.joints[a].tracking_state
            tb = body.joints[b].tracking_state
            if ta == 0 or tb == 0:
                continue
            alpha = 0.6 if (ta == 1 or tb == 1) else 1.0
            color = tuple(int(c * alpha) for c in bc)
            cv2.line(frame_img, pa, pb, color, 2, cv2.LINE_AA)

        for i, (px, py) in enumerate(pixels):
            ts = body.joints[i].tracking_state
            if ts == 0:
                continue
            radius = 4
            fill = jc if ts == 2 else (128, 128, 128)
            cv2.circle(frame_img, (px, py), radius, fill, -1, cv2.LINE_AA)
            cv2.circle(frame_img, (px, py), radius, (0, 0, 0), 1, cv2.LINE_AA)

    return frame_img


def resize_to(frame, target_w: int, target_h: int):
    if target_w <= 0 and target_h <= 0:
        return frame
    h, w = frame.shape[:2]
    if target_w > 0 and target_h > 0:
        new_w, new_h = target_w, target_h
    elif target_w > 0:
        new_w = target_w
        new_h = max(1, int(h * target_w / w))
    else:
        new_h = target_h
        new_w = max(1, int(w * target_h / h))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _display_size(orig_w: int, orig_h: int, target_w: int, target_h: int) -> Tuple[int, int]:
    if target_w > 0 and target_h > 0:
        return target_w, target_h
    elif target_w > 0:
        return target_w, max(1, int(orig_h * target_w / orig_w))
    elif target_h > 0:
        return max(1, int(orig_w * target_h / orig_h)), target_h
    else:
        return orig_w, orig_h


# ── Batch file discovery ──────────────────────────────────────────────────────

def find_action_pairs(video_folder: str, action: str,
                      skeleton_folder: Optional[str] = None) -> List[Tuple[Optional[str], str]]:
    """
    Return sorted list of (video_path_or_None, skeleton_path) for the given action.
    Accepts action as 'A031', '031', or '31'.
    skeleton_folder defaults to video_folder when not provided.
    """
    action_tag = "A" + action.lstrip("Aa").zfill(3)
    skel_dir = skeleton_folder or video_folder

    pairs = []
    for entry in sorted(os.listdir(skel_dir)):
        if not entry.endswith(".skeleton"):
            continue
        stem = entry[: -len(".skeleton")]
        if action_tag not in stem:
            continue
        skeleton_path = os.path.join(skel_dir, entry)
        video_path = None
        for suffix in ("_rgb.avi", "_rgb.mp4", ".avi", ".mp4"):
            candidate = os.path.join(video_folder, stem + suffix)
            if os.path.isfile(candidate):
                video_path = candidate
                break
        pairs.append((video_path, skeleton_path))

    return pairs


# ── Clip player ───────────────────────────────────────────────────────────────

def play_clip(video_path: str, skeleton_frames: List[Frame],
              disp_w: int, disp_h: int,
              label: str = "", show_next_prompt: bool = False) -> bool:
    """
    Play one video+skeleton clip.
    Returns True to continue to the next clip, False to quit entirely.
    show_next_prompt: when True, pause at clip end with "press any key for next".
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return True

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    win_w, win_h = _display_size(orig_w, orig_h, disp_w, disp_h)

    print(f"  Video: {orig_w}x{orig_h}  {fps:.1f} fps  {total} frames")

    cv2.namedWindow("NTU Skeleton GT", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("NTU Skeleton GT", win_w, win_h)

    delay   = max(1, int(1000 / fps))
    paused  = False
    frame_i = 0
    result  = True

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                if show_next_prompt:
                    overlay = np.zeros((win_h, win_w, 3), dtype=np.uint8)
                    msg = f"{label}  |  any key = next clip    q = quit"
                    cv2.putText(overlay, msg, (20, win_h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2, cv2.LINE_AA)
                    cv2.imshow("NTU Skeleton GT", overlay)
                    while True:
                        k = cv2.waitKey(100) & 0xFF
                        if k == ord('q'):
                            result = False
                        elif k != 255:
                            result = True
                        else:
                            continue
                        break
                break

            frame = resize_to(frame, disp_w, disp_h)
            if frame_i < len(skeleton_frames):
                draw_skeleton(frame, skeleton_frames[frame_i], orig_w, orig_h)

            info = (f"{label}  Frame {frame_i}/{total}  "
                    f"[SPACE=pause  q=quit]")
            cv2.putText(frame, info, (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0),  3, cv2.LINE_AA)
            cv2.putText(frame, info, (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("NTU Skeleton GT", frame)
            frame_i += 1

        key = cv2.waitKey(1 if paused else delay) & 0xFF

        if key == ord('q'):
            result = False
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('n') and paused:
            ret, frame = cap.read()
            if ret:
                frame = resize_to(frame, disp_w, disp_h)
                if frame_i < len(skeleton_frames):
                    draw_skeleton(frame, skeleton_frames[frame_i], orig_w, orig_h)
                cv2.imshow("NTU Skeleton GT", frame)
                frame_i += 1

    cap.release()
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize NTU RGB+D ground-truth skeleton on RGB video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Single-clip:  -v clip.avi -s clip.skeleton\n"
            "Batch mode:   -f /data/videos -F /data/skeletons -a A031\n"
            "              -f /data/nturgbd -a A031          (same folder)"
        ),
    )
    p.add_argument("--video",    "-v", help="Path to the RGB video file (.avi/.mp4)")
    p.add_argument("--skeleton", "-s", help="Path to the .skeleton ground-truth file")
    p.add_argument("--folder",   "-f", help="Folder containing NTU RGB+D files (batch mode)", default='d:/datasets/NTU_RGB/nturgb+d_rgb')
    p.add_argument("--action",   "-a", help="Action label to play, e.g. A031 or 031 (batch mode)", default='A031')
    p.add_argument("--skeleton-folder", "-F", help="Folder containing .skeleton files (defaults to --folder)", default='d:/datasets/NTU_RGB/nturgb+d_skeletons')
    # p.add_argument("--folder",          "-f", help="Folder containing RGB video files (batch mode)")
    # p.add_argument("--skeleton-folder", "-F", help="Folder containing .skeleton files (defaults to --folder)")
    # p.add_argument("--action",          "-a", help="Action label to play, e.g. A031 or 031 (batch mode)")
    p.add_argument("--width",  "-W", type=int, default=800, help="Display width  (default: 800)")
    p.add_argument("--height", "-H", type=int, default=0,   help="Display height (default: auto)")
    return p.parse_args()


def main():
    args = parse_args()

    batch_mode  = bool(args.folder and args.action)
    single_mode = bool(args.video  and args.skeleton)

    if not batch_mode and not single_mode:
        print("Error: provide either  --video + --skeleton  or  --folder + --action")
        raise SystemExit(1)

    if batch_mode:
        skel_folder = args.skeleton_folder or args.folder
        pairs = find_action_pairs(args.folder, args.action, skel_folder)
        if not pairs:
            print(f"No .skeleton files matching action '{args.action}' found in: {skel_folder}")
            raise SystemExit(1)

        action_tag = "A" + args.action.lstrip("Aa").zfill(3)
        print(f"Action {action_tag}: {len(pairs)} clip(s) found")

        for i, (video_path, skeleton_path) in enumerate(pairs):
            clip_num = f"[{i+1}/{len(pairs)}]"
            if video_path is None:
                print(f"{clip_num} No video found for {os.path.basename(skeleton_path)}, skipping")
                continue

            print(f"{clip_num} {os.path.basename(video_path)}")
            print(f"     Parsing skeleton: {os.path.basename(skeleton_path)}")
            skeleton_frames = parse_skeleton(skeleton_path)
            print(f"     Loaded {len(skeleton_frames)} skeleton frames")

            label           = f"{os.path.basename(video_path)}  {clip_num}"
            is_last         = (i == len(pairs) - 1)
            show_next_prompt = not is_last

            if not play_clip(video_path, skeleton_frames, args.width, args.height,
                             label, show_next_prompt):
                break

    else:  # single-clip mode
        if not os.path.isfile(args.video):
            print(f"Video not found: {args.video}"); raise SystemExit(1)
        if not os.path.isfile(args.skeleton):
            print(f"Skeleton not found: {args.skeleton}"); raise SystemExit(1)

        print(f"Parsing skeleton: {args.skeleton}")
        skeleton_frames = parse_skeleton(args.skeleton)
        print(f"  Loaded {len(skeleton_frames)} skeleton frames")

        play_clip(args.video, skeleton_frames, args.width, args.height,
                  label=os.path.basename(args.video), show_next_prompt=False)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
