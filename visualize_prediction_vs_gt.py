"""
Visualize MediaPipe body-pose predictions (exported by pose_detection.py) together with the
NTU RGB+D ground-truth skeleton, overlaid on the same RGB video, for side-by-side comparison.

Usage:
    python visualize_prediction_vs_gt.py --keypoints output/S001C003P004R001A031_rgb_keypoints_full_body.json

    # override any auto-discovered path
    python visualize_prediction_vs_gt.py --keypoints <json> --video <avi> --skeleton <skeleton>

Controls:
    SPACE  - pause / resume
    n      - next frame (while paused)
    q      - quit
"""

import argparse
import json
import os

import cv2
import mediapipe as mp

from visualize_ntu_skeleton import parse_skeleton, draw_skeleton, resize_to, _display_size

mp_pose = mp.solutions.pose

PRED_JOINT_COLOR = (0, 140, 255)
PRED_BONE_COLOR  = (0, 220, 255)

DEFAULT_SKELETON_FOLDER = "d:/datasets/NTU_RGB/nturgb+d_skeletons"
DEFAULT_VIDEO_FOLDER    = "d:/datasets/NTU_RGB/nturgb+d_rgb"


# ── Loading ────────────────────────────────────────────────────────────────────

def load_predictions(json_path: str):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    meta = data["metadata"]
    frames_by_id = {
        frame["frame_id"]: frame["keypoints"]
        for frame in data["frames"]
        if frame.get("detected") and frame.get("keypoints")
    }
    return meta, frames_by_id


def resolve_video_path(meta: dict, video_arg: str) -> str:
    if video_arg:
        return video_arg
    source = meta.get("source")
    if source and os.path.isfile(source):
        return source
    return os.path.join(DEFAULT_VIDEO_FOLDER, meta["source_file"] + ".avi")


def resolve_skeleton_path(meta: dict, skeleton_arg: str, skeleton_folder: str) -> str:
    if skeleton_arg:
        return skeleton_arg
    stem = meta["source_file"]
    if stem.endswith("_rgb"):
        stem = stem[: -len("_rgb")]
    return os.path.join(skeleton_folder, stem + ".skeleton")


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_prediction(frame, keypoints: dict, scale_x: float, scale_y: float):
    pixels = {}
    for kp in keypoints.values():
        px = int(kp["x_px"] * scale_x)
        py = int(kp["y_px"] * scale_y)
        pixels[kp["id"]] = (px, py)

    for a, b in mp_pose.POSE_CONNECTIONS:
        if a in pixels and b in pixels:
            cv2.line(frame, pixels[a], pixels[b], PRED_BONE_COLOR, 2, cv2.LINE_AA)

    for px, py in pixels.values():
        cv2.circle(frame, (px, py), 3, PRED_JOINT_COLOR, -1, cv2.LINE_AA)


# ── Player ────────────────────────────────────────────────────────────────────

def play(video_path, gt_frames, pred_frames, pred_meta, disp_w, disp_h):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    win_w, win_h = _display_size(orig_w, orig_h, disp_w, disp_h)

    pred_w = pred_meta.get("processed_frame_w") or orig_w
    pred_h = pred_meta.get("processed_frame_h") or orig_h
    scale_x, scale_y = win_w / pred_w, win_h / pred_h

    print(f"Video: {orig_w}x{orig_h}  {fps:.1f} fps  {total} frames")
    print(f"GT skeleton frames: {len(gt_frames)}")
    print(f"Prediction frames:  {len(pred_frames)} (of {pred_meta.get('total_frames_processed', '?')} processed)")

    win_name = "Prediction (orange) vs GT (cyan/magenta)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, win_w, win_h)

    delay = max(1, int(1000 / fps))
    paused = False
    frame_i = 0

    def render(raw_frame, idx):
        frame = resize_to(raw_frame, disp_w, disp_h)
        if idx < len(gt_frames):
            draw_skeleton(frame, gt_frames[idx], orig_w, orig_h)
        if idx in pred_frames:
            draw_prediction(frame, pred_frames[idx], scale_x, scale_y)

        info = f"Frame {idx}/{total}   [SPACE=pause  n=step  q=quit]"
        cv2.putText(frame, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    while True:
        if not paused:
            ret, raw_frame = cap.read()
            if not ret:
                break
            cv2.imshow(win_name, render(raw_frame, frame_i))
            frame_i += 1

        key = cv2.waitKey(1 if paused else delay) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('n') and paused:
            ret, raw_frame = cap.read()
            if ret:
                cv2.imshow(win_name, render(raw_frame, frame_i))
                frame_i += 1

    cap.release()
    cv2.destroyAllWindows()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize a pose_detection.py prediction JSON against the NTU RGB+D ground-truth skeleton",
    )
    p.add_argument("--keypoints", "-k", required=True,
                    help="Path to the *_keypoints_full_body.json produced by pose_detection.py")
    p.add_argument("--video", "-v",
                    help="Override video path (default: metadata.source in the JSON)")
    p.add_argument("--skeleton", "-s",
                    help="Override .skeleton GT path (default: derived from the JSON's source file name)")
    p.add_argument("--skeleton-folder", "-F", default=DEFAULT_SKELETON_FOLDER,
                    help="Folder to search for the .skeleton GT file")
    p.add_argument("--width",  "-W", type=int, default=800, help="Display width  (default: 800)")
    p.add_argument("--height", "-H", type=int, default=0,   help="Display height (default: auto)")
    
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.keypoints):
        print(f"Keypoints JSON not found: {args.keypoints}"); raise SystemExit(1)

    meta, pred_frames = load_predictions(args.keypoints)
    if meta.get("mode") != "body":
        print(f"Warning: JSON mode is '{meta.get('mode')}', expected 'body'")

    video_path = resolve_video_path(meta, args.video)
    if not os.path.isfile(video_path):
        print(f"Video not found: {video_path}"); raise SystemExit(1)

    skeleton_path = resolve_skeleton_path(meta, args.skeleton, args.skeleton_folder)
    if not os.path.isfile(skeleton_path):
        print(f"GT skeleton not found: {skeleton_path}"); raise SystemExit(1)

    print(f"Video:    {video_path}")
    print(f"Skeleton: {skeleton_path}")
    gt_frames = parse_skeleton(skeleton_path)

    play(video_path, gt_frames, pred_frames, meta, args.width, args.height)


if __name__ == "__main__":
    main()
