"""
Single-Image Pose Detector
==========================
Runs pose / face / hand detection on one image file.

Usage:
  python detect_image.py --input photo.jpg --mode body
  python detect_image.py --input photo.jpg --mode face --no-display
  python detect_image.py --input photo.jpg --mode hands --output ./output
"""

import cv2
import json
import argparse
from pathlib import Path
from datetime import datetime

import mediapipe as mp

from detector import PoseDetector, POSE_LANDMARK_NAMES, PoseDrawing

from pose_detector import (
    PoseDetector,
    extract_body_keypoints,
    extract_face_keypoints,
    extract_hand_keypoints,    
    POSE_LANDMARK_NAMES,
    HAND_LANDMARK_NAMES,
)

mp_pose  = mp.solutions.pose
mp_face  = mp.solutions.face_mesh
mp_hands = mp.solutions.hands


def make_static_detector(mode: str, confidence: float):
    """Same as PoseDetector but with static_image_mode=True (no tracking)."""
    if mode == "body":
        return mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=confidence,
        )
    elif mode == "face":
        return mp_face.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=confidence,
        )
    elif mode == "hands":
        return mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=confidence,
        )
    raise ValueError(f"Unknown mode: {mode}. Choose: body | face | hands")


def run(image_path: Path, mode: str, output_dir: Path, show_display: bool, confidence: float):
    frame_bgr = cv2.imread(str(image_path))
    if frame_bgr is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")

    frame_h, frame_w = frame_bgr.shape[:2]

    print(f"\n{'='*50}")
    print(f"  Image Pose Detector — mode: {mode.upper()}")
    print(f"  Image:  {image_path}")
    print(f"  Size:   {frame_w}x{frame_h}")
    print(f"{'='*50}\n")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False

    detector = make_static_detector(mode, confidence)
    results = detector.process(frame_rgb)
    detector.close()

    frame_rgb.flags.writeable = True
    annotated = frame_bgr.copy()

    kp_data = None
    det_count = 0

    pose_drawing = PoseDrawing(results)

    if mode == "body":
        kp_data = extract_body_keypoints(results, frame_w, frame_h)
        det_count = 1 if kp_data else 0
        pose_drawing.draw_body(annotated)
    elif mode == "face":
        kp_data = extract_face_keypoints(results, frame_w, frame_h)
        det_count = len(kp_data) if kp_data else 0
        pose_drawing.draw_face(annotated)
    elif mode == "hands":
        kp_data = extract_hand_keypoints(results, frame_w, frame_h)
        det_count = len(kp_data) if kp_data else 0
        pose_drawing.draw_hands(annotated)

    status = f"Detected: {det_count}" if det_count else "No detection"
    print(f"  {status}")

    # HUD overlay
    cv2.rectangle(annotated, (0, 0), (frame_w, 34), (20, 20, 20), -1)
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (frame_w, 34), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
    cv2.putText(
        annotated,
        f"Mode: {mode.upper()}  |  {status}  |  {image_path.name}",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 1, cv2.LINE_AA,
    )

    # Save annotated image
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    out_img = output_dir / f"{stem}_annotated.jpg"
    cv2.imwrite(str(out_img), annotated)
    print(f"  Annotated image saved: {out_img}")

    # Save JSON
    result_data = {
        "metadata": {
            "source":     str(image_path),
            "mode":       mode,
            "timestamp":  datetime.now().isoformat(),
            "frame_w":    frame_w,
            "frame_h":    frame_h,
            "detected":   det_count > 0,
            "det_count":  det_count,
        },
        "keypoints": kp_data,
    }
    out_json = output_dir / f"{stem}_keypoints.json"
    out_json.write_text(json.dumps(result_data, indent=2), encoding="utf-8")
    print(f"  Keypoints JSON saved:  {out_json}")

    if show_display:
        cv2.imshow("Pose Detector — Image", annotated)
        print("\n  Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"\n{'='*50}\n")
    return out_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-image pose detection → annotated image + JSON"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Path to input image (jpg, png, bmp, ...)")
    parser.add_argument("--mode", "-m", choices=["body", "face", "hands"],
                        default="body",
                        help="Detection mode (default: body)")
    parser.add_argument("--output", "-o", default="./output",
                        help="Output directory (default: ./output)")
    parser.add_argument("--no-display", action="store_true",
                        help="Skip the OpenCV preview window")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Min detection confidence 0.0–1.0 (default: 0.5)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        image_path=Path(args.input),
        mode=args.mode,
        output_dir=Path(args.output),
        show_display=not args.no_display,
        confidence=args.confidence,
    )
