"""
Pose Detection Pipeline
========================
Detectează keypoints din video/webcam folosind MediaPipe BlazePose,
vizualizează în OpenCV și salvează automat în JSON per frame.

Use cases automotive biometrics:
  - Body pose  → head pose, drowsiness (cap aplecat)
  - Face mesh  → blink rate, gaze direction
  - Hands      → gesture recognition, grip detection

Instalare:
  pip install mediapipe opencv-python

Rulare:
  python pose_detector.py --input video.mp4 --mode body
  python pose_detector.py --input 0 --mode face          (webcam)
  python pose_detector.py --input video.mp4 --mode hands
  python pose_detector.py --input video.mp4 --mode body --no-display
"""

import cv2
import time
import argparse
from pathlib import Path
from datetime import datetime

# ── MediaPipe imports ──────────────────────────────────────────────────────────
import mediapipe as mp

from utils import save_json

mp_pose     = mp.solutions.pose
mp_face     = mp.solutions.face_mesh
mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils
mp_styles   = mp.solutions.drawing_styles


# ── Keypoint name maps ─────────────────────────────────────────────────────────
POSE_LANDMARK_NAMES = {i: lm.name for i, lm in enumerate(mp_pose.PoseLandmark)}

HAND_LANDMARK_NAMES = {
    0: "WRIST",
    1: "THUMB_CMC", 2: "THUMB_MCP", 3: "THUMB_IP", 4: "THUMB_TIP",
    5: "INDEX_MCP",  6: "INDEX_PIP",  7: "INDEX_DIP",  8: "INDEX_TIP",
    9: "MIDDLE_MCP", 10: "MIDDLE_PIP", 11: "MIDDLE_DIP", 12: "MIDDLE_TIP",
    13: "RING_MCP",  14: "RING_PIP",  15: "RING_DIP",  16: "RING_TIP",
    17: "PINKY_MCP", 18: "PINKY_PIP", 19: "PINKY_DIP", 20: "PINKY_TIP",
}


# ── Core detector class ────────────────────────────────────────────────────────
class PoseDetector:
    def __init__(self, mode: str = "body", min_detection_confidence: float = 0.5):
        self.mode = mode
        self.detector = None

        if mode == "body":
            self.detector = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,               # 0=lite, 1=full, 2=heavy
                enable_segmentation=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=0.5,
            )
        elif mode == "face":
            self.detector = mp_face.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,            # include iris landmarks
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=0.5,
            )
        elif mode == "hands":
            self.detector = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=1,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=0.5,
            )
        else:
            raise ValueError(f"Mode necunoscut: {mode}. Alege: body | face | hands")

    def process(self, frame_rgb):
        return self.detector.process(frame_rgb)

    def close(self):
        if self.detector:
            self.detector.close()


# ── Keypoint extraction ────────────────────────────────────────────────────────
def extract_body_keypoints(results, frame_w, frame_h):
    """Extrage 33 keypoints de corp cu coordonate normalizate și pixel."""
    if not results.pose_landmarks:
        return None

    keypoints = {}
    for idx, lm in enumerate(results.pose_landmarks.landmark):
        name = POSE_LANDMARK_NAMES.get(idx, f"kp_{idx}")
        keypoints[name] = {
            "id": idx,
            "x_norm": round(lm.x, 6),
            "y_norm": round(lm.y, 6),
            "z_norm": round(lm.z, 6),         # adâncime relativă
            "x_px":   int(lm.x * frame_w),
            "y_px":   int(lm.y * frame_h),
            "visibility": round(lm.visibility, 4),
        }
    return keypoints


def extract_face_keypoints(results, frame_w, frame_h):
    """Extrage primele 468 (+ 10 iris) landmark-uri faciale."""
    if not results.multi_face_landmarks:
        return None

    face_data = []
    for face_lms in results.multi_face_landmarks:
        kps = []
        for idx, lm in enumerate(face_lms.landmark):
            kps.append({
                "id":     idx,
                "x_norm": round(lm.x, 6),
                "y_norm": round(lm.y, 6),
                "z_norm": round(lm.z, 6),
                "x_px":   int(lm.x * frame_w),
                "y_px":   int(lm.y * frame_h),
            })
        face_data.append(kps)
    return face_data


def extract_hand_keypoints(results, frame_w, frame_h):
    """Extrage 21 keypoints per mână, cu eticheta Left/Right."""
    if not results.multi_hand_landmarks:
        return None

    hands_data = []
    handedness = results.multi_handedness if results.multi_handedness else []

    for i, hand_lms in enumerate(results.multi_hand_landmarks):
        label = handedness[i].classification[0].label if i < len(handedness) else "Unknown"
        kps = {}
        for idx, lm in enumerate(hand_lms.landmark):
            name = HAND_LANDMARK_NAMES.get(idx, f"kp_{idx}")
            kps[name] = {
                "id":   idx,
                "x_norm": round(lm.x, 6),
                "y_norm": round(lm.y, 6),
                "z_norm": round(lm.z, 6),
                "x_px": int(lm.x * frame_w),
                "y_px": int(lm.y * frame_h),
            }
        hands_data.append({"hand": label, "keypoints": kps})
    return hands_data


# ── Visualization ──────────────────────────────────────────────────────────────
def draw_body(frame, results):
    mp_drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
            color=(0, 255, 120), thickness=2, circle_radius=3
        ),
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(255, 200, 0), thickness=2
        ),
    )


def draw_face(frame, results):
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
            )
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style(),
            )
            # Iris (dacă refine_landmarks=True)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_iris_connections_style(),
            )


def draw_hands(frame, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )


def draw_overlay(frame, frame_idx, fps, detection_count, mode):
    """HUD info pe frame."""
    h, w = frame.shape[:2]
    # Fundal semi-transparent
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    text = f"Mode: {mode.upper()}  |  Frame: {frame_idx:05d}  |  FPS: {fps:.1f}  |  Detected: {detection_count}"
    cv2.putText(frame, text, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 1, cv2.LINE_AA)

    # Instrucțiuni
    cv2.putText(frame, "Q = quit  |  S = save snapshot", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


def resize_frame(frame, max_size: int):
    if max_size <= 0:
        return frame

    h, w = frame.shape[:2]
    longest_edge = max(w, h)
    if longest_edge <= max_size:
        return frame

    scale = max_size / longest_edge
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ── Main pipeline ───────────────────────────────────────────────────────────────
def run(input_source, mode: str, output_dir: Path, show_display: bool,
        save_every: int, confidence: float, max_size: int):

    output_dir.mkdir(parents=True, exist_ok=True)

    # Open video source (file or webcam)
    cap_source = 0 if input_source == "0" else input_source
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {input_source}")

    original_frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = original_frame_w
    frame_h = original_frame_h
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"\n{'='*55}")
    print(f"  Pose Detector — mode: {mode.upper()}")
    print(f"  Source:  {input_source}")
    print(f"  Frames: {total_frames if total_frames > 0 else 'live'}")
    print(f"  Original res: {original_frame_w}x{original_frame_h} @ {source_fps:.1f} fps")
    if max_size > 0:
        print(f"  Max frame size: {max_size} px")
    print(f"  Output: {output_dir}")
    print(f"{'='*55}\n")

    detector = PoseDetector(mode=mode, min_detection_confidence=confidence)

    source_path = Path(input_source) if input_source != "0" else None
    source_file = source_path.stem if source_path else "webcam"

    # Final JSON structure
    session_data = {
        "metadata": {
            "source":               str(input_source),
            "source_file":          source_file,
            "mode":                 mode,
            "timestamp":            datetime.now().isoformat(),
            "original_frame_w":     original_frame_w,
            "original_frame_h":     original_frame_h,
            "processed_frame_w":    frame_w,
            "processed_frame_h":    frame_h,
            "source_fps":           source_fps,
            "max_frame_size":       max_size,
        },
        "frames": []
    }

    frame_idx      = 0
    detected_count = 0
    fps_timer      = time.time()
    fps_display    = 0.0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = resize_frame(frame, max_size)
            frame_h, frame_w = frame.shape[:2]
            session_data["metadata"]["processed_frame_w"] = frame_w
            session_data["metadata"]["processed_frame_h"] = frame_h

            # FPS calc
            now = time.time()
            elapsed = now - fps_timer
            if elapsed > 0:
                fps_display = 1.0 / elapsed
            fps_timer = now

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = detector.process(frame_rgb)
            frame_rgb.flags.writeable = True

            # Extrage keypoints
            kp_data   = None
            det_count = 0

            if mode == "body":
                kp_data = extract_body_keypoints(results, frame_w, frame_h)
                det_count = 1 if kp_data else 0
                if show_display:
                    draw_body(frame, results)

            elif mode == "face":
                kp_data = extract_face_keypoints(results, frame_w, frame_h)
                det_count = len(kp_data) if kp_data else 0
                if show_display:
                    draw_face(frame, results)

            elif mode == "hands":
                kp_data = extract_hand_keypoints(results, frame_w, frame_h)
                det_count = len(kp_data) if kp_data else 0
                if show_display:
                    draw_hands(frame, results)

            detected_count += det_count

            if det_count > 0:

                # Add frame to JSON
                frame_entry = {
                    "frame_id":   frame_idx,
                    "timestamp_s": round(frame_idx / source_fps, 4),
                    "detected":   det_count > 0,
                    "keypoints":  kp_data,
                }
                session_data["frames"].append(frame_entry)

                # Save periodic JSON (checkpoint)
                if save_every > 0 and frame_idx > 0 and frame_idx % save_every == 0:
                    save_json(session_data, output_dir, f"checkpoint_{frame_idx:06d}.json")
                    print(f"  [checkpoint] frame {frame_idx} saved")

            # Display
            if show_display:
                draw_overlay(frame, frame_idx, fps_display, det_count, mode)
                cv2.imshow("Pose Detector", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n  [Q] Quit.")
                    break
                elif key == ord('s'):
                    snap_path = output_dir / f"snapshot_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(snap_path), frame)
                    print(f"  [S] Snapshot saved: {snap_path}")

            frame_idx += 1

            if frame_idx % 100 == 0:
                print(f"  Processing frame {frame_idx}/{total_frames if total_frames > 0 else '?'} ...")

    finally:
        cap.release()
        if show_display:
            cv2.destroyAllWindows()
        detector.close()

    # Save final JSON
    session_data["metadata"]["total_frames_processed"] = frame_idx
    session_data["metadata"]["total_detections"]       = detected_count
    json_path = save_json(session_data, output_dir, source_file + "_keypoints_full.json")

    # Delete checkpoints now that the full file is saved
    for cp in output_dir.glob("checkpoint_*.json"):
        cp.unlink()

    print(f"\n{'='*55}")
    print(f"  Done! Frames processed: {frame_idx}")
    print(f"  Total detections:        {detected_count}")
    print(f"  JSON saved:            {json_path}")
    print(f"{'='*55}\n")

    return json_path

# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Pose Detection → OpenCV visualization → JSON export"
    )
    parser.add_argument(
        "--input", "-i",
        default="0",
        help="Call video (ex: video.mp4) or '0' for webcam (default: 0)"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["body", "face", "hands"],
        default="body",
        help="Detection type: body (33 kp) | face (468 kp) | hands (21 kp) (default: body)"
    )
    parser.add_argument(
        "--output", "-o",
        default="./output",
        help="Output directory for JSON (default: ./output)"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Headless mode (no OpenCV window, faster)"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=500,
        help="Save checkpoint JSON every N frames (0 = disabled, default: 500)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum detection threshold (0.0 – 1.0, default: 0.5)"
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=0,
        help="Maximum frame edge size for OpenCV input. Frames larger than this will be resized while preserving aspect ratio. 0 = disabled"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        input_source=args.input,
        mode=args.mode,
        output_dir=Path(args.output),
        show_display=not args.no_display,
        save_every=args.save_every,
        confidence=args.confidence,
        max_size=args.max_size,
    )