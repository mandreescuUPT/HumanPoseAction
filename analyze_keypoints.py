"""
Analyze Keypoints JSON
======================
Read exported JSON-ul by pose_detector.py and compute movement metrics 
for automotive drowsiness/gesture analysis.

Usage:
  python analyze_keypoints.py --input output/keypoints_full.json --mode body
"""

import json
import argparse
import math
from pathlib import Path

from utils import dist2d, velocity, center_of_mass

def analyze_body(frames):
    """
    Compute:
      - head_drop_angle: unghiul de aplecare a capului (drowsiness proxy)
      - shoulder_asymmetry: diferența de înălțime umeri
      - nose velocity  (proxy pentru mișcare generală cap)
      - com_velocity: speed of body center of mass (px/s)
    """
    results = []
    prev_nose = None
    prev_com  = None
    prev_ts   = None

    for frame in frames:
        if not frame["detected"] or frame["keypoints"] is None:
            results.append({"frame_id": frame["frame_id"], "detected": False})
            continue

        kps = frame["keypoints"]
        ts  = frame["timestamp_s"]

        nose         = kps.get("NOSE")
        left_ear     = kps.get("LEFT_EAR")
        right_ear    = kps.get("RIGHT_EAR")
        left_shoulder  = kps.get("LEFT_SHOULDER")
        right_shoulder = kps.get("RIGHT_SHOULDER")

        # Head tilt: unghi față de orizontală între urechi
        head_tilt_deg = None
        if left_ear and right_ear:
            dx = right_ear["x_px"] - left_ear["x_px"]
            dy = right_ear["y_px"] - left_ear["y_px"]
            head_tilt_deg = round(math.degrees(math.atan2(dy, dx)), 2)

        # Head drop: distanța verticală nas față de linia umerilor
        head_drop = None
        if nose and left_shoulder and right_shoulder:
            shoulder_y = (left_shoulder["y_px"] + right_shoulder["y_px"]) / 2
            head_drop = round(shoulder_y - nose["y_px"], 2)  # pozitiv = cap sus

        # Shoulder asymmetry
        shoulder_asym = None
        if left_shoulder and right_shoulder:
            shoulder_asym = round(abs(left_shoulder["y_px"] - right_shoulder["y_px"]), 2)

        # Head velocity
        nose_velocity = None
        if prev_nose and nose and prev_ts is not None:
            dt = ts - prev_ts
            nose_velocity = round(velocity(prev_nose, nose, dt), 2)

        # Center of mass velocity
        com = center_of_mass(kps)
        com_velocity = None
        if prev_com and com and prev_ts is not None:
            dt = ts - prev_ts
            com_velocity = round(velocity(prev_com, com, dt), 2)

        results.append({
            "frame_id":           frame["frame_id"],
            "timestamp_s":        ts,
            "detected":           True,
            "head_tilt_deg":      head_tilt_deg,
            "head_drop_px":       head_drop,
            "shoulder_asym_px":   shoulder_asym,
            "nose_velocity_px_s": nose_velocity,
            "com_velocity_px_s":  com_velocity,
        })

        prev_nose = nose
        prev_com  = com
        prev_ts   = ts

    return results


# ── Face metrics ───────────────────────────────────────────────────────────────
# MediaPipe FaceMesh landmark IDs relevante
EYE_LEFT_TOP    = 386
EYE_LEFT_BOTTOM = 374
EYE_LEFT_LEFT   = 263
EYE_LEFT_RIGHT  = 362

EYE_RIGHT_TOP    = 159
EYE_RIGHT_BOTTOM = 145
EYE_RIGHT_LEFT   = 33
EYE_RIGHT_RIGHT  = 133

def eye_aspect_ratio(lms, top_id, bottom_id, left_id, right_id):
    """EAR = raport înălțime/lățime ochi. Scade la închidere (blink/drowsiness)."""
    try:
        h = math.dist(
            (lms[top_id]["x_px"],   lms[top_id]["y_px"]),
            (lms[bottom_id]["x_px"], lms[bottom_id]["y_px"])
        )
        w = math.dist(
            (lms[left_id]["x_px"],  lms[left_id]["y_px"]),
            (lms[right_id]["x_px"], lms[right_id]["y_px"])
        )
        return round(h / w, 4) if w > 0 else 0.0
    except (KeyError, IndexError):
        return None


def analyze_face(frames):
    """
    Calculează EAR (Eye Aspect Ratio) pentru ambii ochi.
    EAR < 0.2 indică ochi închis (blink sau somnolență).
    """
    results = []
    for frame in frames:
        if not frame["detected"] or frame["keypoints"] is None:
            results.append({"frame_id": frame["frame_id"], "detected": False})
            continue

        # keypoints pentru face e o listă de fețe, fiecare o listă de dicts
        faces = frame["keypoints"]
        face_results = []

        for face_lms_list in faces:
            # Convertim lista în dict indexat pe id
            lms = {lm["id"]: lm for lm in face_lms_list}

            ear_left  = eye_aspect_ratio(lms, EYE_LEFT_TOP, EYE_LEFT_BOTTOM,
                                         EYE_LEFT_LEFT, EYE_LEFT_RIGHT)
            ear_right = eye_aspect_ratio(lms, EYE_RIGHT_TOP, EYE_RIGHT_BOTTOM,
                                         EYE_RIGHT_LEFT, EYE_RIGHT_RIGHT)

            ear_avg = round((ear_left + ear_right) / 2, 4) \
                if ear_left and ear_right else None

            face_results.append({
                "ear_left":  ear_left,
                "ear_right": ear_right,
                "ear_avg":   ear_avg,
                "blink_detected": ear_avg is not None and ear_avg < 0.20,
            })

        results.append({
            "frame_id":    frame["frame_id"],
            "timestamp_s": frame["timestamp_s"],
            "detected":    True,
            "faces":       face_results,
        })
    return results


# ── Hand metrics ───────────────────────────────────────────────────────────────
def analyze_hands(frames):
    """
    Calculează:
      - finger_spread: distance between fingers (apertura mână)
      - wrist_velocity: speed of wrist (proxy pentru gesture speed)
    """
    results = []
    prev_wrists = {}
    prev_ts = None

    FINGERTIPS = ["THUMB_TIP", "INDEX_TIP", "MIDDLE_TIP", "RING_TIP", "PINKY_TIP"]

    for frame in frames:
        if not frame["detected"] or frame["keypoints"] is None:
            results.append({"frame_id": frame["frame_id"], "detected": False})
            prev_ts = frame.get("timestamp_s")
            continue

        ts = frame["timestamp_s"]
        hands_out = []

        for hand_data in frame["keypoints"]:
            label = hand_data["hand"]
            kps   = hand_data["keypoints"]

            # Finger spread (distanța medie între vârfuri consecutive)
            tips = [kps[t] for t in FINGERTIPS if t in kps]
            spread = 0.0
            if len(tips) >= 2:
                dists = [dist2d(tips[i], tips[i+1]) for i in range(len(tips)-1)]
                spread = round(sum(dists) / len(dists), 2)

            # Wrist velocity
            wrist = kps.get("WRIST")
            wrist_vel = None
            if wrist and label in prev_wrists and prev_ts is not None:
                dt = ts - prev_ts
                wrist_vel = round(velocity(prev_wrists[label], wrist, dt), 2)
            if wrist:
                prev_wrists[label] = wrist

            hands_out.append({
                "hand":           label,
                "finger_spread_px": spread,
                "wrist_velocity_px_s": wrist_vel,
            })

        results.append({
            "frame_id":    frame["frame_id"],
            "timestamp_s": ts,
            "detected":    True,
            "hands":       hands_out,
        })
        prev_ts = ts

    return results


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Analyze JSON keypoints exported by pose_detector.py")
    parser.add_argument("--input",  "-i", required=True, help="JSON input")
    parser.add_argument("--mode",   "-m", choices=["body", "face", "hands"], default="body")
    parser.add_argument("--output", "-o", default=None, help="JSON output (default: <input>_metrics.json)")
    args = parser.parse_args()

    input_path = Path(args.input)
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    frames = data["frames"]
    meta   = data["metadata"]

    print(f"\nAnalyza: {input_path.name}")
    print(f"  Mode: {args.mode}  |  Frames: {len(frames)}")

    if args.mode == "body":
        metrics = analyze_body(frames)
    elif args.mode == "face":
        metrics = analyze_face(frames)
    elif args.mode == "hands":
        metrics = analyze_hands(frames)

    # Statistics
    detected_frames = [m for m in metrics if m.get("detected")]
    print(f"  Frames with detection: {len(detected_frames)}/{len(frames)}")

    if args.mode == "body" and detected_frames:
        tilts = [m["head_tilt_deg"] for m in detected_frames if m.get("head_tilt_deg") is not None]
        if tilts:
            print(f"  Head tilt — min: {min(tilts):.1f}°  max: {max(tilts):.1f}°  avg: {sum(tilts)/len(tilts):.1f}°")
        com_speeds = [m["com_velocity_px_s"] for m in detected_frames if m.get("com_velocity_px_s") is not None]
        if com_speeds:
            print(f"  COM speed  — min: {min(com_speeds):.1f}  max: {max(com_speeds):.1f}  avg: {sum(com_speeds)/len(com_speeds):.1f} px/s")

    if args.mode == "face" and detected_frames:
        ears = [m["faces"][0]["ear_avg"] for m in detected_frames
                if m.get("faces") and m["faces"][0].get("ear_avg")]
        if ears:
            blinks = sum(1 for e in ears if e < 0.20)
            print(f"  EAR avg: {sum(ears)/len(ears):.3f}  |  Blink events (EAR<0.20): {blinks}")

    # Save metrics
    out_path = Path(args.output) if args.output else input_path.with_name(
        input_path.stem + "_metrics.json"
    )
    output_data = {
        "metadata": meta,
        "analysis_mode": args.mode,
        "metrics": metrics,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"  Saved metrics: {out_path}\n")


if __name__ == "__main__":
    main()
