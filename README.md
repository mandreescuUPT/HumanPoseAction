# Pose Detection Pipeline

Detectează keypoints din video/webcam, vizualizează în OpenCV și salvează automat în JSON per frame.
Relevant pentru use case-urile de biometrică automotive (drowsiness, head pose, gesture, grip).

## Instalare

```bash
pip install -r requirements.txt
```

## Rulare

### Body pose (33 keypoints — head pose, drowsiness)
```bash
python pose_detection.py --input video.mp4 --mode body
python pose_detection.py --input 0 --mode body          # webcam
```

### Face mesh (468 keypoints + iris — blink rate, gaze)
```bash
python pose_detection.py --input video.mp4 --mode face
```

### Hands (21 keypoints × 2 mâini — gesture, grip)
```bash
python pose_detection.py --input video.mp4 --mode hands
```

### Headless (fără fereastră OpenCV, mai rapid pentru batch)
```bash
python pose_detection.py --input video.mp4 --mode body --no-display
```

### Toți parametrii
```
--input       Cale video sau '0' webcam
--mode        body | face | hands
--output      Director output (default: ./output)
--no-display  Rulare fără fereastră
--save-every  Checkpoint JSON la fiecare N frames (default: 500)
--confidence  Prag detecție 0.0–1.0 (default: 0.5)
```

## Analiză metrici

```bash
# Calculează head tilt, head drop, shoulder asymmetry, velocity
python analyze_keypoints.py --input output/keypoints_full.json --mode body

# Calculează EAR (Eye Aspect Ratio) pentru blink/drowsiness
python analyze_keypoints.py --input output/keypoints_full.json --mode face

# Calculează finger spread, wrist velocity
python analyze_keypoints.py --input output/keypoints_full.json --mode hands
```

---

## Format JSON exportat

```json
{
  "metadata": {
    "source": "video.mp4",    
    "mode": "body",
    "timestamp": "2026-05-11T10:30:00",
    "frame_w": 1280,
    "frame_h": 720,
    "source_fps": 30.0,
    "total_frames_processed": 900,
    "total_detections": 847
  },
  "frames": [
    {
      "frame_id": 0,
      "timestamp_s": 0.0,
      "detected": true,
      "keypoints": {
        "NOSE": {
          "id": 0,
          "x_norm": 0.512,
          "y_norm": 0.234,
          "z_norm": -0.041,
          "x_px": 655,
          "y_px": 168,
          "visibility": 0.9987
        },
        "LEFT_EAR": { ... },
        "RIGHT_EAR": { ... },
        "LEFT_SHOULDER": { ... },
        ...
      }
    }
  ]
}
```

### Body keypoints (33 total)
```
NOSE, LEFT_EYE_INNER, LEFT_EYE, LEFT_EYE_OUTER,
RIGHT_EYE_INNER, RIGHT_EYE, RIGHT_EYE_OUTER,
LEFT_EAR, RIGHT_EAR, MOUTH_LEFT, MOUTH_RIGHT,
LEFT_SHOULDER, RIGHT_SHOULDER,
LEFT_ELBOW, RIGHT_ELBOW,
LEFT_WRIST, RIGHT_WRIST,
LEFT_PINKY, RIGHT_PINKY, LEFT_INDEX, RIGHT_INDEX,
LEFT_THUMB, RIGHT_THUMB,
LEFT_HIP, RIGHT_HIP,
LEFT_KNEE, RIGHT_KNEE,
LEFT_ANKLE, RIGHT_ANKLE,
LEFT_HEEL, RIGHT_HEEL,
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX
```

### Coordonate
- `x_norm`, `y_norm`, `z_norm` — normalizate [0.0, 1.0]
- `x_px`, `y_px` — pixeli absoluti în frame
- `z_norm` — adâncime relativă față de sold (body) sau fata (face)
- `visibility` — probabilitate ca keypoint-ul e vizibil (body only)

---

### Vizualizare
```bash
# Miscarea punctelor se poate vizualiza cu ajutorul unui script care 
# primeste ca argument fisierul *.json in care avem punctele salvate 
# pentru fiecare frame.  
python animate_keypoints.py --input output/walk_normal_keypoints_full.json 
```

## Metrici calculate de analyze_keypoints.py (momentan in lucru)

| Mode  | Metrică | Use Case |
|-------|---------|----------|
| body  | `head_tilt_deg` | Detectie cap aplecat lateral |
| body  | `head_drop_px` | Detecție cap căzut (drowsiness) |
| body  | `shoulder_asym_px` | Postura generala |
| body  | `nose_velocity_px_s` | Viteza mișcării capului |
| face  | `ear_left`, `ear_right` | Eye Aspect Ratio per ochi |
| face  | `ear_avg` | EAR mediu |
| face  | `blink_detected` | EAR < 0.20 = ochi închis |
| hands | `finger_spread_px` | Apertura mâinii |
| hands | `wrist_velocity_px_s` | Viteza gestului |

---

## Integrare cu modelul matematic (momentan in lucru)

Keypoints-urile exportate pot fi folosite direct pentru:
- **Vectorul de mișcare**: `v(t) = [Δx_px/Δt, Δy_px/Δt]` per keypoint
- **Head pose (Euler angles)**: din poziția relativă nas/urechi/umeri
- **Motion compensation**: warp events pe baza flow-ului estimat din Δkeypoints
- **Kalman filter**: starea `[x, y, vx, vy]` per keypoint
