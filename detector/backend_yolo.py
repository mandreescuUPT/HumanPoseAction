"""
Ultralytics YOLO11-Pose Backend
================================
Uses `ultralytics` to run YOLO11-Pose for 17-keypoint COCO body estimation.
When multiple persons are detected, the highest-confidence one is kept.

Requires:  pip install ultralytics
Model auto-downloads on first use (~6 MB for yolo11n-pose).
"""

from config.constants import COCO_KEYPOINT_NAMES
from detector.base_backend import BasePoseBackend


class YOLOBackend(BasePoseBackend):

    def __init__(self, confidence: float = 0.5, model_name: str = "yolo11n-pose.pt"):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "Ultralytics is required for the YOLO backend.\n"
                "Install it with:  pip install ultralytics"
            )
        self._model = YOLO(model_name)
        self._confidence = confidence

    @property
    def backend_name(self) -> str:
        return "yolo"

    @property
    def keypoint_names(self) -> dict[int, str]:
        return COCO_KEYPOINT_NAMES

    def process(self, frame_rgb, frame_w: int, frame_h: int) -> dict | None:
        # Ultralytics expects BGR; frame_rgb is RGB — convert in-place
        import cv2
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        results = self._model(
            frame_bgr,
            conf=self._confidence,
            verbose=False,
        )

        result = results[0]

        # No detections
        if result.keypoints is None or len(result.keypoints) == 0:
            return None

        kpts = result.keypoints       # ultralytics Keypoints object
        confs = result.boxes.conf      # detection confidence per person

        if len(confs) == 0:
            return None

        # Pick the person with highest detection confidence
        best_idx = int(confs.argmax())

        # kpts.data shape: (num_persons, 17, 3)  — x_px, y_px, kp_confidence
        person_kpts = kpts.data[best_idx]   # (17, 3)

        keypoints = {}
        for idx in range(person_kpts.shape[0]):
            name = COCO_KEYPOINT_NAMES.get(idx, f"kp_{idx}")
            x_px = float(person_kpts[idx, 0])
            y_px = float(person_kpts[idx, 1])
            vis  = float(person_kpts[idx, 2])

            x_norm = x_px / frame_w if frame_w > 0 else 0.0
            y_norm = y_px / frame_h if frame_h > 0 else 0.0

            keypoints[name] = {
                "id":         idx,
                "x_norm":     round(x_norm, 6),
                "y_norm":     round(y_norm, 6),
                "z_norm":     0.0,          # YOLO-Pose is 2D — no depth
                "x_px":       int(x_px),
                "y_px":       int(y_px),
                "visibility": round(vis, 4),
            }

        return keypoints

    def close(self):
        # Ultralytics models don't need explicit cleanup
        pass
