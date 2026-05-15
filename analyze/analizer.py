import numpy as np

from pose_detector import POSE_LANDMARK_NAMES

class BodyPosture:
    def __init__(self, keypoints):
        self.keypoints = keypoints

    def calculate_angles(self):
        # Implement angle calculation logic here
        pass

    def extract_body_keypoints(self, results, frame_w, frame_h):
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

class Movement:
    def __init__(self, keypoints_sequence):
        self.keypoints_sequence = keypoints_sequence

    def calculate_velocity(self):
        # Implement velocity calculation logic here
        pass

class Analyzer:
    def __init__(self, inference_results):
        self.inference_results = inference_results
        self._center_of_mass = None

    def analyze_body(self, frames):
        pass

    def center_of_mass(self, frame):
        if self._center_of_mass is None:
            # Calculate center of mass logic here
            pass
        return self._center_of_mass