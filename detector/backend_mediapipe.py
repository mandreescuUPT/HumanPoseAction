"""
MediaPipe BlazePose Backend
============================
Wraps the existing PoseDetector + KeyPointsExtractor for body-mode detection.
Output is identical to the original pipeline — this is a thin adapter.
"""

import mediapipe as mp

from config.constants import *
from detector.base_backend import BasePoseBackend

mp_pose = mp.solutions.pose

POSE_LANDMARK_NAMES = {i: lm.name for i, lm in enumerate(mp_pose.PoseLandmark)}


class MediaPipeBackend(BasePoseBackend):

    def __init__(self, confidence: float = 0.5):
        self._detector = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=confidence,
            min_tracking_confidence=0.5,
        )

    @property
    def backend_name(self) -> str:
        return "mediapipe"

    @property
    def keypoint_names(self) -> dict[int, str]:
        return POSE_LANDMARK_NAMES

    def process(self, frame_rgb, frame_w: int, frame_h: int) -> dict | None:
        frame_rgb.flags.writeable = False
        results = self._detector.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if not results.pose_landmarks:
            return None

        keypoints = {}
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            name = POSE_LANDMARK_NAMES.get(idx, f"kp_{idx}")
            keypoints[name] = {
                "id":         idx,
                "x_norm":     round(lm.x, 6),
                "y_norm":     round(lm.y, 6),
                "z_norm":     round(lm.z, 6),
                "x_px":       int(lm.x * frame_w),
                "y_px":       int(lm.y * frame_h),
                "visibility": round(lm.visibility, 4),
            }
        return keypoints

    def close(self):
        if self._detector:
            self._detector.close()
