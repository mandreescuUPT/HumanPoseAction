import mediapipe as mp

mp_pose     = mp.solutions.pose
mp_face     = mp.solutions.face_mesh
mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils
mp_styles   = mp.solutions.drawing_styles

POSE_LANDMARK_NAMES = {i: lm.name for i, lm in enumerate(mp_pose.PoseLandmark)}

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
            raise ValueError(f"Unknown mode: {mode}. Choose: body | face | hands")

    def process(self, frame_rgb):
        return self.detector.process(frame_rgb)

    def close(self):
        if self.detector:
            self.detector.close()

