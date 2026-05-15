import mediapipe as mp

from config.constants import *

mp_pose     = mp.solutions.pose
mp_face     = mp.solutions.face_mesh
mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils
mp_styles   = mp.solutions.drawing_styles

POSE_LANDMARK_NAMES = {i: lm.name for i, lm in enumerate(mp_pose.PoseLandmark)}

# ── Core detector class ────────────────────────────────────────────────────────
class PoseDetector:
    def __init__(self, mode: str = "body", 
                 min_detection_confidence: float = 0.5, 
                 static_image: bool = False):
        self.mode = mode
        self.detector = None

        if mode == "body":
            self.detector = mp_pose.Pose(
                static_image_mode=static_image,
                model_complexity=1,               # 0=lite, 1=full, 2=heavy
                enable_segmentation=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=0.5,
            )
        elif mode == "face":
            self.detector = mp_face.FaceMesh(
                static_image_mode=static_image,
                max_num_faces=1,
                refine_landmarks=True,            # include iris landmarks
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=0.5,
            )
        elif mode == "hands":
            self.detector = mp_hands.Hands(
                static_image_mode=static_image,
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

class KeyPointsExtractor:
    def __init__(self, results):
        self.results = results
        # self.landmarks = results.pose_landmarks.landmark if results.pose_landmarks else []
        # self.points = {POSE_LANDMARK_NAMES[i]: (lm.x, lm.y, lm.z) for i, lm in enumerate(self.landmarks)}

    def body_keypoints(self, frame_w, frame_h):
        """Extrage 33 keypoints de corp cu coordonate normalizate și pixel."""
        if not self.results.pose_landmarks:
            return None

        keypoints = {}
        for idx, lm in enumerate(self.results.pose_landmarks.landmark):
            name = POSE_LANDMARK_NAMES.get(idx, f"kp_{idx}")
            keypoints[name] = {
                "id": idx,
                "x_norm": round(lm.x, 6),
                "y_norm": round(lm.y, 6),
                "z_norm": round(lm.z, 6),         # relative depth (negative = in front of camera)
                "x_px":   int(lm.x * frame_w),
                "y_px":   int(lm.y * frame_h),
                "visibility": round(lm.visibility, 4),
            }
        return keypoints
    
    # def extract_face_keypoints(results, frame_w, frame_h):
    # """Extrage primele 468 (+ 10 iris) landmark-uri faciale."""
    # if not results.multi_face_landmarks:
    #     return None

    # face_data = []
    # for face_lms in results.multi_face_landmarks:
    #     kps = []
    #     for idx, lm in enumerate(face_lms.landmark):
    #         kps.append({
    #             "id":     idx,
    #             "x_norm": round(lm.x, 6),
    #             "y_norm": round(lm.y, 6),
    #             "z_norm": round(lm.z, 6),
    #             "x_px":   int(lm.x * frame_w),
    #             "y_px":   int(lm.y * frame_h),
    #         })
    #     face_data.append(kps)
    # return face_data

    def face_keypoints(self, frame_w, frame_h):
        """Extrage primele 468 (+ 10 iris) landmark-uri faciale."""
        if not self.results.multi_face_landmarks:
            return None

        face_data = []
        for face_lms in self.results.multi_face_landmarks:
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
    
    def hands_keypoints(self, frame_w, frame_h):
        """Extrage 21 keypoints per mână, cu eticheta Left/Right."""
        if not self.results.multi_hand_landmarks:
            return None

        hands_data = []
        handedness = self.results.multi_handedness if self.results.multi_handedness else []

        for i, hand_lms in enumerate(self.results.multi_hand_landmarks):
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