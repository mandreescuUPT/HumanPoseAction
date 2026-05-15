import mediapipe as mp


mp_pose     = mp.solutions.pose
mp_face     = mp.solutions.face_mesh
mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils
mp_styles   = mp.solutions.drawing_styles

class PoseDrawing:
    def __init__(self, results):
        self.results = results
    
    def draw_body(self, frame):
        if self.results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                self.results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                # landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 255, 120), thickness=2, circle_radius=3
                    ),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(255, 200, 0), thickness=2
                ),
            )

    def draw_face(self, frame):
        if self.results.face_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                self.results.face_landmarks,
                mp_face.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
            )

    def draw_hands(self, frame):
        if self.results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                self.results.left_hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
            )
        if self.results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                self.results.right_hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
            )