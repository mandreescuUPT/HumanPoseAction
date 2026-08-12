from .pose_detector import PoseDetector, POSE_LANDMARK_NAMES, KeyPointsExtractor
from .pose_drawing import PoseDrawing
from .base_backend import BasePoseBackend

AVAILABLE_BACKENDS = ["mediapipe", "yolo", "movenet", "movenet_thunder", "movenet_lightning"]


def create_backend(name: str, confidence: float = 0.5):
    """Factory: instantiate a pose backend by name.

    Parameters
    ----------
    name : str
        One of 'mediapipe', 'yolo', 'movenet', 'movenet_thunder', 'movenet_lightning'.
    confidence : float
        Minimum detection confidence (0.0–1.0).

    Returns
    -------
    BasePoseBackend
    """
    if name == "mediapipe":
        from .backend_mediapipe import MediaPipeBackend
        return MediaPipeBackend(confidence=confidence)
    elif name == "yolo":
        from .backend_yolo import YOLOBackend
        return YOLOBackend(confidence=confidence)
    elif name in ["movenet", "movenet_thunder"]:
        from .backend_movenet import MoveNetBackend
        return MoveNetBackend(confidence=confidence, model_type="thunder")
    elif name == "movenet_lightning":
        from .backend_movenet import MoveNetBackend
        return MoveNetBackend(confidence=confidence, model_type="lightning")
    else:
        raise ValueError(
            f"Unknown backend: {name!r}. Choose from: {AVAILABLE_BACKENDS}"
        )