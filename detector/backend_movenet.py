"""
MoveNet Thunder Backend
========================
Uses TensorFlow / TFLite to run MoveNet Thunder for 17-keypoint COCO body
estimation (single-person).

Requires:  pip install tensorflow>=2.10   (or pip install tflite-runtime)
Model auto-downloads from TF Hub on first use.
"""

import numpy as np

from config.constants import COCO_KEYPOINT_NAMES
from detector.base_backend import BasePoseBackend

# TF Hub model URLs for MoveNet
_MOVENET_URLS = {
    "thunder": "https://tfhub.dev/google/movenet/singlepose/thunder/4",
    "lightning": "https://tfhub.dev/google/movenet/singlepose/lightning/4"
}
_MOVENET_INPUT_SIZES = {
    "thunder": 256,
    "lightning": 192
}


class MoveNetBackend(BasePoseBackend):

    def __init__(self, confidence: float = 0.5, model_type: str = "thunder"):
        self._confidence = confidence
        self._model_type = model_type
        if self._model_type not in _MOVENET_URLS:
            raise ValueError(f"Unknown movenet type: {model_type}")
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._input_size = None
        self._load_model()

    def _load_model(self):
        """Load MoveNet Thunder via TF Hub as a SavedModel, then wrap it."""
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError(
                "TensorFlow is required for the MoveNet backend.\n"
                "Install it with:  pip install tensorflow"
            )

        # Load the model from TF Hub
        import tensorflow_hub as hub
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        
        url = _MOVENET_URLS[self._model_type]
        self._module = hub.load(url)
        self._movenet = self._module.signatures["serving_default"]

        # MoveNet expects specific input size
        self._input_size = _MOVENET_INPUT_SIZES[self._model_type]

    @property
    def backend_name(self) -> str:
        return f"movenet_{self._model_type}"

    @property
    def keypoint_names(self) -> dict[int, str]:
        return COCO_KEYPOINT_NAMES

    def process(self, frame_rgb, frame_w: int, frame_h: int) -> dict | None:
        import tensorflow as tf

        # Resize to model input size (256x256 for Thunder)
        input_image = tf.image.resize_with_pad(
            tf.expand_dims(frame_rgb, axis=0),
            self._input_size,
            self._input_size,
        )
        input_image = tf.cast(input_image, dtype=tf.int32)

        # Run inference
        outputs = self._movenet(input_image)
        # Output shape: (1, 1, 17, 3) — [y_norm, x_norm, confidence]
        keypoints_with_scores = outputs["output_0"].numpy()[0, 0]  # (17, 3)

        # Check if any keypoint has sufficient confidence
        max_conf = float(np.max(keypoints_with_scores[:, 2]))
        if max_conf < self._confidence:
            return None

        keypoints = {}
        for idx in range(keypoints_with_scores.shape[0]):
            name = COCO_KEYPOINT_NAMES.get(idx, f"kp_{idx}")
            # MoveNet outputs [y, x, score] (note: y first!)
            y_norm = float(keypoints_with_scores[idx, 0])
            x_norm = float(keypoints_with_scores[idx, 1])
            vis    = float(keypoints_with_scores[idx, 2])

            keypoints[name] = {
                "id":         idx,
                "x_norm":     round(x_norm, 6),
                "y_norm":     round(y_norm, 6),
                "z_norm":     0.0,          # MoveNet is 2D — no depth
                "x_px":       int(x_norm * frame_w),
                "y_px":       int(y_norm * frame_h),
                "visibility": round(vis, 4),
            }

        return keypoints

    def close(self):
        self._module = None
        self._movenet = None
