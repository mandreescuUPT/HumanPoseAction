"""
Base Pose Backend
=================
Abstract interface that all pose estimation backends must implement.

Every backend's `process()` returns a flat keypoints dict identical to the
existing MediaPipe output format:

    {"NOSE": {"id": 0, "x_norm": ..., "y_norm": ..., "z_norm": ...,
              "x_px": ..., "y_px": ..., "visibility": ...}, ...}

or None when no person is detected.
"""

from abc import ABC, abstractmethod


class BasePoseBackend(ABC):
    """Common interface for pose estimation backends."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Short identifier, e.g. 'mediapipe', 'yolo', 'movenet'."""

    @property
    @abstractmethod
    def keypoint_names(self) -> dict[int, str]:
        """Mapping of keypoint index → name."""

    @abstractmethod
    def process(self, frame_rgb, frame_w: int, frame_h: int) -> dict | None:
        """Run pose estimation on one RGB frame.

        Parameters
        ----------
        frame_rgb : np.ndarray
            Frame in RGB colour order (H, W, 3).
        frame_w, frame_h : int
            Pixel dimensions of the frame (used for coordinate conversion).

        Returns
        -------
        dict or None
            Flat keypoints dict  {name: {id, x_norm, y_norm, z_norm,
            x_px, y_px, visibility}}  for the single best person,
            or None if nobody detected.
        """

    def close(self):
        """Release resources. Override if the backend needs cleanup."""
        pass
