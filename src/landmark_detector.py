"""
Thin, single-responsibility wrapper around MediaPipe's Face Mesh solution.

This module is intentionally the *only* place in the codebase that imports
``mediapipe`` directly, so that the rest of the application depends on a
small, stable interface (:class:`LandmarkDetector`) rather than the
third-party library's API surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.config import LandmarkConfig
from src.logger import get_logger

logger = get_logger(__name__)

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - exercised only if mediapipe missing
    mp = None  # type: ignore[assignment]


@dataclass
class DetectionResult:
    """Result of running face-mesh detection on a single frame."""

    detected: bool
    landmarks: Optional[np.ndarray]  # shape (468, 3), normalized [0, 1] x/y, relative z
    landmark_count: int


class LandmarkDetector:
    """Detects 468 facial landmarks per frame using MediaPipe Face Mesh.

    Example:
        detector = LandmarkDetector(config)
        result = detector.process(bgr_frame)
        if result.detected:
            use(result.landmarks)
        detector.close()
    """

    def __init__(self, config: LandmarkConfig) -> None:
        if mp is None:
            raise ImportError(
                "mediapipe is not installed. Install it with "
                "'pip install mediapipe' (see requirements.txt)."
            )
        self._config = config
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=config.static_image_mode,
            max_num_faces=config.max_num_faces,
            refine_landmarks=config.refine_landmarks,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        logger.info(
            "LandmarkDetector initialized (max_num_faces=%d, refine_landmarks=%s)",
            config.max_num_faces,
            config.refine_landmarks,
        )

    def process(self, frame_bgr: np.ndarray) -> DetectionResult:
        """Runs face-mesh inference on a single BGR frame.

        Args:
            frame_bgr: An ``(H, W, 3)`` BGR image as produced by OpenCV.

        Returns:
            A :class:`DetectionResult`. If no face is detected,
            ``detected`` is ``False`` and ``landmarks`` is ``None``.
        """
        import cv2  # local import keeps module import order predictable

        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self._face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True

        if not results.multi_face_landmarks:
            return DetectionResult(detected=False, landmarks=None, landmark_count=0)

        face_landmarks = results.multi_face_landmarks[0]
        points = np.array(
            [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark],
            dtype=np.float64,
        )
        return DetectionResult(detected=True, landmarks=points, landmark_count=len(points))

    def close(self) -> None:
        """Releases MediaPipe's internal resources. Safe to call multiple
        times."""
        try:
            self._face_mesh.close()
            logger.info("LandmarkDetector closed.")
        except Exception as exc:  # pragma: no cover - defensive cleanup
            logger.warning("Error while closing LandmarkDetector: %s", exc)

    def __enter__(self) -> LandmarkDetector:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
