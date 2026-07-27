"""
Webcam capture abstraction.

Wraps ``cv2.VideoCapture`` behind a small, testable interface with
sensible defaults, automatic reconnect attempts, and clean resource
management via the context-manager protocol.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np

from src.config import CameraConfig
from src.logger import get_logger

logger = get_logger(__name__)


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened or read from."""


class Camera:
    """Manages a single webcam device.

    Example:
        with Camera(config) as camera:
            while True:
                ok, frame = camera.read()
                if not ok:
                    break
                process(frame)
    """

    def __init__(self, config: CameraConfig, video_capture_factory=None) -> None:
        """
        Args:
            config: Camera configuration (device index, resolution, FPS).
            video_capture_factory: Optional factory callable used to create
                the underlying capture object. Defaults to
                ``cv2.VideoCapture``. Primarily exists to allow dependency
                injection in unit tests without requiring real hardware.
        """
        self._config = config
        self._capture = None
        self._is_open = False

        if video_capture_factory is None:
            import cv2

            self._video_capture_factory = cv2.VideoCapture
        else:
            self._video_capture_factory = video_capture_factory

    def open(self) -> None:
        """Opens the configured camera device, applying resolution/FPS
        settings and performing a short warm-up read sequence.

        Raises:
            CameraError: If the device cannot be opened after all
                reconnect attempts are exhausted.
        """
        last_error: Optional[Exception] = None
        for attempt in range(1, self._config.reconnect_attempts + 1):
            try:
                logger.info(
                    "Opening camera device %d (attempt %d/%d)...",
                    self._config.device_index,
                    attempt,
                    self._config.reconnect_attempts,
                )
                self._capture = self._video_capture_factory(self._config.device_index)
                if self._capture is None or not self._capture.isOpened():
                    raise CameraError(f"Unable to open camera device {self._config.device_index}.")

                self._apply_capture_properties()
                self._warm_up()
                self._is_open = True
                logger.info("Camera started successfully.")
                return
            except Exception as exc:  # noqa: BLE001 - broad by design, retried
                last_error = exc
                logger.warning("Camera open attempt %d failed: %s", attempt, exc)
                time.sleep(self._config.reconnect_delay_seconds)

        raise CameraError(
            f"Failed to open camera after {self._config.reconnect_attempts} attempts."
        ) from last_error

    def _apply_capture_properties(self) -> None:
        import cv2

        assert self._capture is not None
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.frame_width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.frame_height)
        self._capture.set(cv2.CAP_PROP_FPS, self._config.requested_fps)

    def _warm_up(self) -> None:
        """Some webcams return black/garbage frames for the first few reads
        while auto-exposure settles; discard a handful of warm-up frames."""
        assert self._capture is not None
        for _ in range(self._config.warmup_frames):
            self._capture.read()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Reads a single frame.

        Returns:
            A ``(success, frame)`` tuple. ``frame`` is ``None`` if
            ``success`` is ``False``. The frame is horizontally flipped if
            ``config.flip_horizontal`` is enabled (mirror mode).
        """
        if not self._is_open or self._capture is None:
            raise CameraError("Camera is not open. Call open() first.")

        success, frame = self._capture.read()
        if not success or frame is None:
            return False, None

        if self._config.flip_horizontal:
            import cv2

            frame = cv2.flip(frame, 1)
        return True, frame

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def resolution(self) -> Tuple[int, int]:
        return self._config.frame_width, self._config.frame_height

    def release(self) -> None:
        """Releases the underlying capture device. Safe to call multiple
        times or if the camera was never opened."""
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception as exc:  # pragma: no cover - defensive cleanup
                logger.warning("Error releasing camera: %s", exc)
        self._is_open = False
        logger.info("Camera released.")

    def __enter__(self) -> Camera:
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
