"""
Camera Module

Handles webcam initialization,
frame capture,
and cleanup.
"""

import cv2
import time

from src.config import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    WINDOW_NAME,
    QUIT_KEY,
)


class Camera:
    """
    Webcam handler with FPS tracking.
    """

    def __init__(self):
        """
        Initialize webcam with configured settings.
        """
        self.cap = cv2.VideoCapture(CAMERA_INDEX)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam. Check if camera is connected.")

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        # FPS tracking
        self.previous_time = time.time()
        self.fps = 0

    def read(self):
        """
        Capture one frame with FPS calculation.

        Returns
        -------
        success : bool
            True if frame captured successfully
        frame : ndarray
            The captured frame
        fps : int
            Current FPS
        """
        success, frame = self.cap.read()

        # Calculate FPS
        if success:
            current_time = time.time()
            self.fps = 1 / (current_time - self.previous_time)
            self.previous_time = current_time

        return success, frame, int(self.fps)

    def release(self):
        """
        Release webcam resources.
        """
        self.cap.release()
        cv2.destroyAllWindows() 