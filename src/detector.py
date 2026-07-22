"""
Face Detection using OpenCV Haar Cascade
Simple, fast, and no external dependencies!
"""

import cv2
import os

from src.config import MIN_FACE_SIZE


class FaceDetector:
    """Simple face detection using OpenCV."""

    def __init__(self):
        # Try multiple possible paths for the cascade file
        possible_paths = [
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            os.path.join(cv2.__path__[0], 'data', 'haarcascade_frontalface_default.xml'),
            'haarcascade_frontalface_default.xml',
        ]

        self.face_cascade = None

        for path in possible_paths:
            if os.path.exists(path):
                print(f"📁 Found cascade at: {path}")
                self.face_cascade = cv2.CascadeClassifier(path)
                break

        if self.face_cascade is None or self.face_cascade.empty():
            raise RuntimeError("Could not load face cascade classifier")

    def detect(self, frame):
        """
        Detect faces in frame.
        Returns list of face rectangles.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
        )

        return faces

    def draw(self, frame, faces):
        """
        Draw bounding boxes around detected faces.

        NOTE: face-count text is intentionally NOT drawn here anymore -
        it was being drawn a second time by ui.draw_info_overlay(),
        which caused the overlapping "Faces: 1 Faces: 1" text you saw
        on screen. The count is now only drawn once, in the overlay.
        """
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

        return frame
