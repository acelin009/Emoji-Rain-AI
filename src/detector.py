"""
Face Detection using OpenCV Haar Cascade
Simple, fast, and no external dependencies!
"""

import cv2
import os


class FaceDetector:
    """Simple face detection using OpenCV."""
    
    def __init__(self):
        # Try multiple possible paths for the cascade file
        possible_paths = [
            # Standard OpenCV path
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            # Alternative path
            os.path.join(cv2.__path__[0], 'data', 'haarcascade_frontalface_default.xml'),
            # Current directory
            'haarcascade_frontalface_default.xml',
        ]
        
        self.face_cascade = None
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"📁 Found cascade at: {path}")
                self.face_cascade = cv2.CascadeClassifier(path)
                break
        
        if self.face_cascade is None or self.face_cascade.empty():
            # If still not found, download it
            print("⬇️ Downloading face cascade file...")
            self._download_cascade()
        
        if self.face_cascade is None or self.face_cascade.empty():
            raise RuntimeError("Could not load face cascade classifier")

    def _download_cascade(self):
        """Download the Haar cascade file if not found."""
        import urllib.request
        
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        local_path = "haarcascade_frontalface_default.xml"
        
        try:
            urllib.request.urlretrieve(url, local_path)
            print(f"✅ Cascade file downloaded: {local_path}")
            self.face_cascade = cv2.CascadeClassifier(local_path)
        except Exception as e:
            print(f"❌ Could not download cascade: {e}")
            raise

    def detect(self, frame):
        """
        Detect faces in frame.
        Returns list of face rectangles.
        """
        # Convert to grayscale (required for Haar cascade)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )
        
        return faces

    def draw(self, frame, faces):
        """
        Draw bounding boxes around faces.
        """
        for (x, y, w, h) in faces:
            # Draw green rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw center point
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # Add label
            cv2.putText(
                frame,
                "Face Detected",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )
        
        # Show face count
        cv2.putText(
            frame,
            f"Faces: {len(faces)}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )
        
        return frame