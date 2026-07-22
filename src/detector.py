"""
Face Detection using OpenCV Haar Cascade
Simple, fast, and no external dependencies!
"""

import cv2


class FaceDetector:
    """Simple face detection using OpenCV."""
    
    def __init__(self):
        # Load pre-trained face detector
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise RuntimeError("Could not load face cascade classifier")

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