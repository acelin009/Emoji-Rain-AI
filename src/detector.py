"""
Face Detection Module

Uses Google's MediaPipe Face Detection
to detect faces in real time.
"""

import cv2
import mediapipe as mp

from src.config import FACE_DETECTION_CONFIDENCE


class FaceDetector:
    """
    Face detection using MediaPipe.
    Handles detection and visualization of faces.
    """

    def __init__(self):
        """
        Initialize MediaPipe Face Detection.
        """
        self.mp_face = mp.solutions.face_detection
        
        self.detector = self.mp_face.FaceDetection(
            model_selection=0,  # 0 = short range (2m), 1 = long range (5m)
            min_detection_confidence=FACE_DETECTION_CONFIDENCE
        )

    def detect(self, frame):
        """
        Detect faces in the frame.
        
        Parameters:
        - frame: BGR image from OpenCV
        
        Returns:
        - results: MediaPipe FaceDetection results object
        """
        # Convert BGR to RGB (MediaPipe expects RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.detector.process(rgb)
        
        return results

    def draw(self, frame, results):
        """
        Draw bounding boxes and labels on detected faces.
        
        Parameters:
        - frame: Image to draw on
        - results: MediaPipe detection results
        
        Returns:
        - frame: Image with drawings
        """
        height, width, _ = frame.shape
        
        face_count = 0
        
        if results.detections:
            for detection in results.detections:
                face_count += 1
                
                # Get bounding box coordinates (relative)
                bbox = detection.location_data.relative_bounding_box
                
                # Convert to pixel coordinates
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)
                
                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),  # Green
                    2
                )
                
                # Get confidence score
                confidence = detection.score[0]
                
                # Draw confidence above box
                cv2.putText(
                    frame,
                    f"{confidence:.2f}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
                
                # Draw center point
                center_x = x + w // 2
                center_y = y + h // 2
                
                cv2.circle(
                    frame,
                    (center_x, center_y),
                    5,
                    (0, 0, 255),  # Red
                    -1
                )
        
        # Display face count
        cv2.putText(
            frame,
            f"Faces: {face_count}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),  # Yellow
            2
        )
        
        return frame