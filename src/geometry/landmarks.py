"""
MediaPipe Face Landmark extraction (Simplified version)
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, List, Tuple
from ..core.logger import setup_logger


class LandmarkExtractor:
    """Extract and manage face landmarks using MediaPipe (simplified)"""
    
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.logger = setup_logger('LandmarkExtractor')
        self.landmarks = None
        self.image_shape = None
    
    def process(self, image: np.ndarray) -> Optional[List]:
        """Process image and extract landmarks"""
        if image is None or len(image.shape) != 3:
            return None
        
        self.image_shape = image.shape
        
        try:
            # Convert to RGB if needed (MediaPipe expects RGB)
            if image.shape[2] == 3:
                # Check if it's already RGB (MediaPipe uses RGB)
                # Most cameras give BGR, so we convert
                if image[0, 0, 0] > 255:
                    image = image.astype(np.uint8)
                
                # MediaPipe FaceMesh works with RGB
                # If it's BGR, convert
                if np.mean(image[0, 0]) > np.mean(image[0, 0, 2]):
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            results = self.face_mesh.process(image)
            
            if results.multi_face_landmarks:
                self.landmarks = results.multi_face_landmarks[0].landmark
                return self.landmarks
            
            return None
            
        except Exception as e:
            self.logger.error(f"Landmark extraction error: {e}")
            return None
    
    def get_landmark_coordinates(self, landmark_idx: int) -> Tuple[float, float]:
        """Get x, y coordinates of a landmark in image space"""
        if self.landmarks is None or landmark_idx >= len(self.landmarks):
            return (0, 0)
        
        h, w = self.image_shape[:2]
        landmark = self.landmarks[landmark_idx]
        return (landmark.x * w, landmark.y * h)
    
    def get_landmark_normalized(self, landmark_idx: int) -> Tuple[float, float]:
        """Get normalized (0-1) coordinates of a landmark"""
        if self.landmarks is None or landmark_idx >= len(self.landmarks):
            return (0, 0)
        
        landmark = self.landmarks[landmark_idx]
        return (landmark.x, landmark.y)
    
    def get_landmarks_by_indices(self, indices: List[int]) -> List[Tuple[float, float]]:
        """Get coordinates for multiple landmarks"""
        return [self.get_landmark_coordinates(idx) for idx in indices]
    
    def get_landmarks_normalized_by_indices(self, indices: List[int]) -> List[Tuple[float, float]]:
        """Get normalized coordinates for multiple landmarks"""
        return [self.get_landmark_normalized(idx) for idx in indices]
    
    def cleanup(self):
        """Clean up MediaPipe resources"""
        if self.face_mesh:
            self.face_mesh.close()