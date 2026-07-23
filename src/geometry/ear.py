"""
Eye Aspect Ratio (EAR) calculation
"""

import numpy as np
from typing import Tuple, List, Optional


class EyeAspectRatio:
    """Calculate Eye Aspect Ratio for blink detection"""
    
    # MediaPipe indices for eye landmarks
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  # p1, p2, p3, p4, p5, p6
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    
    def __init__(self, landmark_extractor):
        self.landmark_extractor = landmark_extractor
        self.ear_history = {'left': [], 'right': []}
        self.history_length = 10
    
    def calculate_ear(self, eye_points: List[Tuple[float, float]]) -> float:
        """
        Calculate Eye Aspect Ratio
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        """
        if len(eye_points) != 6:
            return 0.0
        
        # Convert to numpy arrays
        eye_points = np.array(eye_points)
        
        # Compute distances
        p1, p2, p3, p4, p5, p6 = eye_points
        vertical1 = np.linalg.norm(p2 - p6)
        vertical2 = np.linalg.norm(p3 - p5)
        horizontal = np.linalg.norm(p1 - p4)
        
        if horizontal == 0:
            return 0.0
        
        ear = (vertical1 + vertical2) / (2.0 * horizontal)
        return float(ear)
    
    def get_left_ear(self) -> float:
        """Get EAR for left eye"""
        if self.landmark_extractor.landmarks is None:
            return 0.0
        
        points = self.landmark_extractor.get_landmarks_by_indices(self.LEFT_EYE_INDICES)
        ear = self.calculate_ear(points)
        
        # Update history
        self.ear_history['left'].append(ear)
        if len(self.ear_history['left']) > self.history_length:
            self.ear_history['left'].pop(0)
        
        return ear
    
    def get_right_ear(self) -> float:
        """Get EAR for right eye"""
        if self.landmark_extractor.landmarks is None:
            return 0.0
        
        points = self.landmark_extractor.get_landmarks_by_indices(self.RIGHT_EYE_INDICES)
        ear = self.calculate_ear(points)
        
        # Update history
        self.ear_history['right'].append(ear)
        if len(self.ear_history['right']) > self.history_length:
            self.ear_history['right'].pop(0)
        
        return ear
    
    def get_ear(self) -> Tuple[float, float]:
        """Get both left and right EAR"""
        return (self.get_left_ear(), self.get_right_ear())
    
    def get_average_ear(self) -> float:
        """Get average of both eyes EAR"""
        left, right = self.get_ear()
        return (left + right) / 2.0
    
    def is_blinking(self, threshold=0.15) -> bool:
        """Detect if person is blinking"""
        ear = self.get_average_ear()
        return ear < threshold
    
    def are_eyes_closed(self, threshold=0.15) -> bool:
        """Check if eyes are closed (both)"""
        left, right = self.get_ear()
        return left < threshold and right < threshold