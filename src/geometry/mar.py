"""
Mouth Aspect Ratio (MAR) calculation
"""

import numpy as np
from typing import Tuple, List, Optional


class MouthAspectRatio:
    """Calculate Mouth Aspect Ratio for mouth open detection"""
    
    # MediaPipe indices for mouth landmarks
    MOUTH_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]
    # Outer mouth corners: 61 (left), 291 (right)
    # Upper lip: 13, 14, 15, 16, 17
    # Lower lip: 84, 85, 86, 87, 88, 89, 90
    
    def __init__(self, landmark_extractor):
        self.landmark_extractor = landmark_extractor
        self.mar_history = []
        self.history_length = 10
    
    def calculate_mar(self, mouth_points: List[Tuple[float, float]]) -> float:
        """
        Calculate Mouth Aspect Ratio
        MAR = (vertical_distances) / horizontal_distance
        """
        if len(mouth_points) < 12:
            return 0.0
        
        # Use specific points for calculation
        # Upper lip center (13) and lower lip center (14)
        # Left corner (61) and right corner (291)
        mouth_points = np.array(mouth_points)
        
        # Get specific landmarks (indices in the mouth_points list)
        # Using: upper lip = index 10, lower lip = index 4
        # left corner = index 0, right corner = index 6
        upper_lip = mouth_points[10]
        lower_lip = mouth_points[4]
        left_corner = mouth_points[0]
        right_corner = mouth_points[6]
        
        vertical = np.linalg.norm(upper_lip - lower_lip)
        horizontal = np.linalg.norm(left_corner - right_corner)
        
        if horizontal == 0:
            return 0.0
        
        mar = vertical / horizontal
        return float(mar)
    
    def get_mar(self) -> float:
        """Get current Mouth Aspect Ratio"""
        if self.landmark_extractor.landmarks is None:
            return 0.0
        
        points = self.landmark_extractor.get_landmarks_by_indices(self.MOUTH_INDICES)
        mar = self.calculate_mar(points)
        
        # Update history
        self.mar_history.append(mar)
        if len(self.mar_history) > self.history_length:
            self.mar_history.pop(0)
        
        return mar
    
    def is_mouth_open(self, threshold=1.0) -> bool:
        """Check if mouth is open"""
        return self.get_mar() > threshold
    
    def get_jaw_open_ratio(self) -> float:
        """Get jaw opening ratio (0-1 range)"""
        mar = self.get_mar()
        # Normalize MAR to 0-1 range (assuming max MAR ~2.0)
        return min(mar / 2.0, 1.0)