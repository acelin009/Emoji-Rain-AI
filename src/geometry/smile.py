"""
Smile detection and intensity calculation
"""

import numpy as np
from typing import Tuple, List, Optional


class SmileDetector:
    """Detect smile intensity"""
    
    # MediaPipe indices for mouth corners and upper lip
    LEFT_MOUTH_CORNER = 61
    RIGHT_MOUTH_CORNER = 291
    UPPER_LIP_CENTER = 13
    LOWER_LIP_CENTER = 14
    NOSE_TIP = 1
    
    def __init__(self, landmark_extractor):
        self.landmark_extractor = landmark_extractor
        self.smile_history = []
        self.history_length = 10
    
    def get_smile_score(self) -> float:
        """
        Calculate smile intensity (0-1)
        Based on lip corner elevation relative to nose
        """
        if self.landmark_extractor.landmarks is None:
            return 0.0
        
        # Get normalized coordinates
        left_corner = self.landmark_extractor.get_landmark_normalized(self.LEFT_MOUTH_CORNER)
        right_corner = self.landmark_extractor.get_landmark_normalized(self.RIGHT_MOUTH_CORNER)
        upper_lip = self.landmark_extractor.get_landmark_normalized(self.UPPER_LIP_CENTER)
        nose = self.landmark_extractor.get_landmark_normalized(self.NOSE_TIP)
        
        # Calculate average lip corner height
        avg_corner_y = (left_corner[1] + right_corner[1]) / 2
        
        # Smile = corners elevated relative to upper lip
        # In normalized coordinates, y=0 is top, y=1 is bottom
        # Smile pulls corners up (lower y value)
        lip_center_y = upper_lip[1]
        nose_y = nose[1]
        
        # Smile intensity: how much corners are pulled up relative to lip center
        # Normalize to 0-1 range
        raw_smile = (lip_center_y - avg_corner_y) / (lip_center_y - nose_y + 0.001)
        smile_score = max(0, min(raw_smile, 1.0))
        
        # Update history
        self.smile_history.append(smile_score)
        if len(self.smile_history) > self.history_length:
            self.smile_history.pop(0)
        
        return smile_score
    
    def is_smiling(self, threshold=0.4) -> bool:
        """Check if person is smiling"""
        return self.get_smile_score() > threshold
    
    def is_big_smile(self, threshold=0.7) -> bool:
        """Check if person has a big smile"""
        return self.get_smile_score() > threshold
    
    def get_teeth_visible(self) -> float:
        """
        Estimate teeth visibility (0-1)
        Based on mouth openness and smile intensity
        """
        # This is a simplified estimation
        # For accurate teeth detection, we'd need more sophisticated analysis
        from .mar import MouthAspectRatio
        mar_calc = MouthAspectRatio(self.landmark_extractor)
        
        smile = self.get_smile_score()
        mar = mar_calc.get_mar()
        
        # Teeth visible when mouth is somewhat open and smiling
        teeth_score = min(smile * mar * 2, 1.0)
        return teeth_score