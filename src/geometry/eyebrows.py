"""
Eyebrow movement tracking
"""

import numpy as np
from typing import Tuple, List, Optional


class EyebrowTracker:
    """Track eyebrow positions for expression detection"""
    
    # MediaPipe indices for eyebrows
    LEFT_EYEBROW_OUTER = 46
    LEFT_EYEBROW_CENTER = 53
    LEFT_EYEBROW_INNER = 55
    RIGHT_EYEBROW_OUTER = 276
    RIGHT_EYEBROW_CENTER = 283
    RIGHT_EYEBROW_INNER = 285
    
    # Eye indices for reference
    LEFT_EYE_CENTER = 33
    RIGHT_EYE_CENTER = 263
    
    def __init__(self, landmark_extractor):
        self.landmark_extractor = landmark_extractor
        self.brow_history = {'left': [], 'right': []}
        self.history_length = 10
    
    def get_eyebrow_height(self, side='left') -> float:
        """
        Get eyebrow height relative to eye (0-1 range)
        Higher values = raised eyebrows
        """
        if self.landmark_extractor.landmarks is None:
            return 0.0
        
        if side == 'left':
            brow_idx = self.LEFT_EYEBROW_CENTER
            eye_idx = self.LEFT_EYE_CENTER
        else:
            brow_idx = self.RIGHT_EYEBROW_CENTER
            eye_idx = self.RIGHT_EYE_CENTER
        
        brow_y = self.landmark_extractor.get_landmark_normalized(brow_idx)[1]
        eye_y = self.landmark_extractor.get_landmark_normalized(eye_idx)[1]
        
        # Height is distance between brow and eye (normalized)
        # In normalized coords, lower y = higher position
        height = eye_y - brow_y
        
        # Normalize to 0-1 range (typical range is 0-0.3)
        height = max(0, min(height / 0.3, 1.0))
        
        return height
    
    def get_brow_furrow(self) -> float:
        """
        Get eyebrow furrow/brow lowering intensity (0-1)
        Higher values = angry/furrowed brows
        """
        if self.landmark_extractor.landmarks is None:
            return 0.0
        
        # Get inner eyebrow positions
        left_inner = self.landmark_extractor.get_landmark_normalized(self.LEFT_EYEBROW_INNER)
        right_inner = self.landmark_extractor.get_landmark_normalized(self.RIGHT_EYEBROW_INNER)
        
        # Get center eyebrow positions
        left_center = self.landmark_extractor.get_landmark_normalized(self.LEFT_EYEBROW_CENTER)
        right_center = self.landmark_extractor.get_landmark_normalized(self.RIGHT_EYEBROW_CENTER)
        
        # Furrow = inner brows pulled down and together
        # Calculate distance between inner brows (they get closer)
        inner_distance = abs(left_inner[0] - right_inner[0])
        
        # Calculate how much brows are lowered
        # Reference: eye centers
        left_eye = self.landmark_extractor.get_landmark_normalized(self.LEFT_EYE_CENTER)
        right_eye = self.landmark_extractor.get_landmark_normalized(self.RIGHT_EYE_CENTER)
        
        left_brow_lower = left_eye[1] - left_inner[1]
        right_brow_lower = right_eye[1] - right_inner[1]
        avg_brow_lower = (left_brow_lower + right_brow_lower) / 2
        
        # Normalize (brows should be above eyes, so positive value)
        # Furrow = brows lowered (smaller distance)
        brow_lower_score = max(0, min(1 - avg_brow_lower / 0.2, 1.0))
        
        # Combine: furrow = lowered brows + close together
        # Normalize inner distance (0.2-0.4 range typically)
        distance_score = 1 - min(inner_distance / 0.3, 1.0)
        
        furrow_score = (brow_lower_score * 0.7 + distance_score * 0.3)
        furrow_score = max(0, min(furrow_score, 1.0))
        
        return furrow_score
    
    def get_brow_raise(self) -> float:
        """Get eyebrow raise intensity (0-1)"""
        left_height = self.get_eyebrow_height('left')
        right_height = self.get_eyebrow_height('right')
        return (left_height + right_height) / 2
    
    def are_brows_raised(self, threshold=0.6) -> bool:
        """Check if eyebrows are raised (surprise)"""
        return self.get_brow_raise() > threshold