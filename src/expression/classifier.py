"""
Expression classifier using rule-based approach
"""

from typing import Dict, Any, Optional
import numpy as np
from ..core.logger import setup_logger


class ExpressionClassifier:
    """Classify facial expressions based on geometric features"""
    
    EXPRESSIONS = {
        'neutral': '😐',
        'happy_small': '🙂',
        'happy_big': '😁',
        'happy_eyes_closed': '😆',
        'tongue_out': '😛',
        'wink_tongue': '😜',
        'mouth_open': '😮',
        'surprise': '😲',
        'angry': '😡',
        'sad': '😢',
        'kiss': '😘',
        'wink': '😉'
    }
    
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger('ExpressionClassifier')
        self.last_expression = 'neutral'
        self.expression_confidence = 0.0
    
    def classify(self, features: Dict[str, Any]) -> tuple:
        """
        Classify expression from features
        Returns: (emoji, expression_name, confidence)
        """
        if not features:
            return ('😐', 'neutral', 0.0)
        
        # Extract features
        ear_left = features.get('ear_left', 0.3)
        ear_right = features.get('ear_right', 0.3)
        mar = features.get('mar', 0.5)
        smile_score = features.get('smile_score', 0.0)
        brow_furrow = features.get('brow_furrow', 0.0)
        brow_raise = features.get('brow_raise', 0.0)
        head_tilt = features.get('head_tilt', 0.0)
        tongue_out = features.get('tongue_out', False)
        mouth_width = features.get('mouth_width', 0.0)
        
        # Calculate average EAR
        avg_ear = (ear_left + ear_right) / 2
        
        # Check for expression priorities (from most specific to general)
        
        # 1. Wink + tongue (😜)
        if tongue_out and (ear_left < self.config.EYE_CLOSED_THRESHOLD or 
                          ear_right < self.config.EYE_CLOSED_THRESHOLD):
            if smile_score > 0.3:
                return ('😜', 'wink_tongue', 0.85)
        
        # 2. Tongue out (😛)
        if tongue_out and smile_score > 0.3:
            return ('😛', 'tongue_out', 0.8)
        
        # 3. Eyes closed + big smile (😆)
        if (ear_left < self.config.EYE_CLOSED_THRESHOLD and 
            ear_right < self.config.EYE_CLOSED_THRESHOLD and 
            smile_score > self.config.BIG_SMILE_THRESHOLD):
            return ('😆', 'happy_eyes_closed', 0.9)
        
        # 4. Big smile with teeth (😁)
        if smile_score > self.config.BIG_SMILE_THRESHOLD and mar > 0.7:
            return ('😁', 'happy_big', 0.85)
        
        # 5. Small smile (🙂)
        if smile_score > self.config.SMILE_THRESHOLD:
            return ('🙂', 'happy_small', 0.75)
        
        # 6. Angry (😡)
        if brow_furrow > self.config.BROW_FURROW_THRESHOLD and smile_score < 0.3:
            return ('😡', 'angry', 0.8)
        
        # 7. Sad (😢)
        # Check for sad mouth shape (corners pulled down)
        if features.get('sad_mouth', False) or (mouth_width < -0.2 and smile_score < 0.2):
            return ('😢', 'sad', 0.7)
        
        # 8. Surprise (😲)
        if brow_raise > self.config.SURPRISE_BROW_THRESHOLD and mar > self.config.MOUTH_OPEN_THRESHOLD:
            return ('😲', 'surprise', 0.85)
        
        # 9. Mouth open (😮)
        if mar > self.config.MOUTH_OPEN_THRESHOLD:
            return ('😮', 'mouth_open', 0.7)
        
        # 10. Wink (😉)
        if (ear_left < self.config.EYE_CLOSED_THRESHOLD or 
            ear_right < self.config.EYE_CLOSED_THRESHOLD):
            if smile_score > 0.2:
                return ('😉', 'wink', 0.7)
        
        # Default: neutral
        return ('😐', 'neutral', 0.6)
    
    def get_expression_confidence(self, emoji: str, features: Dict[str, Any]) -> float:
        """Calculate confidence score for the classified expression"""
        # Base confidence on feature strength
        confidence = 0.5
        
        if emoji == '😁' or emoji == '😆':
            confidence = min(features.get('smile_score', 0) * 1.2, 0.95)
        elif emoji == '😮' or emoji == '😲':
            confidence = min(features.get('mar', 0) / 1.5, 0.95)
        elif emoji == '😡':
            confidence = min(features.get('brow_furrow', 0) * 1.3, 0.95)
        elif emoji == '😢':
            confidence = 0.7 if features.get('sad_mouth', False) else 0.5
        
        return max(0.4, min(confidence, 0.95))