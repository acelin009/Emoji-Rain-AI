"""
Temporal smoothing for expression stability
"""

import numpy as np
from collections import deque
from typing import Tuple, Optional


class ExpressionSmoother:
    """Apply temporal smoothing to expression classification"""
    
    def __init__(self, alpha=0.7, history_length=10):
        self.alpha = alpha  # Smoothing factor (0-1)
        self.history_length = history_length
        self.expression_history = deque(maxlen=history_length)
        self.emoji_history = deque(maxlen=history_length)
        self.smoothed_emoji = '😐'
        self.smoothed_expression = 'neutral'
        self.emoji_confidence = 0.0
    
    def update(self, emoji: str, expression: str, confidence: float) -> Tuple[str, str, float]:
        """
        Update smoothed expression
        Returns: (smoothed_emoji, smoothed_expression, smoothed_confidence)
        """
        # Add to history
        self.expression_history.append((emoji, expression, confidence))
        self.emoji_history.append(emoji)
        
        # If history is empty, return current
        if len(self.expression_history) == 0:
            self.smoothed_emoji = emoji
            self.smoothed_expression = expression
            self.emoji_confidence = confidence
            return (emoji, expression, confidence)
        
        # Get most common emoji in history
        from collections import Counter
        emoji_counts = Counter(self.emoji_history)
        most_common_emoji = emoji_counts.most_common(1)[0][0]
        
        # Calculate confidence based on frequency in history
        freq_confidence = emoji_counts[most_common_emoji] / len(self.emoji_history)
        
        # Apply exponential smoothing
        if self.smoothed_emoji != most_common_emoji:
            # Only change if new emoji has been consistent
            if freq_confidence > 0.6:  # 60% of recent frames show this emoji
                self.smoothed_emoji = most_common_emoji
                # Find the associated expression
                for e, expr, _ in self.expression_history:
                    if e == most_common_emoji:
                        self.smoothed_expression = expr
                        break
        
        # Calculate smoothed confidence
        self.emoji_confidence = self.emoji_confidence * (1 - self.alpha) + confidence * self.alpha
        
        return (self.smoothed_emoji, self.smoothed_expression, self.emoji_confidence)
    
    def reset(self):
        """Reset smoother state"""
        self.expression_history.clear()
        self.emoji_history.clear()
        self.smoothed_emoji = '😐'
        self.smoothed_expression = 'neutral'
        self.emoji_confidence = 0.0
    
    def get_dominant_expression(self) -> Tuple[str, str, float]:
        """
        Get the dominant expression from history
        Returns: (emoji, expression, confidence)
        """
        if len(self.expression_history) == 0:
            return ('😐', 'neutral', 0.0)
        
        from collections import Counter
        emoji_counts = Counter(self.emoji_history)
        most_common_emoji = emoji_counts.most_common(1)[0][0]
        
        # Find expression for this emoji
        for e, expr, _ in self.expression_history:
            if e == most_common_emoji:
                confidence = emoji_counts[most_common_emoji] / len(self.emoji_history)
                return (most_common_emoji, expr, confidence)
        
        return ('😐', 'neutral', 0.0)