"""
Utility Functions for Visualization
"""

import cv2
import numpy as np

from src.config import EMOTION_COLORS, EMOTION_LABELS


def draw_emotion_bar(frame, predictions, x, y, width=200, height=150):
    """Draw a bar chart showing emotion probabilities."""
    if predictions is None:
        return frame
    
    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Bar settings
    bar_width = width - 20
    bar_height = height // len(EMOTION_LABELS) - 5
    start_x = x + 10
    
    for i, (label, prob) in enumerate(zip(EMOTION_LABELS, predictions)):
        bar_y = y + 10 + i * (bar_height + 5)
        bar_length = int(prob * bar_width)
        
        color = EMOTION_COLORS.get(label, (255, 255, 255))
        
        cv2.rectangle(
            frame,
            (start_x, bar_y),
            (start_x + bar_length, bar_y + bar_height),
            color,
            -1
        )
        
        label_text = f"{label}: {prob:.2f}"
        cv2.putText(
            frame,
            label_text,
            (start_x + 5, bar_y + bar_height - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1
        )
    
    return frame


def draw_emotion_label(frame, emotion, confidence, x, y):
    """Draw a large label showing the detected emotion."""
    if emotion is None:
        return frame
    
    color = EMOTION_COLORS.get(emotion, (255, 255, 255))
    label = f"{emotion}: {confidence:.2f}"
    
    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    )
    
    cv2.rectangle(
        frame,
        (x, y - text_height - 10),
        (x + text_width + 20, y + 10),
        (0, 0, 0),
        -1
    )
    
    cv2.putText(
        frame,
        label,
        (x + 10, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )
    
    return frame


def get_face_center(bbox, width, height):
    """Calculate center point of face bounding box."""
    x = int(bbox.xmin * width)
    y = int(bbox.ymin * height)
    w = int(bbox.width * width)
    h = int(bbox.height * height)
    
    center_x = x + w // 2
    center_y = y + h // 2
    
    return center_x, center_y, x, y, w, h