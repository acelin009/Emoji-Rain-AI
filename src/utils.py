"""
Utility Functions for Visualization

NOTE: these are older, unused duplicates of functions that now live in
ui.py (which also correctly renders real emoji via emoji_render.py).
Nothing in the app imports this module anymore - it's kept only so it
doesn't error out if something else references it. Prefer src/ui.py.
"""

import cv2

from src.config import EMOTION_COLORS, EMOTION_LABELS


def draw_emotion_bar(frame, predictions, x, y, width=200, height=150):
    """Draw a bar chart showing emotion probabilities."""
    if predictions is None:
        return frame

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

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


def get_face_center(bbox, width, height):
    """Calculate center point of face bounding box."""
    x = int(bbox.xmin * width)
    y = int(bbox.ymin * height)
    w = int(bbox.width * width)
    h = int(bbox.height * height)

    center_x = x + w // 2
    center_y = y + h // 2

    return center_x, center_y, x, y, w, h
