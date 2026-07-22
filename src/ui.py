"""
UI Drawing Functions
All visualization in one place.
"""

import cv2

from src.config import EMOTION_COLORS, EMOTION_TO_EMOJI
from src.emoji_render import draw_emoji


def draw_emotion_bar(frame, predictions, x, y, width=200, height=150):
    """Draw emotion probability bar chart."""
    if predictions is None:
        return frame

    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Bars
    bar_width = width - 20
    bar_height = height // len(emotions) - 5
    start_x = x + 10

    for i, label in enumerate(emotions):
        prob = predictions.get(label, 0.0)
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
    """Draw emotion label + real emoji icon above face."""
    if emotion is None:
        return frame

    color = EMOTION_COLORS.get(emotion, (255, 255, 255))
    label = f"{emotion}: {confidence:.2f}"

    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    )

    # Background
    cv2.rectangle(
        frame,
        (x, y - text_height - 15),
        (x + text_width + 55, y + 5),
        (0, 0, 0),
        -1
    )

    # Text
    cv2.putText(
        frame,
        label,
        (x + 10, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    # Real emoji icon (rendered with PIL, not cv2.putText)
    emoji_icon = EMOTION_TO_EMOJI.get(emotion, ['😐'])[0]
    icon_center_x = x + text_width + 35
    icon_center_y = y - text_height // 2 - 3
    frame = draw_emoji(frame, emoji_icon, icon_center_x, icon_center_y, size=32)

    return frame


def draw_emoji_particles(frame, particles):
    """Render emoji particles on frame using real emoji glyphs."""
    for particle in particles:
        if not particle.alive:
            continue

        alpha = particle.get_alpha()
        if alpha <= 0:
            continue

        frame = draw_emoji(
            frame,
            particle.emoji,
            particle.x,
            particle.y,
            size=int(particle.size),
            alpha=alpha
        )

    return frame


def draw_info_overlay(frame, fps, face_count, emotion, particle_count):
    """Draw information overlay."""
    y_pos = 30

    # FPS
    cv2.putText(
        frame,
        f"FPS: {fps}",
        (20, y_pos),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )
    y_pos += 30

    # Face count
    cv2.putText(
        frame,
        f"Faces: {face_count}",
        (20, y_pos),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )
    y_pos += 30

    # Current emotion
    if emotion:
        color = EMOTION_COLORS.get(emotion, (255, 255, 255))
        cv2.putText(
            frame,
            f"Emotion: {emotion}",
            (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )
        y_pos += 30

    # Particle count
    cv2.putText(
        frame,
        f"Emojis: {particle_count}",
        (20, y_pos),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        2
    )

    # Instructions at bottom (plain text only - cv2.putText can't draw emoji)
    h, w = frame.shape[:2]
    cv2.putText(
        frame,
        "Press 'Q' to quit | Try different expressions!",
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )

    return frame


def draw_bounding_box(frame, x, y, w, h, emotion=None):
    """Draw face bounding box with emotion color."""
    if emotion and emotion in EMOTION_COLORS:
        color = EMOTION_COLORS[emotion]
    else:
        color = (0, 255, 0)

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    center_x = x + w // 2
    center_y = y + h // 2
    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

    return frame
