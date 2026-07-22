"""
Emoji Rendering

cv2.putText() only understands the built-in Hershey fonts, which are
plain ASCII line-art — they cannot draw Unicode emoji at all. That is
why emoji were showing up as "?" / "????" in the app.

This module renders real color emoji glyphs with Pillow (using the
OS's built-in color-emoji font) and composites them onto an OpenCV
(BGR, numpy) frame with proper alpha blending. If no emoji font can be
found on the system, it falls back to drawing a plain colored circle
so the app still runs instead of crashing.
"""

import os
import platform

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.config import FONT_PATH


def _find_emoji_font():
    """Locate a color-emoji font on the current OS."""
    if FONT_PATH and os.path.exists(FONT_PATH):
        return FONT_PATH

    system = platform.system()
    if system == "Windows":
        candidates = [
            os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "seguiemj.ttf"),
        ]
    elif system == "Darwin":
        candidates = [
            "/System/Library/Fonts/Apple Color Emoji.ttc",
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
            "/usr/share/fonts/noto/NotoColorEmoji.ttf",
            "/usr/local/share/fonts/NotoColorEmoji.ttf",
        ]

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


EMOJI_FONT_PATH = _find_emoji_font()

_FONT_CACHE = {}
_GLYPH_CACHE = {}


def _get_font(size):
    if size not in _FONT_CACHE:
        if EMOJI_FONT_PATH is None:
            _FONT_CACHE[size] = None
        else:
            try:
                _FONT_CACHE[size] = ImageFont.truetype(EMOJI_FONT_PATH, size)
            except OSError:
                _FONT_CACHE[size] = None
    return _FONT_CACHE[size]


def _render_glyph(emoji, size):
    """Render one emoji to an RGBA numpy array, cached by (emoji, size)."""
    font = _get_font(size)
    if font is None:
        return None

    pad = size // 2
    canvas = size + pad * 2
    img = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        draw.text((pad, pad), emoji, font=font, embedded_color=True)
    except TypeError:
        # Older Pillow versions don't support embedded_color
        draw.text((pad, pad), emoji, font=font, fill=(255, 255, 255, 255))
    return np.array(img)


def get_glyph(emoji, size):
    key = (emoji, int(size))
    if key not in _GLYPH_CACHE:
        _GLYPH_CACHE[key] = _render_glyph(emoji, int(size))
    return _GLYPH_CACHE[key]


def emoji_font_available():
    return EMOJI_FONT_PATH is not None


def draw_emoji(frame, emoji, x, y, size=40, alpha=1.0):
    """
    Draw a single emoji glyph centered at (x, y) on a BGR OpenCV frame.
    Falls back to a colored circle if no emoji font is available on
    this system.
    """
    glyph = get_glyph(emoji, size)

    if glyph is None:
        cv2.circle(frame, (int(x), int(y)), max(4, size // 2), (0, 215, 255), -1)
        return frame

    h, w = glyph.shape[:2]
    top_left_x = int(x - w // 2)
    top_left_y = int(y - h // 2)

    frame_h, frame_w = frame.shape[:2]

    x1, y1 = max(0, top_left_x), max(0, top_left_y)
    x2, y2 = min(frame_w, top_left_x + w), min(frame_h, top_left_y + h)
    if x1 >= x2 or y1 >= y2:
        return frame  # fully off-screen, nothing to draw

    px1, py1 = x1 - top_left_x, y1 - top_left_y
    px2, py2 = px1 + (x2 - x1), py1 + (y2 - y1)

    roi = frame[y1:y2, x1:x2].astype(np.float32)
    patch_rgb = glyph[py1:py2, px1:px2, :3][:, :, ::-1].astype(np.float32)  # RGB -> BGR
    patch_a = glyph[py1:py2, px1:px2, 3:4].astype(np.float32) / 255.0

    if alpha < 1.0:
        patch_a = patch_a * alpha

    blended = roi * (1.0 - patch_a) + patch_rgb * patch_a
    frame[y1:y2, x1:x2] = blended.astype(np.uint8)

    return frame
