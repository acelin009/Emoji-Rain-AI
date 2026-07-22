"""
Emoji Rendering Module
Renders real emoji glyphs using PIL for proper Unicode support.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

from src.config import FONT_PATH


def get_emoji_font(size):
    """
    Get a font that supports emoji rendering.
    
    Tries multiple approaches:
    1. User-specified font path
    2. System emoji fonts
    3. Fallback to default font
    """
    # Try user-specified font first
    if FONT_PATH and os.path.exists(FONT_PATH):
        try:
            return ImageFont.truetype(FONT_PATH, size)
        except:
            pass
    
    # Try common emoji fonts by platform
    emoji_fonts = [
        # Windows
        "C:/Windows/Fonts/SegoeUIEmoji.ttf",
        "C:/Windows/Fonts/seguiemj.ttf",
        # macOS
        "/System/Library/Fonts/Apple Color Emoji.ttc",
        # Linux
        "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
        "/usr/share/fonts/google-noto-emoji/NotoColorEmoji.ttf",
    ]
    
    for font_path in emoji_fonts:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except:
                continue
    
    # Fallback: use default font (won't show emoji but won't crash)
    try:
        return ImageFont.load_default()
    except:
        return None


def draw_emoji(frame, emoji_char, x, y, size=40, alpha=1.0):
    """
    Draw a single emoji on an OpenCV frame using PIL.
    
    Parameters:
    - frame: OpenCV BGR image
    - emoji_char: Single emoji character (e.g., '😊')
    - x, y: Center position for the emoji
    - size: Font size in pixels
    - alpha: Opacity (0.0 to 1.0)
    
    Returns:
    - frame: Updated OpenCV image
    """
    if alpha <= 0 or emoji_char is None:
        return frame
    
    # Convert OpenCV BGR to PIL RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    
    # Create drawing context
    draw = ImageDraw.Draw(pil_image)
    
    # Get font
    font = get_emoji_font(size)
    if font is None:
        # Fallback: just draw text with cv2
        cv2.putText(frame, emoji_char, (int(x - size/2), int(y + size/3)),
                   cv2.FONT_HERSHEY_SIMPLEX, size/30, (255, 255, 255), 2)
        return frame
    
    # Calculate text position (centered)
    try:
        # Get text bounding box
        bbox = draw.textbbox((0, 0), emoji_char, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Position text (centered at x, y)
        text_x = int(x - text_width / 2)
        text_y = int(y - text_height / 2)
    except:
        # Fallback if textbbox fails
        text_x = int(x - size/2)
        text_y = int(y - size/2)
    
    # Draw emoji with alpha
    if alpha < 1.0:
        # Create temporary image for alpha blending
        temp_img = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
        temp_draw = ImageDraw.Draw(temp_img)
        temp_draw.text((text_x, text_y), emoji_char, font=font, fill=(255, 255, 255, int(alpha * 255)))
        pil_image = Image.alpha_composite(pil_image.convert('RGBA'), temp_img).convert('RGB')
    else:
        draw.text((text_x, text_y), emoji_char, font=font, fill=(255, 255, 255))
    
    # Convert back to OpenCV BGR
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    return frame