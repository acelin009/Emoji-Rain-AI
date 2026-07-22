"""
Project Configuration
"""

# Camera Settings
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Window Settings
WINDOW_NAME = "Emoji Rain AI"

# Keyboard
QUIT_KEY = ord("q")

# Face Detection
FACE_DETECTION_CONFIDENCE = 0.6
BOX_COLOR = (0, 255, 0)
BOX_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)
CENTER_COLOR = (0, 0, 255)

# Emotion Recognition
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Color mapping for each emotion (BGR format)
EMOTION_COLORS = {
    'Angry': (0, 0, 255),      # Red
    'Disgust': (0, 128, 128),   # Greenish-yellow
    'Fear': (255, 0, 255),      # Magenta
    'Happy': (0, 255, 255),     # Yellow
    'Neutral': (255, 255, 255), # White
    'Sad': (255, 0, 0),         # Blue
    'Surprise': (0, 255, 0)     # Green
}

# Model Settings
MODEL_PATH = "models/emotion_model.h5"
MIN_FACE_SIZE = 50  # Minimum pixels for face to be recognized
EMOTION_DISPLAY_DURATION = 30  # Frames to show emotion after detection

# Visualization
SHOW_EMOTION_BAR = True
SHOW_ALL_EMOTIONS = True