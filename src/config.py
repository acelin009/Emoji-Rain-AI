"""
Project Configuration
All settings in one place.
"""

# ===== Camera Settings =====
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# ===== Window Settings =====
WINDOW_NAME = "Emoji Rain AI"
QUIT_KEY = ord("q")

# ===== Face Detection =====
FACE_DETECTION_CONFIDENCE = 0.6
MIN_FACE_SIZE = 50

# ===== Emotion Recognition =====
EMOTION_UPDATE_INTERVAL = 0.25  # Seconds between emotion predictions
SMOOTHING_WINDOW = 7  # Number of frames to smooth over

# Canonical ordering used everywhere a list (rather than a dict) of
# emotions is needed, e.g. bar charts.
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ===== Emotion Colors (BGR format) =====
EMOTION_COLORS = {
    'Angry': (0, 0, 255),      # Red
    'Disgust': (0, 128, 128),  # Greenish-yellow
    'Fear': (255, 0, 255),     # Magenta
    'Happy': (0, 255, 255),    # Yellow
    'Neutral': (255, 255, 255),# White
    'Sad': (255, 0, 0),        # Blue
    'Surprise': (0, 255, 0)    # Green
}

# ===== Emoji Mapping =====
EMOTION_TO_EMOJI = {
    'Happy': ['😊', '😂', '🥰', '❤️', '✨', '🎉', '😍', '🥳'],
    'Sad': ['😢', '😭', '💔', '🌧️', '😞', '😔'],
    'Angry': ['😠', '😡', '💢', '🔥', '👿', '🤬'],
    'Surprise': ['😲', '😮', '🤯', '🌟', '🎊', '💫'],
    'Fear': ['😨', '😱', '💀', '👻', '😰'],
    'Disgust': ['🤢', '🤮', '👎', '💩', '😖'],
    'Neutral': ['😐', '😑', '🤔', '🙂', '😶']
}

# ===== Emoji Engine =====
MAX_PARTICLES = 300
PARTICLE_SPEED = 3
PARTICLE_SIZE = 40
SPAWN_RATE = 5  # Emojis per second
EMOJI_LIFETIME = 3.0  # Seconds before fading

# ===== UI Settings =====
SHOW_EMOTION_BAR = True
SHOW_FPS = True
SHOW_FACE_COUNT = True

# ===== Font Settings =====
# Path to a color-emoji font (.ttf/.ttc). Leave as None to auto-detect
FONT_PATH = None