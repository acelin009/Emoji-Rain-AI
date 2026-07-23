"""
Configuration settings for Face-to-Emoji application
"""

# Camera Settings
CAMERA_ID = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# MediaPipe Settings
MEDIAPIPE_MODEL = 'face_landmarker.task'  # Will be downloaded automatically
NUM_FACES = 1
MIN_FACE_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Expression Thresholds
SMILE_THRESHOLD = 0.4
BIG_SMILE_THRESHOLD = 0.7
EYE_CLOSED_THRESHOLD = 0.15
MOUTH_OPEN_THRESHOLD = 1.0
BROW_FURROW_THRESHOLD = 0.5
SAD_MOUTH_THRESHOLD = -0.3
SURPRISE_BROW_THRESHOLD = 0.3

# Particle Settings
PARTICLE_LIFETIME = 2.0  # seconds
MAX_PARTICLES = 50
PARTICLE_GRAVITY = 100
PARTICLE_SPAWN_RATE = 5  # particles per expression
PARTICLE_MIN_SCALE = 0.5
PARTICLE_MAX_SCALE = 1.5

# Smoothing Settings
SMOOTHING_ALPHA = 0.7  # Higher = smoother (0-1)
CONFIDENCE_THRESHOLD = 0.6

# UI Settings
UI_GLASS_ALPHA = 0.3
UI_BORDER_RADIUS = 15
UI_PADDING = 20
UI_COLORS = {
    'primary': (255, 255, 255, 255),
    'secondary': (200, 200, 255, 200),
    'accent': (100, 200, 255, 200),
    'background': (30, 30, 40, 180),
    'glass': (255, 255, 255, 80),
}

# FPS Display
SHOW_FPS = True
FPS_UPDATE_INTERVAL = 0.5  # seconds