"""
Emoji Rain AI
=============

A real-time computer-vision application that detects facial expressions
using MediaPipe Face Mesh landmarks and renders an animated particle
"rain" of emojis that correspond to the currently detected expression.

This package contains the full application source code, organized using
clean architecture principles: each module has a single responsibility
and dependencies flow inward (from I/O -> analysis -> presentation).
"""

__title__ = "emoji-rain-ai"
__description__ = (
    "Real-time facial expression detection with an animated emoji "
    "particle rain, powered by MediaPipe Face Mesh."
)
__version__ = "1.0.0"
__author__ = "Emoji Rain AI Contributors"
__license__ = "MIT"

__all__ = [
    "__title__",
    "__description__",
    "__version__",
    "__author__",
    "__license__",
]
