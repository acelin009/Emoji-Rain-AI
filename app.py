"""
Emoji Rain AI - Main Application
Real-time emotion recognition with emoji rain effect.
"""

import cv2
import sys
import time
import random

from src.camera import Camera
from src.detector import FaceDetector
from src.emotion import EmotionRecognizer
from src.emoji_engine import EmojiEngine
from src.ui import (
    draw_emotion_bar,
    draw_emotion_label,
    draw_emoji_particles,
    draw_info_overlay,
    draw_bounding_box
)
from src.config import (
    WINDOW_NAME,
    QUIT_KEY,
    EMOTION_COLORS,
    SHOW_EMOTION_BAR,
    MIN_FACE_SIZE
)


def main():
    """Main application loop."""
    camera = None
    detector = None
    emotion_recognizer = None
    emoji_engine = None
    
    try:
        # Initialize modules
        print("=" * 50)
        print("🎬 Emoji Rain AI - Starting...")
        print("=" * 50)
        
        print("📷 Initializing camera...")
        camera = Camera()
        print("✅ Camera ready")
        
        print("👤 Initializing face detector...")
        detector = FaceDetector()
        print("✅ Face detector ready")
        
        print("🎭 Initializing emotion recognizer...")
        emotion_recognizer = EmotionRecognizer()
        print("✅ Emotion recognizer ready")
        
        print("🎨 Initializing emoji engine...")
        emoji_engine = EmojiEngine()
        print("✅ Emoji engine ready")
        
        print("\n" + "=" * 50)
        print("🎯 REAL-TIME EMOTION RECOGNITION + EMOJI RAIN")
        print("   Try different expressions to trigger emojis!")
        print("   Press 'Q' to quit")
        print("=" * 50 + "\n")
        
        # State variables
        current_emotion = 'Neutral'
        last_emotion_update = time.time()
        emotion_update_interval = 0.25  # Update emotion every 250ms
        
        while True:
            frame_start = time.time()
            
            # Capture frame
            success, frame, fps = camera.read()
            if not success:
                print("⚠️ Failed to capture frame")
                break
            
            # Detect faces
            faces = detector.detect(frame)
            frame = detector.draw(frame, faces)
            
            # Process first face
            if len(faces) > 0:
                x, y, w, h = faces[0]
                
                # Ensure crop is within bounds
                x = max(0, x)
                y = max(0, y)
                x2 = min(frame.shape[1], x + w)
                y2 = min(frame.shape[0], y + h)
                
                if x2 > x and y2 > y and w > MIN_FACE_SIZE and h > MIN_FACE_SIZE:
                    # Crop face
                    face_crop = frame[y:y2, x:x2]
                    
                    # Get emotion (with smoothing)
                    emotion, confidence, all_emotions = emotion_recognizer.get_smoothed_emotion(face_crop)
                    
                    if emotion:
                        current_emotion = emotion
                        emoji_engine.set_emotion(emotion)
                    
                    # Draw emotion label
                    if emotion and confidence:
                        frame = draw_emotion_label(frame, emotion, confidence, x, y - 10)
                    
                    # Draw bounding box with emotion color
                    frame = draw_bounding_box(frame, x, y, w, h, current_emotion)
                    
                    # Draw emotion bar chart
                    if SHOW_EMOTION_BAR and all_emotions:
                        bar_x = frame.shape[1] - 220
                        bar_y = frame.shape[0] - 200
                        frame = draw_emotion_bar(frame, all_emotions, bar_x, bar_y)
            
            # Update emoji engine
            dt = 1.0 / max(fps, 1)
            emoji_engine.update(dt, frame.shape[1], frame.shape[0])
            
            # Draw emoji particles
            frame = draw_emoji_particles(frame, emoji_engine.get_particles())
            
            # Draw info overlay
            face_count = len(faces)
            particle_count = emoji_engine.get_particle_count()
            frame = draw_info_overlay(frame, int(fps), face_count, current_emotion, particle_count)
            
            # Show frame
            cv2.imshow(WINDOW_NAME, frame)
            
            # Quit check
            if cv2.waitKey(1) & 0xFF == QUIT_KEY:
                print("\n🛑 Closing application...")
                break
    
    except Exception as error:
        print(f"\n❌ Error: {error}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        if camera is not None:
            camera.release()
            print("✅ Camera released")
        cv2.destroyAllWindows()
        print("✅ Application closed")


if __name__ == "__main__":
    main()