"""
Main Application Entry Point
"""

import cv2
import sys
import time
import random
import numpy as np

from src.camera import Camera
from src.detector import FaceDetector
from src.utils import draw_emotion_bar, draw_emotion_label, get_face_center
from src.config import (
    WINDOW_NAME,
    QUIT_KEY,
    EMOTION_COLORS,
    MIN_FACE_SIZE,
    SHOW_EMOTION_BAR,
)

# Try to import emotion model (optional)
try:
    from src.emotion_model import EmotionRecognizer
    EMOTION_AVAILABLE = True
    print("✅ Emotion recognition module loaded")
except ImportError as e:
    print(f"⚠️ Emotion recognition not available: {e}")
    print("   Running in face detection only mode")
    EMOTION_AVAILABLE = False


def main():
    """Main application loop."""
    camera = None
    detector = None
    emotion_recognizer = None
    
    try:
        print("📷 Starting webcam...")
        camera = Camera()
        print("✅ Webcam started")
        
        print("👤 Initializing face detector...")
        detector = FaceDetector()
        print("✅ Face detector ready")
        
        # Initialize emotion recognizer if available
        if EMOTION_AVAILABLE:
            try:
                print("🎭 Loading emotion recognition model...")
                emotion_recognizer = EmotionRecognizer(load_pretrained=False)
                print("✅ Emotion recognizer initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize emotion model: {e}")
                emotion_recognizer = None
        
        print("\nPress 'Q' to quit")
        
        while True:
            success, frame, fps = camera.read()
            
            if not success:
                print("⚠️ Failed to capture frame")
                break
            
            # Detect faces
            results = detector.detect(frame)
            
            # Process each face
            if results.detections:
                for detection in results.detections:
                    # Get face coordinates
                    bbox = detection.location_data.relative_bounding_box
                    height, width, _ = frame.shape
                    
                    # Convert to pixel coordinates
                    x = int(bbox.xmin * width)
                    y = int(bbox.ymin * height)
                    w = int(bbox.width * width)
                    h = int(bbox.height * height)
                    
                    # Make sure face is large enough
                    if w > MIN_FACE_SIZE and h > MIN_FACE_SIZE:
                        # Draw bounding box
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        # Try emotion recognition if available
                        if emotion_recognizer is not None:
                            try:
                                face = frame[y:y+h, x:x+w]
                                # For demonstration with random emotions (since model isn't trained)
                                random_index = random.randint(0, 6)
                                emotion = list(EMOTION_COLORS.keys())[random_index]
                                confidence = random.uniform(0.5, 0.95)
                                predictions = np.random.dirichlet(np.ones(7))
                                
                                # Draw emotion label
                                label = f"{emotion}: {confidence:.2f}"
                                color = EMOTION_COLORS.get(emotion, (0, 255, 0))
                                cv2.putText(
                                    frame,
                                    label,
                                    (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    color,
                                    2
                                )
                                
                                # Draw emotion bar
                                if SHOW_EMOTION_BAR:
                                    bar_x = width - 220
                                    bar_y = height - 200
                                    draw_emotion_bar(frame, predictions, bar_x, bar_y)
                            except Exception as e:
                                print(f"⚠️ Emotion prediction error: {e}")
            
            # Display FPS
            cv2.putText(
                frame,
                f"FPS: {fps}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Show frame
            cv2.imshow(WINDOW_NAME, frame)
            
            # Quit check
            if cv2.waitKey(1) & 0xFF == QUIT_KEY:
                print("🛑 Closing application...")
                break
    
    except Exception as error:
        print(f"❌ Error: {error}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        if camera is not None:
            camera.release()
            print("✅ Camera released")


if __name__ == "__main__":
    main()