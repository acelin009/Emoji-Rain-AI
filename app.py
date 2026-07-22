"""
Main Application Entry Point

Live webcam feed with:
- Face detection
- Emotion recognition
- Real-time visualization
"""

import cv2
import sys
import time
import numpy as np

from src.camera import Camera
from src.detector import FaceDetector
from src.emotion_model import EmotionRecognizer
from src.utils import draw_emotion_bar, draw_emotion_label, get_face_center
from src.config import (
    WINDOW_NAME,
    QUIT_KEY,
    EMOTION_COLORS,
    MIN_FACE_SIZE,
    SHOW_EMOTION_BAR,
)


def main():
    """
    Main application loop with emotion recognition.
    """
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
        
        print("🎭 Loading emotion recognition model...")
        emotion_recognizer = EmotionRecognizer(load_pretrained=False)
        print("✅ Emotion recognizer initialized")
        print("⚠️ Note: Model needs to be trained or pre-trained weights loaded")
        print("For now, using random predictions for demonstration")
        print("\nPress 'Q' to quit")
        
        # For demonstration - generate random emotions
        import random
        
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
                        # Crop face
                        face = frame[y:y+h, x:x+w]
                        
                        # For demonstration: random emotion prediction
                        # In production: emotion_recognizer.predict_emotion(face)
                        random_index = random.randint(0, 6)
                        emotion = EMOTION_COLORS.get(
                            list(EMOTION_COLORS.keys())[random_index],
                            'Neutral'
                        )
                        confidence = random.uniform(0.5, 0.95)
                        predictions = np.random.dirichlet(np.ones(7))
                        
                        # Draw bounding box with emotion color
                        color = EMOTION_COLORS.get(emotion, (0, 255, 0))
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        
                        # Draw emotion label above face
                        label = f"{emotion}: {confidence:.2f}"
                        cv2.putText(
                            frame,
                            label,
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2
                        )
                        
                        # Draw center point
                        center_x = x + w // 2
                        center_y = y + h // 2
                        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                        
                        # Draw emotion bar chart (if enabled)
                        if SHOW_EMOTION_BAR:
                            bar_x = width - 220
                            bar_y = height - 200
                            draw_emotion_bar(frame, predictions, bar_x, bar_y)
            
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