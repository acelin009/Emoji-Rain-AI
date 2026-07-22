"""
Main Application - Using OpenCV Face Detection
"""

import cv2
import sys
import random
import numpy as np

from src.camera import Camera
from src.detector_cv import FaceDetector  # Using OpenCV detector
from src.config import WINDOW_NAME, QUIT_KEY, EMOTION_COLORS, SHOW_EMOTION_BAR


def main():
    """Main application loop with face detection."""
    camera = None
    detector = None
    
    try:
        print("📷 Starting webcam...")
        camera = Camera()
        print("✅ Webcam started")
        
        print("👤 Initializing face detector (OpenCV)...")
        detector = FaceDetector()
        print("✅ Face detector ready")
        
        print("\n🎭 Press 'Q' to quit")
        print("   Face detection running with random emotions for demo")
        
        while True:
            success, frame, fps = camera.read()
            
            if not success:
                print("⚠️ Failed to capture frame")
                break
            
            # Detect faces
            faces = detector.detect(frame)
            
            # Draw faces
            frame = detector.draw(frame, faces)
            
            # Process each face for emotion (random for demo)
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    # Random emotion for demo
                    random_index = random.randint(0, len(EMOTION_COLORS) - 1)
                    emotion = list(EMOTION_COLORS.keys())[random_index]
                    confidence = random.uniform(0.6, 0.95)
                    
                    # Get color for this emotion
                    color = EMOTION_COLORS.get(emotion, (0, 255, 255))
                    
                    # Draw emotion label above face
                    label = f"{emotion}: {confidence:.2f}"
                    cv2.putText(
                        frame,
                        label,
                        (x, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )
                    
                    # Draw emotion bar chart (if enabled)
                    if SHOW_EMOTION_BAR:
                        # Create random predictions
                        predictions = np.random.dirichlet(np.ones(7))
                        
                        # Draw bar chart on the right side
                        bar_x = frame.shape[1] - 220
                        bar_y = frame.shape[0] - 200
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


def draw_emotion_bar(frame, predictions, x, y, width=200, height=150):
    """
    Draw a bar chart showing emotion probabilities.
    """
    if predictions is None:
        return frame
    
    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Emotion labels (same as in config)
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    
    # Bar settings
    bar_width = width - 20
    bar_height = height // len(emotions) - 5
    start_x = x + 10
    
    for i, (label, prob) in enumerate(zip(emotions, predictions)):
        bar_y = y + 10 + i * (bar_height + 5)
        bar_length = int(prob * bar_width)
        
        # Get color for this emotion
        color = EMOTION_COLORS.get(label, (255, 255, 255))
        
        # Draw bar
        cv2.rectangle(
            frame,
            (start_x, bar_y),
            (start_x + bar_length, bar_y + bar_height),
            color,
            -1
        )
        
        # Draw label
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


if __name__ == "__main__":
    main()