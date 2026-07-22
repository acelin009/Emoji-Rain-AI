"""
Main Application - Real Emotion Recognition with DeepFace
"""

import cv2
import sys
import numpy as np
from collections import deque

from src.camera import Camera
from src.detector import FaceDetector
from src.emoji_mapper import EmotionRecognizer
from src.config import WINDOW_NAME, QUIT_KEY, EMOTION_COLORS, SHOW_EMOTION_BAR


def main():
    """Main application loop with real emotion recognition."""
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
        
        print("🎭 Initializing emotion recognizer...")
        emotion_recognizer = EmotionRecognizer()
        print("✅ Emotion recognizer ready")
        
        print("\n🎯 REAL EMOTION RECOGNITION ACTIVE!")
        print("   Try different expressions: 😊😢😠😲")
        print("   Press 'Q' to quit\n")
        
        # For emotion smoothing
        emotion_history = deque(maxlen=10)
        current_emotion = 'Neutral'
        
        while True:
            success, frame, fps = camera.read()
            
            if not success:
                print("⚠️ Failed to capture frame")
                break
            
            # Detect faces
            faces = detector.detect(frame)
            
            # Draw faces
            frame = detector.draw(frame, faces)
            
            # Process first face for emotion
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                
                # Crop face
                face_crop = frame[y:y+h, x:x+w]
                
                # Get emotion
                emotion, confidence, all_emotions = emotion_recognizer.get_emotion_with_smoothing(face_crop)
                
                if emotion:
                    # Update history for smoothing
                    emotion_history.append(emotion)
                    if len(emotion_history) > 0:
                        from collections import Counter
                        current_emotion = Counter(emotion_history).most_common(1)[0][0]
                    
                    # Get color
                    color = EMOTION_COLORS.get(current_emotion, (0, 255, 255))
                    
                    # Draw emotion label
                    label = f"{current_emotion}: {confidence:.2f}" if confidence else current_emotion
                    cv2.putText(
                        frame,
                        label,
                        (x, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2
                    )
                    
                    # Draw emoji icon
                    emoji_map = {
                        'Happy': '😊',
                        'Sad': '😢',
                        'Angry': '😠',
                        'Surprise': '😲',
                        'Fear': '😨',
                        'Disgust': '🤢',
                        'Neutral': '😐'
                    }
                    icon = emoji_map.get(current_emotion, '😐')
                    cv2.putText(
                        frame,
                        icon,
                        (x + w + 10, y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        color,
                        2
                    )
                    
                    # Draw emotion bar chart
                    if SHOW_EMOTION_BAR and all_emotions:
                        # Convert to array
                        predictions = np.array([all_emotions.get(label, 0) for label in EMOTION_COLORS.keys()])
                        # Normalize
                        if predictions.sum() > 0:
                            predictions = predictions / predictions.sum()
                        
                        bar_x = frame.shape[1] - 220
                        bar_y = frame.shape[0] - 200
                        draw_emotion_bar(frame, predictions, bar_x, bar_y)
            
            # Display FPS and info
            cv2.putText(
                frame,
                f"FPS: {fps} | Emotion: {current_emotion}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
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
    """Draw a bar chart showing emotion probabilities."""
    if predictions is None:
        return frame
    
    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    
    bar_width = width - 20
    bar_height = height // len(emotions) - 5
    start_x = x + 10
    
    for i, (label, prob) in enumerate(zip(emotions, predictions)):
        bar_y = y + 10 + i * (bar_height + 5)
        bar_length = int(prob * bar_width)
        
        color = EMOTION_COLORS.get(label, (255, 255, 255))
        
        cv2.rectangle(
            frame,
            (start_x, bar_y),
            (start_x + bar_length, bar_y + bar_height),
            color,
            -1
        )
        
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