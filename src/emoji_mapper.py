"""
Lightweight Emotion Recognition using FER
No TensorFlow required!
"""

import cv2
import numpy as np
from fer import FER
import time
from collections import Counter


class EmotionRecognizer:
    """
    Lightweight emotion recognition using FER.
    Uses a pre-trained model without TensorFlow.
    """
    
    def __init__(self):
        print("🎭 Initializing FER emotion recognizer...")
        print("   This will download a pre-trained model on first use (approx. 50MB)")
        print("   ⏳ Please wait...")
        
        try:
            # Initialize FER detector
            self.detector = FER(mtcnn=False)  # Use OpenCV face detector (lighter)
            print("✅ FER initialized successfully!")
        except Exception as e:
            print(f"⚠️ FER initialization error: {e}")
            raise
        
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.last_prediction_time = 0
        self.prediction_interval = 0.2  # Predict every 0.2 seconds
        self.last_emotion = 'Neutral'
        self.last_confidence = 0.0
        self.emotion_history = []
        
    def predict_emotion(self, face_image):
        """
        Predict emotion from face image.
        
        Parameters:
        - face_image: BGR face crop
        
        Returns:
        - emotion: String label
        - confidence: Float between 0-1
        - all_emotions: Dict of all emotions
        """
        if face_image is None or face_image.size == 0:
            return None, 0.0, None
        
        # Check prediction interval for smoothing
        current_time = time.time()
        if current_time - self.last_prediction_time < self.prediction_interval:
            return self.last_emotion, self.last_confidence, None
        
        try:
            # Convert BGR to RGB (FER expects RGB)
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Predict emotion
            result = self.detector.detect_emotions(rgb_image)
            
            if result and len(result) > 0:
                # Get emotions dict
                emotions = result[0]['emotions']
                
                # Find dominant emotion
                dominant_emotion = max(emotions, key=emotions.get)
                confidence = emotions[dominant_emotion]
                
                # Update last prediction
                self.last_emotion = dominant_emotion
                self.last_confidence = confidence
                self.last_prediction_time = current_time
                
                # Add to history
                self.emotion_history.append(dominant_emotion)
                if len(self.emotion_history) > 10:
                    self.emotion_history.pop(0)
                
                # Smooth by taking most common in history
                if len(self.emotion_history) > 3:
                    smoothed = Counter(self.emotion_history).most_common(1)[0][0]
                    if confidence < 0.6:  # Only override if confidence is low
                        return smoothed, confidence, emotions
                
                return dominant_emotion, confidence, emotions
            
        except Exception as e:
            # If error, return last known prediction
            pass
        
        return self.last_emotion, self.last_confidence, None
    
    def get_emotion_with_smoothing(self, face_image):
        """
        Get emotion with additional smoothing.
        """
        emotion, confidence, all_emotions = self.predict_emotion(face_image)
        
        # Additional smoothing: if we have enough history, use mode
        if len(self.emotion_history) >= 5:
            most_common = Counter(self.emotion_history).most_common(1)[0][0]
            # If current confidence is low, use historical mode
            if confidence is not None and confidence < 0.5:
                return most_common, confidence, all_emotions
        
        return emotion, confidence, all_emotions