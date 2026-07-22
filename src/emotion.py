"""
Simple Emotion Recognition using OpenCV
No deep learning required - uses facial geometry
"""

import cv2
import numpy as np
import time
from collections import Counter


class EmotionRecognizer:
    """
    Simple rule-based emotion recognition.
    Uses facial feature ratios from OpenCV.
    """
    
    def __init__(self):
        print("🎭 Initializing simple emotion recognizer...")
        
        # Load face landmarks detector (for eyes and mouth)
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_smile.xml'
        )
        
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.last_emotion = 'Neutral'
        self.last_confidence = 0.5
        self.last_update_time = 0
        self.emotion_history = []
        
        print("✅ Emotion recognizer ready (no deep learning)")

    def predict_emotion(self, face_image):
        """
        Predict emotion using facial features.
        Returns emotion, confidence, and all predictions.
        """
        if face_image is None or face_image.size == 0:
            return None, 0.0, None
        
        # Convert to grayscale
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 5)
        
        # Detect smile
        smiles = self.smile_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.8,
            minNeighbors=20,
            minSize=(25, 25)
        )
        
        # Calculate features
        eye_count = len(eyes)
        smile_count = len(smiles)
        
        # Calculate mouth region intensity (proxy for mouth openness)
        mouth_region = gray[int(h*0.6):int(h*0.85), int(w*0.2):int(w*0.8)]
        if mouth_region.size > 0:
            mouth_openness = np.mean(mouth_region) / 255.0
        else:
            mouth_openness = 0.5
        
        # Rule-based emotion detection
        emotion = 'Neutral'
        confidence = 0.5
        
        # Simple rules
        if smile_count >= 1 and eye_count >= 1:
            emotion = 'Happy'
            confidence = min(0.9, 0.6 + 0.3 * (smile_count / 3))
        elif mouth_openness > 0.7 and eye_count >= 2:
            emotion = 'Surprise'
            confidence = min(0.85, 0.5 + mouth_openness * 0.5)
        elif smile_count == 0 and eye_count >= 1 and mouth_openness < 0.3:
            emotion = 'Sad'
            confidence = 0.6
        elif eye_count < 1:
            emotion = 'Angry'
            confidence = 0.5
        else:
            emotion = 'Neutral'
            confidence = 0.7
        
        # Create predictions dict (for bar chart)
        all_emotions = {label: 0.0 for label in self.emotion_labels}
        all_emotions[emotion] = confidence
        # Add some noise for other emotions
        for label in self.emotion_labels:
            if label != emotion:
                all_emotions[label] = (1 - confidence) / 6
        
        # Update history
        self.emotion_history.append(emotion)
        if len(self.emotion_history) > 10:
            self.emotion_history.pop(0)
        
        # Smoothing
        if len(self.emotion_history) >= 3:
            smoothed = Counter(self.emotion_history).most_common(1)[0][0]
            if confidence < 0.6:
                return smoothed, confidence, all_emotions
        
        return emotion, confidence, all_emotions

    def get_smoothed_emotion(self, face_image):
        """Get emotion with time-based smoothing."""
        current_time = time.time()
        if current_time - self.last_update_time < 0.2:  # Update every 200ms
            return self.last_emotion, self.last_confidence, None
        
        emotion, confidence, all_emotions = self.predict_emotion(face_image)
        if emotion:
            self.last_emotion = emotion
            self.last_confidence = confidence
            self.last_update_time = current_time
        
        return self.last_emotion, self.last_confidence, all_emotions