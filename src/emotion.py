"""
Simple Emotion Recognition using OpenCV
No deep learning required - uses facial geometry
"""

import cv2
import numpy as np
import time
from collections import Counter

from src.config import EMOTION_UPDATE_INTERVAL, SMOOTHING_WINDOW


class EmotionRecognizer:
    """
    Simple rule-based emotion recognition.
    Uses facial feature ratios from OpenCV.

    NOTE: this is a lightweight heuristic (no neural network), so it
    will never be as accurate as a trained model - but it needs no
    extra downloads and runs in real time on any machine. If you want
    materially better accuracy later, swapping this class for a small
    CNN (e.g. via the `fer` or `deepface` package) is the natural next
    step, while keeping the same predict_emotion() interface.
    """

    def __init__(self):
        print("🎭 Initializing simple emotion recognizer...")

        self.eye_cascade = None
        self.smile_cascade = None

        try:
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
            if self.eye_cascade.empty():
                self.eye_cascade = None
        except Exception:
            self.eye_cascade = None

        try:
            self.smile_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_smile.xml'
            )
            if self.smile_cascade.empty():
                self.smile_cascade = None
        except Exception:
            self.smile_cascade = None

        if self.eye_cascade is None:
            print("   ⚠️ Eye cascade not found - using fallback detection")
        if self.smile_cascade is None:
            print("   ⚠️ Smile cascade not found - using fallback detection")

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

        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        eye_count = 0
        smile_count = 0

        if self.eye_cascade is not None:
            try:
                eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 5)
                eye_count = len(eyes)
            except Exception:
                eye_count = 0

        if self.smile_cascade is not None:
            try:
                smiles = self.smile_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.8,
                    minNeighbors=20,
                    minSize=(25, 25)
                )
                smile_count = len(smiles)
            except Exception:
                smile_count = 0

        try:
            mouth_region = gray[int(h * 0.6):int(h * 0.85), int(w * 0.2):int(w * 0.8)]
            if mouth_region.size > 0:
                mouth_openness = np.mean(mouth_region) / 255.0
            else:
                mouth_openness = 0.5
        except Exception:
            mouth_openness = 0.5

        brightness = np.mean(gray) / 255.0
        contrast = np.std(gray) / 255.0

        if self.eye_cascade is None and self.smile_cascade is None:
            if brightness > 0.6 and contrast > 0.2:
                emotion = 'Happy'
                confidence = 0.6 + 0.3 * brightness
            elif brightness < 0.3:
                emotion = 'Sad'
                confidence = 0.6
            elif contrast < 0.1:
                emotion = 'Neutral'
                confidence = 0.7
            else:
                emotion = 'Neutral'
                confidence = 0.5
        else:
            emotion = 'Neutral'
            confidence = 0.5

            if smile_count >= 1 and eye_count >= 1:
                emotion = 'Happy'
                confidence = min(0.9, 0.6 + 0.3 * (smile_count / 3))
            elif mouth_openness > 0.7 and eye_count >= 2:
                emotion = 'Surprise'
                confidence = min(0.85, 0.5 + mouth_openness * 0.5)
            elif smile_count == 0 and eye_count >= 1 and mouth_openness < 0.3:
                emotion = 'Sad'
                confidence = 0.6
            elif eye_count < 1 and smile_count == 0:
                emotion = 'Angry'
                confidence = 0.5
            else:
                emotion = 'Neutral'
                confidence = 0.7

        all_emotions = {label: 0.0 for label in self.emotion_labels}
        all_emotions[emotion] = confidence
        remaining = 1.0 - confidence
        if remaining > 0:
            for label in self.emotion_labels:
                if label != emotion:
                    all_emotions[label] = remaining / 6

        self.emotion_history.append(emotion)
        if len(self.emotion_history) > SMOOTHING_WINDOW:
            self.emotion_history.pop(0)

        if len(self.emotion_history) >= 3:
            smoothed = Counter(self.emotion_history).most_common(1)[0][0]
            if confidence < 0.6:
                return smoothed, confidence, all_emotions

        return emotion, confidence, all_emotions

    def get_smoothed_emotion(self, face_image):
        """Get emotion with time-based smoothing."""
        current_time = time.time()
        if current_time - self.last_update_time < EMOTION_UPDATE_INTERVAL:
            return self.last_emotion, self.last_confidence, None

        emotion, confidence, all_emotions = self.predict_emotion(face_image)
        if emotion:
            self.last_emotion = emotion
            self.last_confidence = confidence
            self.last_update_time = current_time

        return self.last_emotion, self.last_confidence, all_emotions
