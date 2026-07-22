"""
Real Emotion Recognition using DeepFace
Pre-trained model - no training required!
"""

import cv2
import numpy as np
from deepface import DeepFace
import time


class EmotionRecognizer:
    """
    Real emotion recognition using DeepFace.
    Detects emotions from face images in real-time.
    """
    
    def __init__(self):
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.last_prediction_time = 0
        self.prediction_interval = 0.3  # Predict every 0.3 seconds (smoothing)
        self.last_emotion = 'Neutral'
        self.last_confidence = 0.0
        self.emotion_history = []  # For smoothing over multiple frames
        
        print("🎭 Initializing DeepFace emotion recognizer...")
        print("   This will download pre-trained models on first use (approx. 100MB)")
        print("   ⏳ Please wait...")
        
        # Test that DeepFace works
        try:
            # Quick test with dummy image
            test_img = np.zeros((48, 48, 3), dtype=np.uint8)
            DeepFace.analyze(img_path=test_img, actions=['emotion'], enforce_detection=False)
            print("✅ DeepFace initialized successfully!")
        except Exception as e:
            print(f"⚠️ DeepFace initialization warning: {e}")
            print("   Will retry on first face detection")

    def predict_emotion(self, face_image):
        """
        Predict emotion from face image using DeepFace.
        
        Parameters:
        - face_image: RGB or BGR face crop
        
        Returns:
        - emotion: String label
        - confidence: Float between 0-1
        - all_predictions: Dictionary of all emotions with scores
        """
        if face_image is None or face_image.size == 0:
            return None, 0.0, None
        
        # Check if we should make a new prediction (for smoothing)
        current_time = time.time()
        if current_time - self.last_prediction_time < self.prediction_interval:
            # Return last prediction for smoothing
            return self.last_emotion, self.last_confidence, None
        
        try:
            # DeepFace expects RGB but can handle BGR
            # Convert if needed
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                # DeepFace works with both BGR and RGB
                pass
            
            # Make prediction
            result = DeepFace.analyze(
                img_path=face_image,
                actions=['emotion'],
                enforce_detection=False,  # We already detected the face
                silent=True  # Reduce console output
            )
            
            # Parse results
            if result and len(result) > 0:
                emotions = result[0]['emotion']
                
                # Find dominant emotion
                dominant_emotion = max(emotions, key=emotions.get)
                confidence = emotions[dominant_emotion] / 100.0  # Convert from percentage
                
                # Update last prediction
                self.last_emotion = dominant_emotion
                self.last_confidence = confidence
                self.last_prediction_time = current_time
                
                # Add to history for additional smoothing
                self.emotion_history.append(dominant_emotion)
                if len(self.emotion_history) > 5:
                    self.emotion_history.pop(0)
                
                return dominant_emotion, confidence, emotions
        
        except Exception as e:
            print(f"⚠️ Emotion prediction error: {e}")
            # Return last valid prediction
            return self.last_emotion, self.last_confidence, None
        
        return self.last_emotion, self.last_confidence, None
    
    def get_smoothed_emotion(self, face_image):
        """
        Get emotion with additional smoothing.
        Returns the most frequent emotion from recent history.
        """
        emotion, confidence, all_emotions = self.predict_emotion(face_image)
        
        # Additional smoothing: if we have history, use majority vote
        if len(self.emotion_history) >= 3:
            from collections import Counter
            most_common = Counter(self.emotion_history).most_common(1)[0][0]
            
            # Only override if confidence is low
            if confidence < 0.6:
                return most_common, confidence, all_emotions
        
        return emotion, confidence, all_emotions