"""
Emotion Recognition Module

Loads pre-trained model and handles emotion prediction.
Uses a lightweight CNN architecture.
"""

import cv2
import numpy as np
import os

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from src.config import EMOTION_LABELS, MODEL_PATH


class EmotionRecognizer:
    """
    Handles emotion recognition from face images.
    """

    def __init__(self, load_pretrained=True):
        """
        Initialize emotion recognition model.
        
        Parameters:
        - load_pretrained: If True, loads saved model; if False, creates new model
        """
        self.emotion_labels = EMOTION_LABELS
        self.input_size = (48, 48)  # Standard size for emotion recognition
        self.model_path = MODEL_PATH
        
        if load_pretrained and os.path.exists(self.model_path):
            print(f"📦 Loading pre-trained model from {self.model_path}")
            self.model = load_model(self.model_path)
        else:
            print("🏗️ Creating new emotion model...")
            self.model = self._create_model()
            if load_pretrained:
                print("⚠️ No pre-trained model found. Training required.")
        
        # Pre-processing constants
        self.mean_pixel = 0.0
        self.std_pixel = 1.0

    def _create_model(self):
        """
        Create lightweight CNN for emotion recognition.
        
        Architecture:
        - 2 Convolutional layers with MaxPooling
        - Dense layers with Dropout for regularization
        - Softmax output for 7 emotions
        """
        model = Sequential([
            # Input layer
            Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
            MaxPooling2D((2, 2)),
            
            # Hidden layers
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            
            # Flatten and fully connected layers
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            
            # Output layer (7 emotions)
            Dense(7, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def preprocess_face(self, face_image):
        """
        Preprocess face image for model input.
        
        Steps:
        1. Convert to grayscale
        2. Resize to 48x48
        3. Normalize pixel values
        4. Add channel dimension
        
        Parameters:
        - face_image: RGB face crop from detector
        
        Returns:
        - Preprocessed image ready for model input
        """
        # Convert to grayscale
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Resize to model input size
        resized = cv2.resize(gray, self.input_size)
        
        # Normalize (0-255 -> 0-1)
        normalized = resized / 255.0
        
        # Add channel dimension (48, 48, 1)
        input_array = np.expand_dims(normalized, axis=-1)
        
        # Add batch dimension (1, 48, 48, 1)
        input_array = np.expand_dims(input_array, axis=0)
        
        return input_array

    def predict_emotion(self, face_image):
        """
        Predict emotion from face image.
        
        Parameters:
        - face_image: RGB face crop
        
        Returns:
        - emotion: String label of dominant emotion
        - confidence: Float between 0-1
        - predictions: Array of 7 probabilities
        """
        if face_image is None or face_image.size == 0:
            return None, 0.0, None
        
        # Preprocess
        input_data = self.preprocess_face(face_image)
        
        # Predict
        predictions = self.model.predict(input_data, verbose=0)
        
        # Get dominant emotion
        emotion_index = np.argmax(predictions[0])
        confidence = predictions[0][emotion_index]
        emotion = self.emotion_labels[emotion_index]
        
        return emotion, confidence, predictions[0]

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=20):
        """
        Train the emotion model.
        
        This is for demonstration - you'll need actual training data.
        In production, you'd load pre-trained weights from FER2013 dataset.
        """
        print("Training emotion recognition model...")
        print("⚠️ This requires the FER2013 dataset or similar.")
        
        # Your training code here
        # For now, we'll just print a message
        print("Please use pre-trained weights instead of training from scratch.")

    def save_model(self):
        """Save trained model to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)
        print(f"✅ Model saved to {self.model_path}")