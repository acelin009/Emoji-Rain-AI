"""
Face-to-Emoji Expression Detector
Main application entry point
"""

import sys
import time
import pygame
import cv2
import numpy as np
from typing import Optional, Dict, Any

# Import all modules
from config import *
from src.core.logger import setup_logger
from src.core.fps_counter import FPSCounter
from src.camera.manager import CameraManager
from src.geometry.landmarks import LandmarkExtractor
from src.geometry.ear import EyeAspectRatio
from src.geometry.mar import MouthAspectRatio
from src.geometry.smile import SmileDetector
from src.geometry.head_pose import HeadPoseEstimator
from src.geometry.eyebrows import EyebrowTracker
from src.expression.classifier import ExpressionClassifier
from src.expression.smoother import ExpressionSmoother
from src.particles.system import ParticleSystem
from src.ui.hud import HUD


class FaceToEmojiApp:
    """Main application class"""
    
    def __init__(self):
        self.logger = setup_logger('FaceToEmojiApp')
        self.running = False
        self.clock = None
        self.screen = None
        self.screen_width = 1280
        self.screen_height = 720
        
        # Initialize components
        self.camera = None
        self.landmark_extractor = None
        self.ear_calculator = None
        self.mar_calculator = None
        self.smile_detector = None
        self.head_pose = None
        self.eyebrow_tracker = None
        self.classifier = None
        self.smoother = None
        self.particle_system = None
        self.hud = None
        self.fps_counter = None
        
        # State
        self.current_emoji = '😐'
        self.current_expression = 'neutral'
        self.confidence = 0.0
        self.features = {}
        self.paused = False
        
        # Initialize Pygame
        self._init_pygame()
        
        # Initialize all components
        self._init_components()
    
    def _init_pygame(self):
        """Initialize Pygame"""
        try:
            pygame.init()
            
            # Set up display
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height),
                pygame.HWSURFACE | pygame.DOUBLEBUF
            )
            pygame.display.set_caption("Face to Emoji - Expression Detector")
            
            # Set up clock
            self.clock = pygame.time.Clock()
            
            self.logger.info("Pygame initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Pygame: {e}")
            sys.exit(1)
    
    def _init_components(self):
        """Initialize all components"""
        try:
            # Camera
            self.camera = CameraManager(
                camera_id=CAMERA_ID,
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT,
                fps=CAMERA_FPS
            )
            
            # MediaPipe landmark extractor
            self.landmark_extractor = LandmarkExtractor(
                min_detection_confidence=MIN_FACE_DETECTION_CONFIDENCE,
                min_tracking_confidence=MIN_TRACKING_CONFIDENCE
            )
            
            # Geometry calculators
            self.ear_calculator = EyeAspectRatio(self.landmark_extractor)
            self.mar_calculator = MouthAspectRatio(self.landmark_extractor)
            self.smile_detector = SmileDetector(self.landmark_extractor)
            self.head_pose = HeadPoseEstimator(self.landmark_extractor)
            self.eyebrow_tracker = EyebrowTracker(self.landmark_extractor)
            
            # Expression classifier
            self.classifier = ExpressionClassifier(config)
            
            # Expression smoother
            self.smoother = ExpressionSmoother(
                alpha=SMOOTHING_ALPHA,
                history_length=10
            )
            
            # Particle system
            self.particle_system = ParticleSystem(max_particles=MAX_PARTICLES)
            
            # HUD
            self.hud = HUD(self.screen_width, self.screen_height, config)
            
            # FPS counter
            self.fps_counter = FPSCounter()
            
            self.logger.info("All components initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            sys.exit(1)
    
    def start(self):
        """Start the application"""
        if not self.camera.start():
            self.logger.error("Failed to start camera")
            return
        
        self.running = True
        self.logger.info("Application started")
        self._main_loop()
    
    def _main_loop(self):
        """Main application loop"""
        while self.running:
            # Handle events
            self._handle_events()
            
            if not self.paused:
                # Process frame
                self._process_frame()
            
            # Render
            self._render()
            
            # Cap frame rate
            self.clock.tick(60)
            
            # Update FPS
            self.fps_counter.tick()
    
    def _handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    self.logger.info(f"Paused: {self.paused}")
                elif event.key == pygame.K_r:
                    self.particle_system.clear()
                    self.smoother.reset()
                    self.logger.info("Reset particles and smoother")
    
    def _process_frame(self):
        """Process a single frame"""
        # Get frame from camera
        frame = self.camera.get_frame()
        if frame is None:
            return
        
        # Process with MediaPipe
        landmarks = self.landmark_extractor.process(frame)
        
        if landmarks is not None:
            # Extract features
            self._extract_features()
            
            # Classify expression
            emoji, expression, confidence = self.classifier.classify(self.features)
            
            # Apply smoothing
            self.current_emoji, self.current_expression, self.confidence = self.smoother.update(
                emoji, expression, confidence
            )
            
            # Spawn particles on expression change
            self._spawn_particles(emoji)
            
            # Update HUD
            self.hud.update(
                self.current_emoji,
                self.current_expression,
                self.confidence,
                self.fps_counter.fps,
                self.particle_system.get_particle_count(),
                self.features
            )
    
    def _extract_features(self):
        """Extract all geometric features"""
        try:
            # EAR
            ear_left, ear_right = self.ear_calculator.get_ear()
            
            # MAR
            mar = self.mar_calculator.get_mar()
            
            # Smile
            smile_score = self.smile_detector.get_smile_score()
            
            # Head pose
            pitch, yaw, roll = self.head_pose.get_head_pose()
            
            # Eyebrows
            brow_furrow = self.eyebrow_tracker.get_brow_furrow()
            brow_raise = self.eyebrow_tracker.get_brow_raise()
            
            # Additional features for sad mouth detection
            # Check if mouth corners are pulled down
            sad_mouth = self._detect_sad_mouth()
            
            # Tongue detection (simplified - based on mouth inner shape)
            tongue_out = self._detect_tongue()
            
            # Mouth width (for sad mouth)
            mouth_width = self._get_mouth_width()
            
            self.features = {
                'ear_left': ear_left,
                'ear_right': ear_right,
                'ear': (ear_left, ear_right),
                'mar': mar,
                'smile_score': smile_score,
                'head_tilt': roll,
                'brow_furrow': brow_furrow,
                'brow_raise': brow_raise,
                'sad_mouth': sad_mouth,
                'tongue_out': tongue_out,
                'mouth_width': mouth_width
            }
            
        except Exception as e:
            self.logger.error(f"Feature extraction error: {e}")
            self.features = {}
    
    def _detect_sad_mouth(self) -> bool:
        """Detect sad mouth shape (corners pulled down)"""
        if self.landmark_extractor.landmarks is None:
            return False
        
        # Get mouth corner positions
        left_corner = self.landmark_extractor.get_landmark_normalized(61)
        right_corner = self.landmark_extractor.get_landmark_normalized(291)
        upper_lip = self.landmark_extractor.get_landmark_normalized(13)
        
        # In normalized coords, y=0 is top, y=1 is bottom
        # Sad mouth: corners are lower than upper lip
        avg_corner_y = (left_corner[1] + right_corner[1]) / 2
        
        # If corners are significantly below upper lip, it's a sad mouth
        return avg_corner_y > upper_lip[1] + 0.05
    
    def _detect_tongue(self) -> bool:
        """Detect if tongue is out (simplified)"""
        if self.landmark_extractor.landmarks is None:
            return False
        
        # Simplified: check if mouth is open and inner mouth is visible
        mar = self.mar_calculator.get_mar()
        
        # Tongue out usually has mouth open with specific inner lip positions
        if mar > 0.8:
            # Check inner lip positions (if available)
            try:
                inner_lip_top = self.landmark_extractor.get_landmark_normalized(13)
                inner_lip_bottom = self.landmark_extractor.get_landmark_normalized(14)
                inner_distance = abs(inner_lip_top[1] - inner_lip_bottom[1])
                
                # If inner mouth is open significantly
                return inner_distance > 0.1
            except:
                pass
        
        return False
    
    def _get_mouth_width(self) -> float:
        """Get mouth width (normalized)"""
        if self.landmark_extractor.landmarks is None:
            return 0.0
        
        left_corner = self.landmark_extractor.get_landmark_normalized(61)
        right_corner = self.landmark_extractor.get_landmark_normalized(291)
        
        # Calculate width relative to face size
        width = abs(left_corner[0] - right_corner[0])
        return width
    
    def _spawn_particles(self, emoji: str):
        """Spawn particles when expression changes"""
        # Only spawn if confidence is high enough
        if self.confidence < CONFIDENCE_THRESHOLD:
            return
        
        # Get center of screen
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2
        
        # Spawn particles
        self.particle_system.spawn(
            emoji,
            center_x + np.random.randint(-100, 100),
            center_y + np.random.randint(-50, 50),
            count=PARTICLE_SPAWN_RATE
        )
    
    def _render(self):
        """Render everything"""
        # Clear screen
        self.screen.fill((20, 20, 30))
        
        # Get camera frame for background
        frame = self.camera.get_frame_bgr()
        if frame is not None:
            # Resize and blur for background
            frame_small = cv2.resize(frame, (self.screen_width, self.screen_height))
            frame_small = cv2.GaussianBlur(frame_small, (21, 21), 0)
            frame_small = cv2.resize(frame_small, (self.screen_width, self.screen_height))
            
            # Convert to pygame surface
            frame_surface = pygame.surfarray.make_surface(frame_small)
            self.screen.blit(frame_surface, (0, 0))
        
        # Update and render particles
        dt = self.clock.get_time() / 1000.0  # Convert to seconds
        self.particle_system.update(dt)
        self.particle_system.render(self.screen)
        
        # Render HUD
        self.hud.draw(self.screen)
        
        # Update display
        pygame.display.flip()
    
    def stop(self):
        """Stop the application"""
        self.running = False
        self.camera.stop()
        self.landmark_extractor.cleanup()
        pygame.quit()
        self.logger.info("Application stopped")


def main():
    """Main entry point"""
    app = FaceToEmojiApp()
    try:
        app.start()
    except KeyboardInterrupt:
        app.logger.info("Interrupted by user")
    except Exception as e:
        app.logger.error(f"Application error: {e}", exc_info=True)
    finally:
        app.stop()


if __name__ == "__main__":
    main()