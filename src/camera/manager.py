"""
Camera management with OpenCV
"""

import cv2
import numpy as np
import threading
import time
from typing import Optional, Tuple
from ..core.logger import setup_logger


class CameraManager:
    def __init__(self, camera_id=0, width=640, height=480, fps=30):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.target_fps = fps
        self.cap = None
        self.frame = None
        self.running = False
        self.thread = None
        self.logger = setup_logger('CameraManager')
        self._frame_lock = threading.Lock()
        self._last_frame_time = 0
        self._frame_interval = 1.0 / fps
    
    def start(self) -> bool:
        """Start camera capture in a separate thread"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
            
            # Set properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open camera {self.camera_id}")
                return False
            
            # Verify actual settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.logger.info(f"Camera opened: {actual_width}x{actual_height} @ {actual_fps:.2f} FPS")
            
            self.running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            return True
            
        except Exception as e:
            self.logger.error(f"Camera error: {e}")
            return False
    
    def _capture_loop(self):
        """Capture frames in a separate thread"""
        while self.running:
            ret, frame = self.cap.read()
            
            if ret:
                # Convert BGR to RGB for consistency
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                with self._frame_lock:
                    self.frame = frame_rgb
                    self._last_frame_time = time.time()
            else:
                self.logger.warning("Failed to capture frame")
                time.sleep(0.01)
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame (thread-safe)"""
        with self._frame_lock:
            return self.frame.copy() if self.frame is not None else None
    
    def get_frame_bgr(self) -> Optional[np.ndarray]:
        """Get the latest frame in BGR format for display"""
        frame = self.get_frame()
        if frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return None
    
    def stop(self):
        """Stop camera capture"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        self.logger.info("Camera stopped")
    
    def is_running(self):
        return self.running and self.cap is not None and self.cap.isOpened()