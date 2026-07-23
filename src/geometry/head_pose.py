"""
Head pose estimation from landmarks
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from ..core.logger import setup_logger


class HeadPoseEstimator:
    """Estimate head pose (rotation) from face landmarks"""
    
    # 3D model points (6 points: nose, chin, left eye, right eye, left mouth, right mouth)
    # These are approximate 3D coordinates in mm from facial landmarks
    MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0),       # Nose tip (1)
        (0.0, -30.0, -10.0),   # Chin (152)
        (-30.0, 10.0, -10.0),  # Left eye corner (33)
        (30.0, 10.0, -10.0),   # Right eye corner (263)
        (-20.0, 30.0, -10.0),  # Left mouth corner (61)
        (20.0, 30.0, -10.0)    # Right mouth corner (291)
    ], dtype=np.float32)
    
    # Corresponding landmark indices
    LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]
    
    def __init__(self, landmark_extractor):
        self.landmark_extractor = landmark_extractor
        self.rvec = None
        self.tvec = None
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # Camera parameters (will be set from image)
        self._setup_camera()
    
    def _setup_camera(self):
        """Setup camera matrix (approximate)"""
        # These will be updated when we have image dimensions
        self.camera_matrix = np.array([
            [640, 0, 320],
            [0, 640, 240],
            [0, 0, 1]
        ], dtype=np.float32)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    
    def _update_camera_matrix(self, image_shape):
        """Update camera matrix based on image dimensions"""
        if image_shape is None:
            return
        
        h, w = image_shape[:2]
        focal_length = w  # Approximate
        self.camera_matrix = np.array([
            [focal_length, 0, w/2],
            [0, focal_length, h/2],
            [0, 0, 1]
        ], dtype=np.float32)
    
    def get_head_pose(self) -> Tuple[float, float, float]:
        """
        Get head pose angles (pitch, yaw, roll) in degrees
        Returns: (pitch, yaw, roll)
        """
        if self.landmark_extractor.landmarks is None:
            return (0, 0, 0)
        
        # Get image shape
        if hasattr(self.landmark_extractor, 'image_shape'):
            self._update_camera_matrix(self.landmark_extractor.image_shape)
        
        # Get 2D landmark coordinates
        points_2d = []
        for idx in self.LANDMARK_INDICES:
            x, y = self.landmark_extractor.get_landmark_coordinates(idx)
            points_2d.append((x, y))
        
        points_2d = np.array(points_2d, dtype=np.float32)
        
        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(
            self.MODEL_POINTS,
            points_2d,
            self.camera_matrix,
            self.dist_coeffs
        )
        
        if not success:
            return (0, 0, 0)
        
        self.rvec = rvec
        self.tvec = tvec
        
        # Convert rotation vector to Euler angles
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        euler_angles = self._rotation_matrix_to_euler(rotation_matrix)
        
        # Convert to degrees
        pitch = np.degrees(euler_angles[0])
        yaw = np.degrees(euler_angles[1])
        roll = np.degrees(euler_angles[2])
        
        return (pitch, yaw, roll)
    
    def _rotation_matrix_to_euler(self, R):
        """
        Convert rotation matrix to Euler angles
        Returns: (pitch, yaw, roll)
        """
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        
        return (x, y, z)
    
    def get_head_tilt(self) -> float:
        """Get head tilt (roll) in degrees"""
        _, _, roll = self.get_head_pose()
        return roll
    
    def is_head_tilted(self, threshold=15) -> bool:
        """Check if head is tilted"""
        return abs(self.get_head_tilt()) > threshold