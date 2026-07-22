"""
Unit tests for camera module.
"""

import pytest
from src.camera import Camera


def test_camera_creation():
    """
    Test that Camera can be instantiated.
    Note: This requires a webcam and will be mocked in CI later.
    """
    try:
        camera = Camera()
        assert camera.cap is not None
        camera.release()
    except RuntimeError:
        pytest.skip("No camera available for testing")