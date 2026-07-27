"""Unit tests for src.camera.Camera using a fake VideoCapture factory so no
real webcam hardware is required to run the test suite (e.g. in CI)."""

from __future__ import annotations

import numpy as np
import pytest
from src.camera import Camera, CameraError
from src.config import CameraConfig


class _FakeVideoCapture:
    """A minimal stand-in for cv2.VideoCapture used in tests."""

    def __init__(self, device_index: int, opened: bool = True, frames_available: int = 100):
        self.device_index = device_index
        self._opened = opened
        self._frames_available = frames_available
        self._properties = {}
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - matches cv2 API naming
        return self._opened

    def set(self, prop_id, value) -> bool:  # noqa: N802 - matches cv2 API naming
        self._properties[prop_id] = value
        return True

    def read(self):
        if self._frames_available <= 0:
            return False, None
        self._frames_available -= 1
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        return True, frame

    def release(self) -> None:
        self.released = True


def _make_config(**overrides) -> CameraConfig:
    base = dict(
        device_index=0,
        frame_width=640,
        frame_height=480,
        requested_fps=30,
        flip_horizontal=False,
        warmup_frames=2,
        reconnect_attempts=2,
        reconnect_delay_seconds=0.0,
    )
    base.update(overrides)
    return CameraConfig(**base)


def _make_factory(opened: bool = True, frames_available: int = 100):
    """Returns a video-capture factory function that always yields a fresh
    `_FakeVideoCapture` configured with the given behavior."""

    def factory(device_index: int) -> _FakeVideoCapture:
        return _FakeVideoCapture(device_index, opened=opened, frames_available=frames_available)

    return factory


def _make_factory_returning(fake: _FakeVideoCapture):
    """Returns a video-capture factory function that always yields the same
    pre-constructed fake instance (useful for asserting on its state after
    the camera releases it)."""

    def factory(device_index: int) -> _FakeVideoCapture:
        return fake

    return factory


def test_camera_opens_successfully():
    config = _make_config()
    factory = _make_factory(opened=True, frames_available=50)

    camera = Camera(config, video_capture_factory=factory)
    camera.open()

    assert camera.is_open is True
    camera.release()


def test_camera_read_returns_frame_after_open():
    config = _make_config()
    factory = _make_factory(opened=True, frames_available=50)

    camera = Camera(config, video_capture_factory=factory)
    camera.open()
    success, frame = camera.read()

    assert success is True
    assert frame is not None
    assert frame.shape == (480, 640, 3)
    camera.release()


def test_camera_read_before_open_raises():
    config = _make_config()
    factory = _make_factory(opened=True)
    camera = Camera(config, video_capture_factory=factory)

    with pytest.raises(CameraError):
        camera.read()


def test_camera_raises_after_exhausting_reconnect_attempts():
    config = _make_config(reconnect_attempts=2)
    factory = _make_factory(opened=False)
    camera = Camera(config, video_capture_factory=factory)

    with pytest.raises(CameraError):
        camera.open()


def test_camera_context_manager_releases_on_exit():
    config = _make_config()
    fake = _FakeVideoCapture(0, opened=True, frames_available=50)
    factory = _make_factory_returning(fake)

    with Camera(config, video_capture_factory=factory) as camera:
        assert camera.is_open is True

    assert fake.released is True


def test_camera_read_handles_dropped_frame():
    config = _make_config()
    factory = _make_factory(opened=True, frames_available=0)
    camera = Camera(config, video_capture_factory=factory)
    camera.open()

    success, frame = camera.read()

    assert success is False
    assert frame is None
    camera.release()
