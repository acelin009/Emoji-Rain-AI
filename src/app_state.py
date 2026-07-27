"""
Shared, frame-to-frame application state.

:class:`AppState` is a small mutable container that the main loop in
``app.py`` updates every frame and passes to the renderer/UI layer. Using
an explicit state object (rather than passing a dozen loose parameters
around) keeps the render/UI function signatures stable as the application
grows.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.expression_analyzer import Expression, FacialFeatures


class CameraStatus(str, Enum):
    """High-level camera connection status shown in the HUD."""

    INITIALIZING = "INITIALIZING"
    CONNECTED = "CONNECTED"
    DISCONNECTED = "DISCONNECTED"
    ERROR = "ERROR"


class DetectionStatus(str, Enum):
    """High-level face-detection status shown in the HUD."""

    SEARCHING = "SEARCHING"
    FACE_DETECTED = "FACE_DETECTED"
    FACE_LOST = "FACE_LOST"


@dataclass
class AppState:
    """Mutable, per-frame snapshot of everything the UI needs to render.

    This object is intentionally plain data: it holds no behavior of its
    own so that it stays trivially testable and serializable.
    """

    fps: float = 0.0
    frame_count: int = 0
    elapsed_seconds: float = 0.0

    camera_status: CameraStatus = CameraStatus.INITIALIZING
    detection_status: DetectionStatus = DetectionStatus.SEARCHING

    current_expression: Expression = Expression.NEUTRAL  # type: ignore
    expression_confidence: float = 0.0
    landmark_count: int = 0

    particle_count: int = 0
    total_particles_spawned: int = 0

    last_error: Optional[str] = None

    # Debug fields
    debug_features: Optional[FacialFeatures] = None
    debug_raw_candidate: Optional[Expression] = None  # type: ignore

    def record_frame(self, fps: float) -> None:
        """Updates per-frame bookkeeping fields."""
        self.fps = fps
        self.frame_count += 1

    def to_hud_dict(self) -> dict:
        """Returns a plain dict of the fields the HUD displays, useful for
        both rendering and testing."""
        return {
            "FPS": f"{self.fps:5.1f}",
            "Expression": self.current_expression.value,
            "Confidence": f"{self.expression_confidence * 100:5.1f}%",
            "Particles": str(self.particle_count),
            "Landmarks": str(self.landmark_count),
            "Detection": self.detection_status.value,
            "Camera": self.camera_status.value,
        }