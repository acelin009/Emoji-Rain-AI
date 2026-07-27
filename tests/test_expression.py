"""Unit tests for src.geometry and src.expression_analyzer.

Rather than hand-crafting full 468-point synthetic face landmark arrays for
every expression (fragile and hard to maintain), classification-tree tests
exercise ``ExpressionAnalyzer._classify`` directly against explicit
:class:`~src.expression_analyzer.FacialFeatures` instances. Geometry
function tests use small, targeted synthetic landmark arrays covering only
the indices each function actually reads.
"""

from __future__ import annotations

import numpy as np
import pytest
from src import geometry
from src.config import ExpressionThresholds
from src.expression_analyzer import Expression, ExpressionAnalyzer, FacialFeatures


def _base_landmarks() -> np.ndarray:
    """A minimal 468-point array with a neutral, open-eyed, closed-mouth
    configuration. Only indices consumed by src.geometry are meaningfully
    positioned; all others default to the face center."""
    points = np.full((468, 3), (0.5, 0.5, 0.0), dtype=np.float64)

    def set_point(index: int, x: float, y: float) -> None:
        points[index] = (x, y, 0.0)

    # Face scale reference.
    set_point(geometry.FOREHEAD_TOP, 0.5, 0.20)
    set_point(geometry.CHIN_BOTTOM, 0.5, 0.70)  # face_scale == 0.5

    # Left eye (open): EAR ~= 0.4
    set_point(33, 0.35, 0.35)
    set_point(160, 0.37, 0.33)
    set_point(158, 0.43, 0.33)
    set_point(133, 0.45, 0.35)
    set_point(153, 0.43, 0.37)
    set_point(144, 0.37, 0.37)

    # Right eye (open): EAR ~= 0.4
    set_point(263, 0.55, 0.35)
    set_point(387, 0.57, 0.33)
    set_point(385, 0.63, 0.33)
    set_point(362, 0.65, 0.35)
    set_point(380, 0.63, 0.37)
    set_point(373, 0.57, 0.37)

    # Eyebrows: neutral gap above eyes.
    for idx in geometry.LEFT_EYEBROW_POINTS:
        set_point(idx, 0.39, 0.30)
    for idx in geometry.RIGHT_EYEBROW_POINTS:
        set_point(idx, 0.61, 0.30)
    set_point(geometry.LEFT_EYE_TOP, 0.39, 0.33)
    set_point(geometry.RIGHT_EYE_TOP, 0.61, 0.33)

    # Mouth: neutral, closed.
    set_point(geometry.MOUTH_LEFT_CORNER, 0.40, 0.56)
    set_point(geometry.MOUTH_RIGHT_CORNER, 0.60, 0.56)
    set_point(geometry.MOUTH_TOP_OUTER, 0.50, 0.55)
    set_point(geometry.MOUTH_BOTTOM_OUTER, 0.50, 0.57)
    set_point(geometry.MOUTH_TOP_INNER, 0.50, 0.555)
    set_point(geometry.MOUTH_BOTTOM_INNER, 0.50, 0.565)

    return points


class TestGeometryFunctions:
    def test_face_scale_is_positive(self):
        landmarks = _base_landmarks()
        assert geometry.face_scale(landmarks) == pytest.approx(0.5, abs=1e-6)

    def test_eye_aspect_ratio_open_eye_is_moderate(self):
        landmarks = _base_landmarks()
        ear = geometry.eye_aspect_ratio(landmarks, geometry.LEFT_EYE_EAR_POINTS)
        assert 0.25 < ear < 0.55

    def test_eye_aspect_ratio_closed_eye_is_low(self):
        landmarks = _base_landmarks()
        # Flatten the left eye vertically to simulate a closed/blinking eye.
        landmarks[160] = (0.37, 0.348, 0.0)
        landmarks[144] = (0.37, 0.352, 0.0)
        landmarks[158] = (0.43, 0.348, 0.0)
        landmarks[153] = (0.43, 0.352, 0.0)
        ear = geometry.eye_aspect_ratio(landmarks, geometry.LEFT_EYE_EAR_POINTS)
        assert ear < 0.15

    def test_mouth_aspect_ratio_closed_mouth_is_low(self):
        landmarks = _base_landmarks()
        mar = geometry.mouth_aspect_ratio(landmarks)
        assert mar < 0.1

    def test_smile_intensity_neutral_is_near_zero(self):
        landmarks = _base_landmarks()
        assert geometry.smile_intensity(landmarks) == pytest.approx(0.0, abs=1e-6)

    def test_smile_intensity_raised_corners_is_positive(self):
        landmarks = _base_landmarks()
        landmarks[geometry.MOUTH_LEFT_CORNER] = (0.40, 0.48, 0.0)
        landmarks[geometry.MOUTH_RIGHT_CORNER] = (0.60, 0.48, 0.0)
        assert geometry.smile_intensity(landmarks) > 0.05

    def test_smile_intensity_lowered_corners_is_negative(self):
        landmarks = _base_landmarks()
        landmarks[geometry.MOUTH_LEFT_CORNER] = (0.40, 0.62, 0.0)
        landmarks[geometry.MOUTH_RIGHT_CORNER] = (0.60, 0.62, 0.0)
        assert geometry.smile_intensity(landmarks) < -0.03

    def test_eyebrow_height_neutral_within_bounds(self):
        landmarks = _base_landmarks()
        value = geometry.eyebrow_height(landmarks)
        assert -0.05 < value < 0.08

    def test_teeth_and_tongue_estimates_are_false_without_frame(self):
        landmarks = _base_landmarks()
        assert geometry.teeth_visibility_estimate(None, landmarks, 640, 480) is False
        assert geometry.tongue_visibility_estimate(None, landmarks, 640, 480) is False


def _features(**overrides) -> FacialFeatures:
    """Builds a FacialFeatures instance with neutral defaults, overridden
    by any keyword arguments provided."""
    defaults = dict(
        ear_left=0.35,
        ear_right=0.35,
        ear_avg=0.35,
        mar=0.05,
        mouth_open=0.05,
        smile=0.0,
        eyebrow=0.02,
        teeth_visible=False,
        tongue_visible=False,
    )
    defaults.update(overrides)
    return FacialFeatures(**defaults)


class TestExpressionClassification:
    def setup_method(self):
        self.analyzer = ExpressionAnalyzer(ExpressionThresholds())

    def test_neutral_face_classifies_as_neutral(self):
        result = self.analyzer._classify(_features())
        assert result == Expression.NEUTRAL

    def test_smile_classifies_as_smile(self):
        result = self.analyzer._classify(_features(smile=0.16))
        assert result == Expression.SMILE

    def test_big_smile_requires_teeth_visible(self):
        without_teeth = self.analyzer._classify(_features(smile=0.30, teeth_visible=False))
        with_teeth = self.analyzer._classify(_features(smile=0.30, teeth_visible=True))
        assert without_teeth == Expression.SMILE
        assert with_teeth == Expression.BIG_SMILE

    def test_wink_detected_from_asymmetric_ear(self):
        result = self.analyzer._classify(_features(ear_left=0.10, ear_right=0.35, ear_avg=0.225))
        assert result == Expression.WINK

    def test_wink_with_tongue_visible_is_tongue_wink(self):
        result = self.analyzer._classify(
            _features(ear_left=0.10, ear_right=0.35, ear_avg=0.225, tongue_visible=True)
        )
        assert result == Expression.TONGUE_WINK

    def test_laugh_requires_open_mouth_and_narrowed_eyes(self):
        result = self.analyzer._classify(
            _features(mouth_open=0.60, ear_avg=0.20, ear_left=0.20, ear_right=0.20, smile=0.10)
        )
        assert result == Expression.LAUGH

    def test_shocked_requires_open_mouth_and_raised_eyebrows(self):
        result = self.analyzer._classify(
            _features(mouth_open=0.45, eyebrow=0.15, ear_avg=0.35, ear_left=0.35, ear_right=0.35)
        )
        assert result == Expression.SHOCKED

    def test_tongue_out_without_wink(self):
        result = self.analyzer._classify(
            _features(mouth_open=0.30, tongue_visible=True, ear_left=0.35, ear_right=0.35)
        )
        assert result == Expression.TONGUE_OUT

    def test_sad_from_negative_smile(self):
        result = self.analyzer._classify(_features(smile=-0.12))
        assert result == Expression.SAD

    def test_angry_from_lowered_eyebrows(self):
        result = self.analyzer._classify(_features(eyebrow=-0.08, smile=0.0))
        assert result == Expression.ANGRY

    def test_temporal_smoothing_prevents_single_frame_flicker(self):
        analyzer = ExpressionAnalyzer(ExpressionThresholds())
        landmarks = _base_landmarks()

        # Feed several consistent neutral frames; expression should settle
        # on NEUTRAL and confidence should be high.
        result = None
        for _ in range(8):
            result = analyzer.analyze(landmarks, None, 640, 480)
        assert result is not None
        assert result.expression == Expression.NEUTRAL
        assert result.confidence > 0.5
