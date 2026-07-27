"""
Rule-based facial expression analysis engine.

Unlike black-box emotion classifiers, this module derives an interpretable
feature vector directly from facial geometry (via :mod:`src.geometry`) and
applies an explicit, tunable decision tree to classify the expression. The
result is smoothed over time using a voting history so that transient,
noisy per-frame predictions do not cause the displayed expression to
flicker.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, Optional

import numpy as np

from src import geometry
from src.config import ExpressionThresholds
from src.logger import get_logger

logger = get_logger(__name__)


class Expression(str, Enum):
    """All expressions recognized by the engine."""

    NEUTRAL = "NEUTRAL"
    SMILE_EYES_OPEN = "SMILE_EYES_OPEN"
    SMILE_EYES_CLOSED = "SMILE_EYES_CLOSED"
    BIG_SMILE_EYES_OPEN = "BIG_SMILE_EYES_OPEN"
    BIG_SMILE_EYES_CLOSED = "BIG_SMILE_EYES_CLOSED"
    WINK = "WINK"
    TONGUE_WINK = "TONGUE_WINK"
    TONGUE_OUT_EYES_OPEN = "TONGUE_OUT_EYES_OPEN"
    TONGUE_OUT_EYES_CLOSED = "TONGUE_OUT_EYES_CLOSED"
    SHOCKED = "SHOCKED"
    SAD = "SAD"
    KISS_EYES_OPEN = "KISS_EYES_OPEN"
    KISS_EYES_CLOSED = "KISS_EYES_CLOSED"


@dataclass
class FacialFeatures:
    """A single frame's worth of derived facial geometry features."""

    ear_left: float
    ear_right: float
    ear_avg: float
    mar: float
    mouth_open: float
    mouth_width: float  # NEW: normalized mouth width for pout detection
    smile: float
    eyebrow: float
    teeth_visible: bool
    tongue_visible: bool
    head_yaw: Optional[float] = None
    head_pitch: Optional[float] = None
    head_roll: Optional[float] = None


@dataclass
class ExpressionResult:
    """Output of the analyzer for a single frame: the stabilized expression
    currently being displayed, its confidence, and the raw per-frame
    candidate/features for diagnostics."""

    expression: Expression
    confidence: float
    raw_candidate: Expression
    features: FacialFeatures


@dataclass
class _History:
    """Rolling window of recent per-frame expression candidates."""

    max_length: int
    items: Deque[Expression] = field(default_factory=deque)

    def push(self, expression: Expression) -> None:
        self.items.append(expression)
        while len(self.items) > self.max_length:
            self.items.popleft()

    def majority(self) -> Optional[Dict[str, float]]:
        if not self.items:
            return None
        counts: Dict[Expression, int] = {}
        for item in self.items:
            counts[item] = counts.get(item, 0) + 1
        winner = max(counts, key=counts.get)
        confidence = counts[winner] / len(self.items)
        return {"expression": winner, "confidence": confidence}


class ExpressionAnalyzer:
    """Classifies a facial expression from landmarks and smooths it over
    time.

    Example:
        analyzer = ExpressionAnalyzer(config.expression)
        result = analyzer.analyze(landmarks, frame_bgr, frame_w, frame_h)
    """

    def __init__(self, thresholds: ExpressionThresholds) -> None:
        self._thresholds = thresholds
        self._history = _History(max_length=thresholds.history_length)
        self._current_expression = Expression.NEUTRAL
        self._current_confidence = 1.0
        self._stable_frame_count = 0
        self._last_majority_expression = None
        self._smoothed_landmarks: Optional[np.ndarray] = None

    def analyze(
        self,
        landmarks: np.ndarray,
        frame_bgr: Optional[np.ndarray],
        frame_width: int,
        frame_height: int,
    ) -> ExpressionResult:
        """Analyzes one frame of landmarks and returns the temporally
        smoothed expression result.

        Args:
            landmarks: ``(468, 3)`` normalized landmark array from
                :class:`~src.landmark_detector.LandmarkDetector`.
            frame_bgr: The source BGR frame, used for color-based teeth/
                tongue heuristics. May be ``None`` to skip those heuristics.
            frame_width: Width of the source frame in pixels.
            frame_height: Height of the source frame in pixels.
        """
        smoothed = self._smooth_landmarks(landmarks)
        features = self._extract_features(smoothed, frame_bgr, frame_width, frame_height)
        candidate = self._classify(features)

        self._history.push(candidate)
        majority = self._history.majority()
        assert majority is not None

        if self._last_majority_expression is None:
            self._last_majority_expression = majority["expression"]
            self._stable_frame_count = 1
        elif majority["expression"] == self._last_majority_expression:
            self._stable_frame_count += 1
        else:
            self._stable_frame_count = 1
            self._last_majority_expression = majority["expression"]

        should_switch = (
            majority["expression"] != self._current_expression
            and self._stable_frame_count >= self._thresholds.min_stable_frames
            and majority["confidence"] >= self._thresholds.min_confidence_to_switch
        )
        if should_switch:
            logger.info(
                "Expression changed: %s -> %s (confidence=%.2f)",
                self._current_expression.value,
                majority["expression"].value,
                majority["confidence"],
            )
            self._current_expression = majority["expression"]
            self._current_confidence = majority["confidence"]
            self._stable_frame_count = 0
        else:
            self._current_confidence = majority["confidence"]

        return ExpressionResult(
            expression=self._current_expression,
            confidence=self._current_confidence,
            raw_candidate=candidate,
            features=features,
        )

    def _smooth_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Applies exponential smoothing to raw landmark coordinates to
        reduce per-frame jitter before feature extraction."""
        alpha = self._thresholds.landmark_ema_alpha
        if self._smoothed_landmarks is None or self._smoothed_landmarks.shape != landmarks.shape:
            self._smoothed_landmarks = landmarks.copy()
        else:
            self._smoothed_landmarks = self._smoothed_landmarks * (1.0 - alpha) + landmarks * alpha
        return self._smoothed_landmarks

    def _eyes_closed(self, features: FacialFeatures) -> bool:
        """Helper: returns True if both eyes are closed (EAR below threshold)."""
        return features.ear_avg < self._thresholds.ear_closed_threshold

    def _extract_features(
        self,
        landmarks: np.ndarray,
        frame_bgr: Optional[np.ndarray],
        frame_width: int,
        frame_height: int,
    ) -> FacialFeatures:
        t = self._thresholds
        
        ear_left = geometry.eye_aspect_ratio(landmarks, geometry.LEFT_EYE_EAR_POINTS)
        ear_right = geometry.eye_aspect_ratio(landmarks, geometry.RIGHT_EYE_EAR_POINTS)
        mar = geometry.mouth_aspect_ratio(landmarks)
        mouth_open = geometry.mouth_opening_normalized(landmarks)
        mouth_width = geometry.mouth_width_normalized(landmarks)  # NEW
        smile = geometry.smile_intensity(landmarks)
        eyebrow = geometry.eyebrow_height(landmarks)
        
        teeth_visible = geometry.teeth_visibility_estimate(
            frame_bgr, 
            landmarks, 
            frame_width, 
            frame_height,
            min_mouth_open=t.teeth_visibility_mouth_open_min,
            brightness_threshold=t.teeth_visibility_brightness_min,
            bright_pixel_ratio_threshold=t.teeth_visibility_bright_ratio_min,
        )
        tongue_visible = geometry.tongue_visibility_estimate(
            frame_bgr,
            landmarks,
            frame_width,
            frame_height,
            min_mouth_open=t.tongue_visibility_mouth_open_min,
            pinkish_ratio_threshold=t.tongue_pinkish_ratio_min,
            hue_min=t.tongue_hue_min,
            hue_max=t.tongue_hue_max,
            saturation_min=t.tongue_saturation_min,
        )
        head_pose = geometry.estimate_head_pose(landmarks, frame_width, frame_height)

        return FacialFeatures(
            ear_left=ear_left,
            ear_right=ear_right,
            ear_avg=(ear_left + ear_right) / 2.0,
            mar=mar,
            mouth_open=mouth_open,
            mouth_width=mouth_width,  # NEW
            smile=smile,
            eyebrow=eyebrow,
            teeth_visible=teeth_visible,
            tongue_visible=tongue_visible,
            head_yaw=head_pose.yaw if head_pose else None,
            head_pitch=head_pose.pitch if head_pose else None,
            head_roll=head_pose.roll if head_pose else None,
        )

    def _classify(self, features: FacialFeatures) -> Expression:
        """Applies an explicit, ordered decision tree over the geometric
        feature vector. Order matters: more specific / higher-priority
        expressions are checked first.
        """
        t = self._thresholds
        ear_diff = abs(features.ear_left - features.ear_right)
        one_eye_closed = (
            ear_diff >= t.ear_wink_difference
            and min(features.ear_left, features.ear_right) < t.ear_closed_threshold
            and max(features.ear_left, features.ear_right) >= t.ear_open_threshold * 0.85
        )

        # 1. Wink family (highest priority: very distinctive geometry).
        if one_eye_closed:
            if features.tongue_visible:
                return Expression.TONGUE_WINK
            return Expression.WINK

        # 2. Kiss/pout: mouth width shrunk, mouth stays mostly closed.
        if (
            features.mouth_width <= t.mouth_pucker_width_threshold
            and features.mouth_open <= t.mouth_pucker_max_open
        ):
            if self._eyes_closed(features):
                return Expression.KISS_EYES_CLOSED
            return Expression.KISS_EYES_OPEN

        # 3. Shocked: mouth open + eyebrows raised (both eyes wide open too).
        if (
            features.mouth_open >= t.mouth_open_threshold
            and features.eyebrow >= t.eyebrow_raise_threshold
            and features.ear_avg >= t.ear_open_threshold
        ):
            return Expression.SHOCKED

        # 4. Tongue out (no wink): mouth open + tongue detected.
        if features.tongue_visible and features.mouth_open >= t.mouth_open_threshold * 0.6:
            if self._eyes_closed(features):
                return Expression.TONGUE_OUT_EYES_CLOSED
            return Expression.TONGUE_OUT_EYES_OPEN

        # 5. Big smile: strong upward mouth curvature + visible teeth.
        if features.smile >= t.big_smile_threshold and features.teeth_visible:
            if self._eyes_closed(features):
                return Expression.BIG_SMILE_EYES_CLOSED
            return Expression.BIG_SMILE_EYES_OPEN

        # 6. Regular smile.
        if features.smile >= t.smile_threshold:
            if self._eyes_closed(features):
                return Expression.SMILE_EYES_CLOSED
            return Expression.SMILE_EYES_OPEN

        # 7. Sad: mouth corners clearly drooping below neutral.
        if features.smile <= t.frown_threshold:
            return Expression.SAD

        # 8. Default: neutral face.
        return Expression.NEUTRAL

    @property
    def current_expression(self) -> Expression:
        return self._current_expression

    def reset(self) -> None:
        """Resets all temporal state, e.g. after the face is lost and
        re-acquired."""
        self._history = _History(max_length=self._thresholds.history_length)
        self._current_expression = Expression.NEUTRAL
        self._current_confidence = 1.0
        self._stable_frame_count = 0
        self._last_majority_expression = None
        self._smoothed_landmarks = None