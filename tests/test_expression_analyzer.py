"""Tests for expression analyzer logic."""

import pytest
import numpy as np
from src.expression_analyzer import ExpressionAnalyzer, Expression
from src.config import ExpressionThresholds


class TestExpressionAnalyzer:
    """Test expression analyzer functionality."""

    @pytest.fixture
    def thresholds(self):
        """Create test thresholds with low values for testing."""
        return ExpressionThresholds(
            ear_closed_threshold=0.21,
            ear_open_threshold=0.24,
            ear_wink_difference=0.06,
            mouth_open_threshold=0.16,
            mouth_wide_open_threshold=0.26,
            smile_threshold=0.045,
            big_smile_threshold=0.09,
            frown_threshold=-0.035,
            eyebrow_raise_threshold=0.045,
            eyebrow_lower_threshold=-0.03,
            history_length=8,
            min_stable_frames=3,
            min_confidence_to_switch=0.5,
        )

    @pytest.fixture
    def analyzer(self, thresholds):
        """Create analyzer instance."""
        return ExpressionAnalyzer(thresholds)

    @pytest.fixture
    def mock_landmarks(self):
        """Create mock 468-point landmarks."""
        return np.random.rand(468, 3).astype(np.float32)

    def test_expression_switches_after_stable_frames(self, analyzer, mock_landmarks):
        """Test that expression switches after enough stable frames.
        
        This is a regression test for the bug where stable_frame_count
        was compared against _current_expression instead of the previous
        majority vote, preventing expressions from ever switching.
        """
        # Mock features for SMILE
        class MockFeatures:
            ear_left = 0.25
            ear_right = 0.25
            ear_avg = 0.25
            mar = 0.15
            mouth_open = 0.12
            smile = 0.08  # Above smile_threshold (0.045)
            eyebrow = 0.02
            teeth_visible = True
            tongue_visible = False
            head_yaw = None
            head_pitch = None
            head_roll = None

        # Monkey-patch _extract_features to return SMILE features
        original_extract = analyzer._extract_features
        
        def mock_extract(*args, **kwargs):
            return MockFeatures()
        
        analyzer._extract_features = mock_extract

        # Simulate 20 frames of SMILE
        for i in range(20):
            result = analyzer.analyze(
                mock_landmarks,
                None,
                640,
                480
            )
            
            # After min_stable_frames (3), expression should switch to SMILE
            if i >= 3:
                assert result.expression == Expression.SMILE, \
                    f"Frame {i}: expression should be SMILE, got {result.expression}"
        
        # Restore original method
        analyzer._extract_features = original_extract