"""
Centralized, strongly-typed configuration for Emoji Rain AI.

Every tunable constant used across the application lives in this module so
that behavior can be adjusted from a single, well-documented location
instead of being scattered as "magic numbers" throughout the codebase.

All configuration is expressed as frozen dataclasses grouped by concern
(camera, expression thresholds, particle system, UI, logging). A single
top-level :class:`AppConfig` aggregates them and is what the rest of the
application imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

# --------------------------------------------------------------------------- #
# Project paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
ASSETS_DIR: Path = PROJECT_ROOT / "assets"
FONTS_DIR: Path = ASSETS_DIR / "fonts"
LOGS_DIR: Path = PROJECT_ROOT / "logs"

RGBColor = Tuple[int, int, int]
RGBAColor = Tuple[int, int, int, int]


# --------------------------------------------------------------------------- #
# Camera configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CameraConfig:
    """Settings controlling webcam capture."""

    device_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    requested_fps: int = 30
    flip_horizontal: bool = True  # Mirror mode, feels natural to the user.
    warmup_frames: int = 5
    reconnect_attempts: int = 3
    reconnect_delay_seconds: float = 1.0


# --------------------------------------------------------------------------- #
# Landmark detection configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LandmarkConfig:
    """Settings for the MediaPipe Face Mesh detector."""

    max_num_faces: int = 1
    refine_landmarks: bool = True
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.6
    static_image_mode: bool = False
    total_landmarks: int = 468


# --------------------------------------------------------------------------- #
# Expression analyzer configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ExpressionThresholds:
    """
    Geometric thresholds used by the rule-based expression engine.

    All ratios are computed from normalized facial geometry (distances
    normalized by inter-ocular distance or face height) so that they remain
    stable across different face sizes and camera distances.
    """

    # Eye Aspect Ratio (EAR). Lower value == more closed eye.
    ear_closed_threshold: float = 0.21
    ear_open_threshold: float = 0.24
    ear_wink_difference: float = 0.06

    # Mouth Aspect Ratio (MAR) / mouth opening.
    mouth_open_threshold: float = 0.16
    # REMOVED: mouth_wide_open_threshold

    # Smile intensity: normalized upward curvature of mouth corners.
    smile_threshold: float = 0.045
    big_smile_threshold: float = 0.09
    frown_threshold: float = -0.035

    # Eyebrows: normalized distance between eyebrow and eye.
    eyebrow_raise_threshold: float = 0.045
    # REMOVED: eyebrow_lower_threshold

    # NEW: Pucker/kiss detection
    mouth_pucker_width_threshold: float = 0.30  # mouth width below this counts as puckered
    mouth_pucker_max_open: float = 0.12        # mouth must stay mostly closed to count as a pout

    # Teeth / tongue visibility heuristics.
    teeth_visibility_mouth_open_min: float = 0.04  # was 0.05
    teeth_visibility_brightness_min: float = 85.0  # was 100.0
    teeth_visibility_bright_ratio_min: float = 0.05  # was 0.08
    tongue_visibility_mouth_open_min: float = 0.10
    tongue_pinkish_ratio_min: float = 0.15
    tongue_hue_min: int = 0
    tongue_hue_max: int = 12
    tongue_saturation_min: int = 60

    # Temporal smoothing.
    history_length: int = 8
    min_stable_frames: int = 3
    min_confidence_to_switch: float = 0.5
    landmark_ema_alpha: float = 0.5


# --------------------------------------------------------------------------- #
# Emoji mapping
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EmojiMappingConfig:
    """Maps each recognized :class:`~src.expression_analyzer.Expression`
    to the emoji glyph(s) and a signature accent color used for UI
    highlights."""

    mapping: Dict[str, str] = field(
        default_factory=lambda: {
            "NEUTRAL": "\U0001f610",  # 😐
            "SMILE_EYES_OPEN": "\U0001f60a",  # 😊
            "SMILE_EYES_CLOSED": "\U0001f60c",  # 😌
            "BIG_SMILE_EYES_OPEN": "\U0001f601",  # 😁
            "BIG_SMILE_EYES_CLOSED": "\U0001f606",  # 😆
            "WINK": "\U0001f609",  # 😉
            "TONGUE_WINK": "\U0001f61c",  # 😜
            "TONGUE_OUT_EYES_OPEN": "\U0001f61b",  # 😛
            "TONGUE_OUT_EYES_CLOSED": "\U0001f61d",  # 😝
            "SHOCKED": "\U0001f632",  # 😲
            "SAD": "\U0001f61e",  # 😞  (was 😢 - removed tear drop)
            "KISS_EYES_OPEN": "\U0001f617",  # 😗
            "KISS_EYES_CLOSED": "\U0001f61a",  # 😚
        }
    )

    accent_colors: Dict[str, RGBColor] = field(
        default_factory=lambda: {
            "NEUTRAL": (170, 170, 180),
            "SMILE_EYES_OPEN": (255, 209, 102),
            "SMILE_EYES_CLOSED": (255, 190, 80),
            "BIG_SMILE_EYES_OPEN": (255, 179, 71),
            "BIG_SMILE_EYES_CLOSED": (255, 160, 60),
            "WINK": (114, 214, 255),
            "TONGUE_WINK": (198, 130, 255),
            "TONGUE_OUT_EYES_OPEN": (168, 230, 163),
            "TONGUE_OUT_EYES_CLOSED": (140, 210, 135),
            "SHOCKED": (255, 107, 129),
            "SAD": (108, 156, 255),
            "KISS_EYES_OPEN": (255, 150, 200),
            "KISS_EYES_CLOSED": (255, 130, 190),
        }
    )


# --------------------------------------------------------------------------- #
# Particle system configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ParticleConfig:
    """Settings governing the emoji particle "rain" animation."""

    max_particles: int = 140
    spawn_rate_per_second: float = 14.0
    min_scale: float = 0.55
    max_scale: float = 1.35
    min_speed: float = 90.0  # pixels / second, upward
    max_speed: float = 190.0
    horizontal_drift_amplitude: float = 28.0
    horizontal_drift_frequency: float = 1.6
    min_rotation_speed: float = -60.0  # degrees / second
    max_rotation_speed: float = 60.0
    min_lifetime: float = 3.2  # seconds
    max_lifetime: float = 5.5
    fade_in_duration: float = 0.15
    fade_out_start_fraction: float = 0.75
    base_font_size: int = 42
    spawn_margin_px: int = 24


# --------------------------------------------------------------------------- #
# UI configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class UIConfig:
    """Settings for the glassmorphism heads-up display."""

    window_title: str = "Emoji Rain AI"
    panel_padding: int = 18
    panel_corner_radius: int = 22
    panel_background_rgba: RGBAColor = (18, 18, 28, 140)
    panel_border_rgba: RGBAColor = (255, 255, 255, 45)
    text_color_primary: RGBColor = (245, 245, 250)
    text_color_secondary: RGBColor = (170, 170, 185)
    accent_color: RGBColor = (114, 214, 255)
    danger_color: RGBColor = (255, 82, 82)
    success_color: RGBColor = (120, 230, 150)
    font_size_title: int = 22
    font_size_body: int = 17
    font_size_small: int = 14
    show_fps: bool = True
    show_landmark_overlay_hint: bool = True
    hud_top_left_margin: Tuple[int, int] = (24, 24)
    hud_width: int = 340  # was 300 - longer names need more room
    show_debug_features: bool = True
    debug_panel_width: int = 260


# --------------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LoggingConfig:
    """Settings for application-wide logging."""

    log_to_file: bool = True
    log_file_name: str = "emoji_rain_ai.log"
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    max_bytes: int = 2_000_000
    backup_count: int = 3


# --------------------------------------------------------------------------- #
# Aggregate application configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AppConfig:
    """Top-level configuration object aggregating every subsystem's
    configuration. Import :data:`DEFAULT_CONFIG` from this module unless a
    custom configuration is required (e.g. in tests)."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    landmarks: LandmarkConfig = field(default_factory=LandmarkConfig)
    expression: ExpressionThresholds = field(default_factory=ExpressionThresholds)
    emoji: EmojiMappingConfig = field(default_factory=EmojiMappingConfig)
    particles: ParticleConfig = field(default_factory=ParticleConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    target_fps: int = 60


DEFAULT_CONFIG = AppConfig()