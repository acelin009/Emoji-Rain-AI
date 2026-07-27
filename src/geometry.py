"""
Pure geometric feature-extraction functions operating on MediaPipe Face
Mesh landmarks.

Every function in this module is stateless and side-effect free: given a
``(468, 3)`` array of normalized landmark coordinates (and, where needed, a
raw BGR frame for color-based heuristics), it returns a single scalar or
tuple describing one aspect of facial geometry. The :class:`~src.
expression_analyzer.ExpressionAnalyzer` composes these primitives into a
full facial-feature vector, which keeps the geometry math independently
testable and free of any classification or smoothing logic.

MediaPipe Face Mesh landmark indices used here follow the canonical 468
point topology (https://github.com/google/mediapipe).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - exercised only if cv2 is missing
    cv2 = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# Landmark index groups (MediaPipe canonical 468-point topology)
# --------------------------------------------------------------------------- #

LEFT_EYE_EAR_POINTS: Tuple[int, int, int, int, int, int] = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_EAR_POINTS: Tuple[int, int, int, int, int, int] = (263, 387, 385, 362, 380, 373)

LEFT_EYEBROW_POINTS: Tuple[int, ...] = (70, 63, 105, 66, 107)
RIGHT_EYEBROW_POINTS: Tuple[int, ...] = (300, 293, 334, 296, 336)
LEFT_EYE_TOP: int = 159
RIGHT_EYE_TOP: int = 386

MOUTH_TOP_OUTER: int = 0
MOUTH_BOTTOM_OUTER: int = 17
MOUTH_TOP_INNER: int = 13
MOUTH_BOTTOM_INNER: int = 14

MOUTH_LEFT_CORNER: int = 61
MOUTH_RIGHT_CORNER: int = 291

FOREHEAD_TOP: int = 10
CHIN_BOTTOM: int = 152
NOSE_TIP: int = 1
LEFT_FACE_EDGE: int = 234
RIGHT_FACE_EDGE: int = 454

# Points used for solvePnP-based head pose estimation.
HEAD_POSE_LANDMARKS: Tuple[int, int, int, int, int, int] = (
    NOSE_TIP,
    CHIN_BOTTOM,
    33,
    263,
    MOUTH_LEFT_CORNER,
    MOUTH_RIGHT_CORNER,
)

# Rough 3D model points (in an arbitrary, consistent unit) corresponding to
# HEAD_POSE_LANDMARKS, used as the object points for solvePnP.
_HEAD_POSE_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye corner
        (225.0, 170.0, -135.0),  # Right eye corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        (150.0, -150.0, -125.0),  # Right mouth corner
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class HeadPose:
    """Estimated head orientation in degrees."""

    yaw: float
    pitch: float
    roll: float


def _point(landmarks: np.ndarray, index: int) -> Tuple[float, float]:
    return float(landmarks[index][0]), float(landmarks[index][1])


def _distance(landmarks: np.ndarray, a: int, b: int) -> float:
    pa, pb = landmarks[a], landmarks[b]
    return float(np.linalg.norm(pa[:2] - pb[:2]))


def face_scale(landmarks: np.ndarray) -> float:
    """Returns a normalization scale (approximate face height in pixels/
    normalized units) used to make other measurements resolution- and
    distance-invariant. Guaranteed to be strictly positive."""
    height = _distance(landmarks, FOREHEAD_TOP, CHIN_BOTTOM)
    return max(height, 1e-6)


def eye_aspect_ratio(landmarks: np.ndarray, points: Tuple[int, int, int, int, int, int]) -> float:
    """Computes the Eye Aspect Ratio (EAR) for one eye.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    A high EAR indicates an open eye; a low EAR indicates a closed eye or
    blink. Points follow the classic Soukupova & Cech (2016) 6-point eye
    convention: p1/p4 are the horizontal corners, p2/p3/p5/p6 trace the
    upper and lower eyelid.
    """
    p1, p2, p3, p4, p5, p6 = points
    vertical_1 = _distance(landmarks, p2, p6)
    vertical_2 = _distance(landmarks, p3, p5)
    horizontal = _distance(landmarks, p1, p4)
    if horizontal <= 1e-6:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def mouth_aspect_ratio(landmarks: np.ndarray) -> float:
    """Computes the Mouth Aspect Ratio (MAR): vertical mouth opening
    normalized by mouth width. Higher values indicate a more open mouth."""
    vertical = _distance(landmarks, MOUTH_TOP_INNER, MOUTH_BOTTOM_INNER)
    horizontal = _distance(landmarks, MOUTH_LEFT_CORNER, MOUTH_RIGHT_CORNER)
    if horizontal <= 1e-6:
        return 0.0
    return vertical / horizontal


def mouth_opening_normalized(landmarks: np.ndarray) -> float:
    """Vertical mouth opening normalized by face height, invariant to
    camera distance."""
    vertical = _distance(landmarks, MOUTH_TOP_INNER, MOUTH_BOTTOM_INNER)
    return vertical / face_scale(landmarks)


def mouth_width_normalized(landmarks: np.ndarray) -> float:
    """Mouth width (distance between corners) normalized by face height.
    
    This is the key signal for detecting a pout/kiss: when lips purse
    forward, the mouth width shrinks significantly compared to a normal
    relaxed closed mouth.
    """
    width = _distance(landmarks, MOUTH_LEFT_CORNER, MOUTH_RIGHT_CORNER)
    return width / face_scale(landmarks)


def smile_intensity(landmarks: np.ndarray) -> float:
    """Estimates how strongly the mouth corners are raised (smiling) versus
    lowered (frowning), normalized by face height.

    Positive values indicate a smile (corners raised above the mouth
    center), negative values indicate a frown (corners drooping below the
    mouth center).
    """
    left_corner_y = landmarks[MOUTH_LEFT_CORNER][1]
    right_corner_y = landmarks[MOUTH_RIGHT_CORNER][1]
    top_y = landmarks[MOUTH_TOP_OUTER][1]
    bottom_y = landmarks[MOUTH_BOTTOM_OUTER][1]
    mouth_center_y = (top_y + bottom_y) / 2.0
    avg_corner_y = (left_corner_y + right_corner_y) / 2.0
    # In image/landmark space, y increases downward, so a raised corner has
    # a *smaller* y value than the mouth center.
    raise_amount = mouth_center_y - avg_corner_y
    return float(raise_amount / face_scale(landmarks))


def eyebrow_height(landmarks: np.ndarray) -> float:
    """Estimates average eyebrow elevation relative to the eyes, normalized
    by face height. Higher values indicate raised eyebrows (surprise);
    lower/negative values indicate furrowed/lowered eyebrows (anger)."""
    left_brow_y = np.mean([landmarks[i][1] for i in LEFT_EYEBROW_POINTS])
    right_brow_y = np.mean([landmarks[i][1] for i in RIGHT_EYEBROW_POINTS])
    left_eye_y = landmarks[LEFT_EYE_TOP][1]
    right_eye_y = landmarks[RIGHT_EYE_TOP][1]
    left_gap = left_eye_y - left_brow_y
    right_gap = right_eye_y - right_brow_y
    avg_gap = (left_gap + right_gap) / 2.0
    return float(avg_gap / face_scale(landmarks))


def jaw_opening(landmarks: np.ndarray) -> float:
    """Normalized vertical distance the jaw has dropped, approximated via
    chin-to-nose distance change is out of scope without a calibration
    frame; instead we approximate using mouth vertical opening as a proxy,
    which correlates strongly with jaw drop in practice."""
    return mouth_opening_normalized(landmarks)


def cheek_raise(landmarks: np.ndarray) -> float:
    """Rough proxy for cheek raise (as seen in genuine smiles / squinting)
    using how much the lower eyelid rises relative to eye width. This is a
    best-effort heuristic since MediaPipe does not track cheek muscles
    directly."""
    left_ear = eye_aspect_ratio(landmarks, LEFT_EYE_EAR_POINTS)
    right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_EAR_POINTS)
    # A gentle squint (moderately reduced EAR while mouth is smiling) is
    # associated with cheek raise; we simply return the inverse of EAR as a
    # bounded proxy signal for the analyzer to combine with smile_intensity.
    return float(max(0.0, 0.30 - ((left_ear + right_ear) / 2.0)))


def estimate_head_pose(
    landmarks: np.ndarray, frame_width: int, frame_height: int
) -> Optional[HeadPose]:
    """Estimates head yaw/pitch/roll (in degrees) using OpenCV's solvePnP
    against a generic 3D face model. Returns ``None`` if pose estimation
    fails (e.g. degenerate geometry) or OpenCV is unavailable."""
    if cv2 is None:
        return None

    image_points = np.array(
        [
            (landmarks[i][0] * frame_width, landmarks[i][1] * frame_height)
            for i in HEAD_POSE_LANDMARKS
        ],
        dtype=np.float64,
    )

    focal_length = frame_width
    center = (frame_width / 2.0, frame_height / 2.0)
    camera_matrix = np.array(
        [
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, _translation_vector = cv2.solvePnP(
        _HEAD_POSE_MODEL_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    sy = float(np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2))
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = np.arctan2(-rotation_matrix[2, 0], sy)
        roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        pitch = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        yaw = np.arctan2(-rotation_matrix[2, 0], sy)
        roll = 0.0

    return HeadPose(
        yaw=float(np.degrees(yaw)),
        pitch=float(np.degrees(pitch)),
        roll=float(np.degrees(roll)),
    )


def teeth_visibility_estimate(
    frame_bgr: Optional[np.ndarray],
    landmarks: np.ndarray,
    frame_width: int,
    frame_height: int,
    min_mouth_open: float = 0.05,
    brightness_threshold: float = 100.0,
    bright_pixel_ratio_threshold: float = 0.08,
) -> bool:
    """Best-effort teeth visibility heuristic: when the mouth is
    sufficiently open, sample the mean brightness inside the mouth region.
    Teeth are usually noticeably brighter than the surrounding lips/tongue.

    Args:
        frame_bgr: Source BGR frame for color sampling.
        landmarks: Face mesh landmarks.
        frame_width: Width of the source frame.
        frame_height: Height of the source frame.
        min_mouth_open: Minimum normalized mouth opening to attempt detection.
        brightness_threshold: Minimum mean brightness to consider teeth visible.
        bright_pixel_ratio_threshold: Minimum ratio of bright pixels (>150).

    Returns:
        ``False`` if the frame is unavailable or the mouth region cannot
        be sampled.
    """
    if frame_bgr is None or cv2 is None:
        return False
    if mouth_opening_normalized(landmarks) < min_mouth_open:
        return False

    roi = _mouth_roi(frame_bgr, landmarks, frame_width, frame_height)
    if roi is None or roi.size == 0:
        return False

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    bright_pixel_ratio = float(np.mean(gray > 150))
    return brightness > brightness_threshold and bright_pixel_ratio > bright_pixel_ratio_threshold


def tongue_visibility_estimate(
    frame_bgr: Optional[np.ndarray],
    landmarks: np.ndarray,
    frame_width: int,
    frame_height: int,
    min_mouth_open: float = 0.10,
    pinkish_ratio_threshold: float = 0.15,
    hue_min: int = 0,
    hue_max: int = 12,
    saturation_min: int = 60,
) -> bool:
    """Best-effort tongue visibility heuristic using HSV color analysis
    inside the mouth region: the tongue tends to be more saturated and
    reddish/pink compared to teeth (bright, low-saturation) or the dark
    interior of an open mouth.

    Args:
        frame_bgr: Source BGR frame for color sampling.
        landmarks: Face mesh landmarks.
        frame_width: Width of the source frame.
        frame_height: Height of the source frame.
        min_mouth_open: Minimum normalized mouth opening to attempt detection.
        pinkish_ratio_threshold: Minimum ratio of pinkish pixels.
        hue_min: Minimum hue for pink/red detection.
        hue_max: Maximum hue for pink/red detection.
        saturation_min: Minimum saturation for pink/red detection.
    """
    if frame_bgr is None or cv2 is None:
        return False
    if mouth_opening_normalized(landmarks) < min_mouth_open:
        return False

    roi = _mouth_roi(frame_bgr, landmarks, frame_width, frame_height)
    if roi is None or roi.size == 0:
        return False

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue, saturation, _value = cv2.split(hsv)
    mask = (hue >= hue_min) & (hue <= hue_max) & (saturation >= saturation_min)
    pinkish_ratio = float(np.mean(mask))
    return pinkish_ratio > pinkish_ratio_threshold


def _mouth_roi(
    frame_bgr: np.ndarray, landmarks: np.ndarray, frame_width: int, frame_height: int
) -> Optional[np.ndarray]:
    """Extracts a tight rectangular region of interest around the mouth."""
    xs = [
        landmarks[i][0] * frame_width
        for i in (MOUTH_LEFT_CORNER, MOUTH_RIGHT_CORNER, MOUTH_TOP_OUTER, MOUTH_BOTTOM_OUTER)
    ]
    ys = [
        landmarks[i][1] * frame_height
        for i in (MOUTH_LEFT_CORNER, MOUTH_RIGHT_CORNER, MOUTH_TOP_OUTER, MOUTH_BOTTOM_OUTER)
    ]
    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))
    pad_x = max(4, int((x_max - x_min) * 0.15))
    pad_y = max(4, int((y_max - y_min) * 0.30))
    x_min = max(0, x_min - pad_x)
    x_max = min(frame_width, x_max + pad_x)
    y_min = max(0, y_min - pad_y)
    y_max = min(frame_height, y_max + pad_y)
    if x_max <= x_min or y_max <= y_min:
        return None
    return frame_bgr[y_min:y_max, x_min:x_max]