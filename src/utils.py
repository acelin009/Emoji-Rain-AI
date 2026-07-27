"""
Small, dependency-free utility helpers shared across the codebase.

Keeping these in one module avoids duplicating simple math/timing helpers
(clamping, linear interpolation, FPS measurement, exponential smoothing)
across unrelated modules.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Tuple


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamps ``value`` to the inclusive range ``[minimum, maximum]``."""
    if minimum > maximum:
        minimum, maximum = maximum, minimum
    return max(minimum, min(maximum, value))


def lerp(start: float, end: float, t: float) -> float:
    """Linearly interpolates between ``start`` and ``end`` by ``t`` in [0, 1]."""
    return start + (end - start) * clamp(t, 0.0, 1.0)


def inverse_lerp(start: float, end: float, value: float) -> float:
    """Returns the interpolation factor ``t`` such that ``lerp(start, end, t)
    == value``. Safe against a zero-length range."""
    if end == start:
        return 0.0
    return clamp((value - start) / (end - start), 0.0, 1.0)


def exponential_moving_average(previous: float, current: float, alpha: float) -> float:
    """Blends ``previous`` and ``current`` using an exponential moving
    average with smoothing factor ``alpha`` (0 = ignore new value, 1 =
    ignore history)."""
    alpha = clamp(alpha, 0.0, 1.0)
    return previous * (1.0 - alpha) + current * alpha


def euclidean_distance_2d(point_a: Tuple[float, float], point_b: Tuple[float, float]) -> float:
    """Euclidean distance between two 2D points."""
    return ((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2) ** 0.5


class FPSCounter:
    """Tracks a rolling-window frames-per-second measurement.

    Usage:
        fps_counter = FPSCounter(window_size=30)
        while running:
            ...
            fps_counter.tick()
            current_fps = fps_counter.fps
    """

    def __init__(self, window_size: int = 30) -> None:
        self._timestamps: Deque[float] = deque(maxlen=max(2, window_size))

    def tick(self) -> float:
        """Records a frame timestamp and returns the current FPS estimate."""
        self._timestamps.append(time.perf_counter())
        return self.fps

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed


class DeltaTimer:
    """Measures the elapsed wall-clock time (in seconds) between successive
    calls to :meth:`tick`, used to drive frame-rate independent animation
    (particle motion, spawn accumulation, etc.)."""

    def __init__(self) -> None:
        self._last_time = time.perf_counter()

    def tick(self, max_delta: float = 0.1) -> float:
        """Returns the elapsed time since the previous call, clamped to
        ``max_delta`` to avoid large simulation jumps after a stall (e.g.
        window drag, breakpoint, GC pause)."""
        now = time.perf_counter()
        delta = now - self._last_time
        self._last_time = now
        return clamp(delta, 0.0, max_delta)
