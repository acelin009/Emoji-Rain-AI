"""
A single animated emoji particle.

Each :class:`Particle` is a lightweight, mutable unit of the "emoji rain"
animation: it knows its own position, velocity, rotation, scale, opacity,
and lifetime, and exposes ``update`` / ``is_dead`` methods that the
:class:`~src.particle_system.ParticleSystem` drives every frame. Rendering
itself is deliberately *not* part of this class (single responsibility) -
see :mod:`src.renderer`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.utils import clamp, inverse_lerp


@dataclass
class Particle:
    """A single floating emoji particle.

    Attributes:
        emoji: The unicode glyph this particle renders.
        color: Accent RGB color associated with the current expression,
            used for subtle glow/shadow effects by the renderer.
        x: Horizontal position in pixels.
        y: Vertical position in pixels.
        base_x: The particle's spawn-time X, used as the center of its
            horizontal drift oscillation.
        vy: Vertical (upward) speed in pixels/second (positive magnitude;
            the renderer/update logic subtracts it from y).
        rotation: Current rotation in degrees.
        rotation_speed: Rotation speed in degrees/second.
        scale: Current uniform scale factor.
        opacity: Current opacity in ``[0.0, 1.0]``.
        age: Seconds elapsed since spawn.
        lifetime: Total seconds this particle should live before despawning.
        drift_amplitude: Horizontal drift oscillation amplitude in pixels.
        drift_frequency: Horizontal drift oscillation frequency in Hz.
        drift_phase: Phase offset so particles don't all drift in sync.
        fade_in_duration: Seconds over which the particle fades in at spawn.
        fade_out_start_fraction: Fraction of ``lifetime`` at which fade-out
            begins (e.g. 0.75 means the last 25% of life fades to 0).
    """

    emoji: str
    color: tuple
    x: float
    y: float
    base_x: float
    vy: float
    rotation: float
    rotation_speed: float
    scale: float
    opacity: float
    age: float
    lifetime: float
    drift_amplitude: float
    drift_frequency: float
    drift_phase: float
    fade_in_duration: float
    fade_out_start_fraction: float

    def update(self, delta_time: float, screen_top_margin: float = 80.0) -> None:
        """Advances the particle's simulation state by ``delta_time``
        seconds: moves it upward, applies horizontal drift, rotates it, and
        updates its fade-based opacity.
        """
        self.age += delta_time
        self.y -= self.vy * delta_time
        oscillation = math.sin((self.age * self.drift_frequency * 2.0 * math.pi) + self.drift_phase)
        self.x = self.base_x + oscillation * self.drift_amplitude
        self.rotation = (self.rotation + self.rotation_speed * delta_time) % 360.0
        self.opacity = self._compute_opacity()

    def _compute_opacity(self) -> float:
        """Computes opacity based on fade-in and fade-out windows."""
        if self.fade_in_duration > 0 and self.age < self.fade_in_duration:
            fade_in = inverse_lerp(0.0, self.fade_in_duration, self.age)
        else:
            fade_in = 1.0

        fade_out_start_time = self.lifetime * self.fade_out_start_fraction
        if self.age >= fade_out_start_time:
            remaining = inverse_lerp(self.lifetime, fade_out_start_time, self.age)
            fade_out = remaining
        else:
            fade_out = 1.0

        return clamp(min(fade_in, fade_out), 0.0, 1.0)

    def is_dead(self, screen_top_margin: float = 80.0) -> bool:
        """A particle is considered dead once its lifetime has elapsed or
        it has floated far enough above the visible screen area that it
        can never be seen again."""
        return self.age >= self.lifetime or self.y < -screen_top_margin

    def fade(self) -> float:
        """Returns the current opacity (alias kept for API clarity / spec
        compliance: particles expose an explicit ``fade`` accessor)."""
        return self.opacity

    def destroy(self) -> None:
        """Marks the particle for immediate removal by forcing it past its
        lifetime. The actual removal from any containing collection is the
        responsibility of :class:`~src.particle_system.ParticleSystem`."""
        self.age = self.lifetime
        self.opacity = 0.0
