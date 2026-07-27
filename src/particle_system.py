"""
Manages the full population of active :class:`~src.particle.Particle`
instances: spawning new particles at a configurable rate, updating them
every frame, and pruning dead ones. Enforces a hard cap on the number of
simultaneously alive particles to keep frame time bounded.
"""

from __future__ import annotations

import random
from typing import List

from src.config import ParticleConfig
from src.emoji_engine import EmojiStyle
from src.logger import get_logger
from src.particle import Particle

logger = get_logger(__name__)


class ParticleSystem:
    """Owns and drives the full set of active emoji particles.

    Example:
        system = ParticleSystem(config.particles)
        system.set_spawn_enabled(True)
        while running:
            system.update(delta_time, active_style, frame_width, frame_height)
            for particle in system.particles:
                render(particle)
    """

    def __init__(self, config: ParticleConfig) -> None:
        self._config = config
        self._particles: List[Particle] = []
        self._spawn_accumulator = 0.0
        self._spawn_enabled = True
        self._total_spawned = 0

    def set_spawn_enabled(self, enabled: bool) -> None:
        """Globally enables/disables spawning of new particles (existing
        particles continue to animate to completion)."""
        self._spawn_enabled = enabled

    def update(
        self,
        delta_time: float,
        style: EmojiStyle,
        frame_width: int,
        frame_height: int,
    ) -> None:
        """Advances the simulation by one frame: spawns new particles (if
        under the population cap and spawning is enabled), updates all
        existing particles, and removes dead ones.
        """
        if self._spawn_enabled:
            self._accumulate_and_spawn(delta_time, style, frame_width, frame_height)

        for particle in self._particles:
            particle.update(delta_time)

        self._particles = [p for p in self._particles if not p.is_dead()]

    def _accumulate_and_spawn(
        self, delta_time: float, style: EmojiStyle, frame_width: int, frame_height: int
    ) -> None:
        self._spawn_accumulator += delta_time * self._config.spawn_rate_per_second
        while self._spawn_accumulator >= 1.0:
            self._spawn_accumulator -= 1.0
            if len(self._particles) < self._config.max_particles:
                self._spawn_particle(style, frame_width, frame_height)

    def _spawn_particle(self, style: EmojiStyle, frame_width: int, frame_height: int) -> None:
        cfg = self._config
        margin = cfg.spawn_margin_px
        base_x = random.uniform(margin, max(margin + 1, frame_width - margin))
        spawn_y = frame_height + random.uniform(10, 60)

        particle = Particle(
            emoji=style.glyph,
            color=style.color,
            x=base_x,
            y=spawn_y,
            base_x=base_x,
            vy=random.uniform(cfg.min_speed, cfg.max_speed),
            rotation=random.uniform(0.0, 360.0),
            rotation_speed=random.uniform(cfg.min_rotation_speed, cfg.max_rotation_speed),
            scale=random.uniform(cfg.min_scale, cfg.max_scale),
            opacity=0.0,
            age=0.0,
            lifetime=random.uniform(cfg.min_lifetime, cfg.max_lifetime),
            drift_amplitude=cfg.horizontal_drift_amplitude * random.uniform(0.5, 1.0),
            drift_frequency=cfg.horizontal_drift_frequency * random.uniform(0.7, 1.3),
            drift_phase=random.uniform(0.0, 6.28318),
            fade_in_duration=cfg.fade_in_duration,
            fade_out_start_fraction=cfg.fade_out_start_fraction,
        )
        self._particles.append(particle)
        self._total_spawned += 1

    def clear(self) -> None:
        """Immediately removes all particles (e.g. on app reset)."""
        count = len(self._particles)
        self._particles.clear()
        if count:
            logger.debug("Cleared %d particles.", count)

    @property
    def particles(self) -> List[Particle]:
        return self._particles

    @property
    def count(self) -> int:
        return len(self._particles)

    @property
    def total_spawned(self) -> int:
        return self._total_spawned

    @property
    def is_at_capacity(self) -> bool:
        return len(self._particles) >= self._config.max_particles
