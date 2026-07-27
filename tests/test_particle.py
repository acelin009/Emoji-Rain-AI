"""Unit tests for src.particle.Particle and src.particle_system.ParticleSystem."""

from __future__ import annotations

from src.config import ParticleConfig
from src.emoji_engine import EmojiStyle
from src.particle import Particle
from src.particle_system import ParticleSystem


def _make_particle(**overrides) -> Particle:
    defaults = dict(
        emoji="\U0001f60a",
        color=(255, 209, 102),
        x=100.0,
        y=500.0,
        base_x=100.0,
        vy=100.0,
        rotation=0.0,
        rotation_speed=10.0,
        scale=1.0,
        opacity=0.0,
        age=0.0,
        lifetime=4.0,
        drift_amplitude=20.0,
        drift_frequency=1.0,
        drift_phase=0.0,
        fade_in_duration=0.2,
        fade_out_start_fraction=0.75,
    )
    defaults.update(overrides)
    return Particle(**defaults)


class TestParticle:
    def test_update_moves_particle_upward(self):
        particle = _make_particle(y=500.0, vy=100.0)
        particle.update(delta_time=1.0)
        assert particle.y == 400.0

    def test_update_increments_age(self):
        particle = _make_particle(age=0.0)
        particle.update(delta_time=0.5)
        assert particle.age == 0.5

    def test_update_rotates_particle(self):
        particle = _make_particle(rotation=0.0, rotation_speed=90.0)
        particle.update(delta_time=1.0)
        assert particle.rotation == 90.0

    def test_update_wraps_rotation_at_360(self):
        particle = _make_particle(rotation=350.0, rotation_speed=20.0)
        particle.update(delta_time=1.0)
        assert particle.rotation == 10.0

    def test_fade_in_ramps_opacity_up(self):
        particle = _make_particle(fade_in_duration=1.0, lifetime=10.0, opacity=0.0)
        particle.update(delta_time=0.5)
        assert 0.0 < particle.fade() < 1.0

    def test_fully_faded_in_reaches_full_opacity(self):
        particle = _make_particle(fade_in_duration=0.2, lifetime=10.0, fade_out_start_fraction=0.9)
        particle.update(delta_time=0.3)
        assert particle.fade() == 1.0

    def test_fade_out_reduces_opacity_near_end_of_life(self):
        particle = _make_particle(
            fade_in_duration=0.1, lifetime=4.0, fade_out_start_fraction=0.5, age=0.0
        )
        particle.update(delta_time=3.9)  # 97.5% through a 4s lifetime
        assert particle.fade() < 0.2

    def test_is_dead_when_lifetime_exceeded(self):
        particle = _make_particle(lifetime=2.0)
        particle.update(delta_time=2.5)
        assert particle.is_dead() is True

    def test_is_not_dead_within_lifetime(self):
        particle = _make_particle(lifetime=5.0)
        particle.update(delta_time=1.0)
        assert particle.is_dead() is False

    def test_is_dead_when_above_screen(self):
        particle = _make_particle(y=-200.0, lifetime=100.0, age=0.0)
        assert particle.is_dead(screen_top_margin=80.0) is True

    def test_destroy_forces_particle_dead(self):
        particle = _make_particle(lifetime=100.0, age=0.0)
        particle.destroy()
        assert particle.is_dead() is True
        assert particle.fade() == 0.0

    def test_horizontal_drift_oscillates_around_base_x(self):
        particle = _make_particle(base_x=200.0, drift_amplitude=30.0, drift_frequency=1.0)
        positions = []
        for _ in range(20):
            particle.update(delta_time=0.05)
            positions.append(particle.x)
        assert max(positions) <= 200.0 + 30.0 + 1e-6
        assert min(positions) >= 200.0 - 30.0 - 1e-6


def _default_particle_config(**overrides) -> ParticleConfig:
    defaults = dict(
        max_particles=10,
        spawn_rate_per_second=5.0,
        min_scale=0.8,
        max_scale=1.2,
        min_speed=50.0,
        max_speed=100.0,
        horizontal_drift_amplitude=10.0,
        horizontal_drift_frequency=1.0,
        min_rotation_speed=-10.0,
        max_rotation_speed=10.0,
        min_lifetime=2.0,
        max_lifetime=3.0,
        fade_in_duration=0.1,
        fade_out_start_fraction=0.75,
        base_font_size=32,
        spawn_margin_px=10,
    )
    defaults.update(overrides)
    return ParticleConfig(**defaults)


class TestParticleSystem:
    def test_spawns_particles_over_time(self):
        system = ParticleSystem(_default_particle_config(spawn_rate_per_second=10.0))
        style = EmojiStyle(glyph="\U0001f60a", color=(255, 209, 102))

        for _ in range(30):
            system.update(delta_time=1.0 / 30.0, style=style, frame_width=640, frame_height=480)

        assert system.count > 0

    def test_respects_max_particle_cap(self):
        config = _default_particle_config(
            max_particles=5, spawn_rate_per_second=1000.0, min_lifetime=10.0, max_lifetime=10.0
        )
        system = ParticleSystem(config)
        style = EmojiStyle(glyph="\U0001f602", color=(255, 138, 91))

        # A single large-delta frame requests far more than 5 particles;
        # the population must still be capped at max_particles.
        system.update(delta_time=1.0, style=style, frame_width=640, frame_height=480)

        assert system.count <= 5
        assert system.is_at_capacity is True

        for _ in range(10):
            system.update(delta_time=0.5, style=style, frame_width=640, frame_height=480)
            assert system.count <= 5

    def test_disabling_spawn_stops_new_particles_but_keeps_existing(self):
        config = _default_particle_config(
            spawn_rate_per_second=1000.0, min_lifetime=5.0, max_lifetime=5.0
        )
        system = ParticleSystem(config)
        style = EmojiStyle(glyph="\U0001f609", color=(114, 214, 255))

        system.update(delta_time=1.0, style=style, frame_width=640, frame_height=480)
        count_after_spawn = system.count
        assert count_after_spawn > 0

        system.set_spawn_enabled(False)
        system.update(delta_time=0.1, style=style, frame_width=640, frame_height=480)

        assert system.count == count_after_spawn

    def test_dead_particles_are_pruned(self):
        config = _default_particle_config(
            spawn_rate_per_second=1000.0, min_lifetime=0.5, max_lifetime=0.5, max_particles=20
        )
        system = ParticleSystem(config)
        style = EmojiStyle(glyph="\U0001f622", color=(108, 156, 255))

        system.update(delta_time=0.05, style=style, frame_width=640, frame_height=480)
        assert system.count > 0

        system.update(delta_time=2.0, style=style, frame_width=640, frame_height=480)
        assert system.count == 0

    def test_clear_removes_all_particles(self):
        config = _default_particle_config(spawn_rate_per_second=1000.0)
        system = ParticleSystem(config)
        style = EmojiStyle(glyph="\U0001f620", color=(255, 82, 82))

        system.update(delta_time=1.0, style=style, frame_width=640, frame_height=480)
        assert system.count > 0

        system.clear()
        assert system.count == 0

    def test_total_spawned_counter_increases(self):
        config = _default_particle_config(spawn_rate_per_second=1000.0, max_particles=1000)
        system = ParticleSystem(config)
        style = EmojiStyle(glyph="\U0001f61c", color=(198, 130, 255))

        system.update(delta_time=1.0, style=style, frame_width=640, frame_height=480)
        assert system.total_spawned > 0
