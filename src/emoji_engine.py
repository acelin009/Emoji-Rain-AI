"""
Emoji Particle Engine
Handles spawning, updating, and rendering emojis.

Particles spawn just below the bottom edge of the frame and float
upward (rain from the bottom of the screen to the top).
"""

import random

from src.config import (
    MAX_PARTICLES,
    PARTICLE_SPEED,
    PARTICLE_SIZE,
    SPAWN_RATE,
    EMOJI_LIFETIME,
    EMOTION_TO_EMOJI
)


class EmojiParticle:
    """Single emoji particle."""

    def __init__(self, emoji, x, y, speed, size, color, lifetime, rotation=0):
        self.emoji = emoji
        self.x = x
        self.y = y
        self.speed = speed
        self.size = size
        self.color = color
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.rotation = rotation
        self.velocity_x = random.uniform(-0.5, 0.5)
        # Negative y-velocity = moving upward (screen coords: y=0 is top)
        self.velocity_y = -speed
        self.alive = True

    def update(self, dt):
        """Update particle position."""
        if not self.alive:
            return

        # Move
        self.x += self.velocity_x * dt * 60
        self.y += self.velocity_y * dt * 60

        # Slight horizontal drift
        self.velocity_x += random.uniform(-0.1, 0.1) * dt

        # Fade lifetime
        self.lifetime -= dt
        if self.lifetime <= 0:
            self.alive = False

    def get_alpha(self):
        """Get alpha based on lifetime."""
        if self.lifetime < 0.5:  # Fade out in last 0.5 seconds
            return max(0.0, self.lifetime / 0.5)
        return 1.0


class EmojiEngine:
    """Manages all emoji particles."""

    def __init__(self, max_particles=MAX_PARTICLES):
        self.particles = []
        self.max_particles = max_particles
        self.spawn_timer = 0
        self.spawn_interval = 1.0 / SPAWN_RATE
        self.current_emotion = 'Neutral'
        self.emoji_pool = EMOTION_TO_EMOJI['Neutral']

    def set_emotion(self, emotion):
        """Set current emotion and update emoji pool."""
        self.current_emotion = emotion
        self.emoji_pool = EMOTION_TO_EMOJI.get(emotion, EMOTION_TO_EMOJI['Neutral'])

    def spawn_emoji(self, frame_width, frame_height):
        """Spawn a new emoji particle just below the bottom of the frame."""
        if len(self.particles) >= self.max_particles:
            return

        if not self.emoji_pool:
            return

        # Random emoji from the current emotion's pool
        emoji = random.choice(self.emoji_pool)

        # Random x position across the width
        x = random.randint(50, max(51, frame_width - 50))

        # Spawn just below the visible frame so it rises INTO view
        y = frame_height + random.randint(0, 40)

        # Random upward speed
        speed = random.uniform(1.5, PARTICLE_SPEED)

        # Random size
        size = random.randint(25, PARTICLE_SIZE)

        # Random lifetime
        lifetime = random.uniform(1.5, EMOJI_LIFETIME)

        particle = EmojiParticle(
            emoji=emoji,
            x=x,
            y=y,
            speed=speed,
            size=size,
            color=(255, 255, 255),
            lifetime=lifetime,
            rotation=random.uniform(-0.5, 0.5)
        )

        self.particles.append(particle)

    def update(self, dt, frame_width, frame_height):
        """Update all particles."""
        # Spawn new particles
        self.spawn_timer += dt
        if self.spawn_timer >= self.spawn_interval:
            self.spawn_emoji(frame_width, frame_height)
            self.spawn_timer = 0

        # Update existing particles
        for particle in self.particles[:]:
            particle.update(dt)
            # Kill particles once they've fully risen off the top of frame
            if particle.y < -particle.size:
                particle.alive = False
            if not particle.alive:
                self.particles.remove(particle)

    def get_particles(self):
        """Get all active particles."""
        return self.particles

    def get_particle_count(self):
        """Get number of active particles."""
        return len(self.particles)
