"""
Particle system manager
"""

import random
import pygame
from typing import List, Optional, Tuple
from .particle import Particle
from ..core.logger import setup_logger


class ParticleSystem:
    """Manage a collection of particles"""
    
    def __init__(self, max_particles=50):
        self.particles: List[Particle] = []
        self.max_particles = max_particles
        self.logger = setup_logger('ParticleSystem')
        self.last_emoji = None
        self.cooldown = 0.0
        self.cooldown_time = 0.1  # seconds between spawns
    
    def spawn(self, emoji: str, x: float, y: float, count: int = 5):
        """Spawn multiple particles"""
        if self.cooldown > 0:
            return
        
        self.cooldown = self.cooldown_time
        
        for _ in range(min(count, self.max_particles - len(self.particles))):
            particle = self._create_particle(emoji, x, y)
            self.particles.append(particle)
        
        self.last_emoji = emoji
    
    def _create_particle(self, emoji: str, x: float, y: float) -> Particle:
        """Create a single particle with random properties"""
        # Random velocity in all directions
        angle = random.uniform(-math.pi, math.pi)
        speed = random.uniform(-150, 150)
        
        vx = math.cos(angle) * speed
        vy = math.sin(angle) * speed - 100  # Slight upward bias
        
        # Random lifetime
        lifetime = random.uniform(1.0, 2.5)
        
        return Particle(
            emoji=emoji,
            x=x + random.uniform(-20, 20),
            y=y + random.uniform(-20, 20),
            vx=vx,
            vy=vy,
            life=lifetime,
            max_life=lifetime,
            gravity=random.uniform(80, 120),
            drag=random.uniform(0.95, 0.99)
        )
    
    def update(self, dt: float):
        """Update all particles"""
        # Update cooldown
        if self.cooldown > 0:
            self.cooldown -= dt
        
        # Update each particle
        for particle in self.particles[:]:
            particle.update(dt)
            if not particle.is_alive():
                self.particles.remove(particle)
    
    def render(self, screen, camera_x: float = 0, camera_y: float = 0):
        """Render all particles"""
        # Sort by size (bigger particles behind smaller ones)
        sorted_particles = sorted(self.particles, key=lambda p: p.get_scale())
        
        for particle in sorted_particles:
            self._render_particle(screen, particle, camera_x, camera_y)
    
    def _render_particle(self, screen, particle: Particle, cx: float, cy: float):
        """Render a single particle"""
        try:
            # Get font (use system default if not available)
            font_size = int(32 * particle.get_scale())
            font = pygame.font.SysFont('segoeuiemoji', font_size)
            
            # Render emoji with alpha
            text_surf = font.render(particle.emoji, True, (255, 255, 255))
            
            # Apply alpha
            text_surf.set_alpha(int(255 * particle.get_alpha()))
            
            # Rotate
            rotated = pygame.transform.rotate(text_surf, particle.rotation)
            
            # Position
            x = particle.x - cx + rotated.get_width() // 2
            y = particle.y - cy + rotated.get_height() // 2
            
            # Draw
            screen.blit(rotated, (int(x), int(y)))
            
        except Exception as e:
            self.logger.debug(f"Error rendering particle: {e}")
    
    def clear(self):
        """Clear all particles"""
        self.particles.clear()
        self.last_emoji = None
    
    def get_particle_count(self) -> int:
        """Get number of active particles"""
        return len(self.particles)