"""
Individual particle (emoji) class
"""

import random
import math
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Particle:
    """A single emoji particle with physics"""
    emoji: str
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    rotation: float = 0.0
    scale: float = 1.0
    rot_speed: float = 0.0
    gravity: float = 50.0
    drag: float = 0.98
    alpha: float = 1.0
    
    def __post_init__(self):
        """Initialize random properties if not set"""
        if self.rotation == 0.0:
            self.rotation = random.uniform(-180, 180)
        if self.rot_speed == 0.0:
            self.rot_speed = random.uniform(-200, 200)
        if self.scale == 1.0:
            self.scale = random.uniform(0.5, 1.5)
    
    def update(self, dt: float):
        """Update particle physics"""
        # Apply gravity
        self.vy += self.gravity * dt
        
        # Apply drag
        self.vx *= self.drag
        self.vy *= self.drag
        
        # Update position
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Update rotation
        self.rotation += self.rot_speed * dt
        
        # Update life
        self.life -= dt
        
        # Update alpha (fade out near end of life)
        life_ratio = self.life / self.max_life
        self.alpha = min(1.0, life_ratio * 2.0)  # Fade out in last 50% of life
    
    def is_alive(self) -> bool:
        """Check if particle is still alive"""
        return self.life > 0
    
    def get_scale(self) -> float:
        """Get current scale (with life-based shrinking)"""
        life_ratio = self.life / self.max_life
        return self.scale * (0.3 + 0.7 * life_ratio)
    
    def get_alpha(self) -> float:
        """Get current alpha (for fading)"""
        return self.alpha