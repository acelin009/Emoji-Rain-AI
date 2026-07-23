"""
Glassmorphism UI components
"""

import pygame
import math
from typing import Tuple, Optional


class GlassPanel:
    """A glassmorphism panel"""
    
    def __init__(self, x: int, y: int, width: int, height: int, 
                 color: Tuple[int, int, int, int] = (255, 255, 255, 30),
                 border_color: Tuple[int, int, int, int] = (255, 255, 255, 80),
                 corner_radius: int = 15,
                 blur: bool = False):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.border_color = border_color
        self.corner_radius = corner_radius
        self.blur = blur
        self.visible = True
    
    def draw(self, screen):
        """Draw the glass panel"""
        if not self.visible:
            return
        
        # Create surface with alpha
        surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        # Draw rounded rectangle background
        rect = pygame.Rect(0, 0, self.width, self.height)
        pygame.draw.rect(surf, self.color, rect, border_radius=self.corner_radius)
        
        # Draw border
        pygame.draw.rect(surf, self.border_color, rect, width=2, border_radius=self.corner_radius)
        
        # Add glass effect (subtle gradient)
        self._add_glass_effect(surf)
        
        # Blit to screen
        screen.blit(surf, (self.x, self.y))
    
    def _add_glass_effect(self, surf):
        """Add subtle glass reflection effect"""
        # Top highlight
        highlight = pygame.Surface((self.width - 20, self.height // 4), pygame.SRCALPHA)
        highlight.fill((255, 255, 255, 15))
        surf.blit(highlight, (10, 10))
        
        # Bottom-right subtle glow
        glow = pygame.Surface((self.width // 2, self.height // 3), pygame.SRCALPHA)
        glow.fill((255, 255, 255, 8))
        surf.blit(glow, (self.width // 2, self.height * 2 // 3))


class GlassHUD:
    """HUD with glassmorphism panels"""
    
    def __init__(self, screen_width, screen_height, config):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.config = config
        self.panels = []
        self.setup_panels()
    
    def setup_panels(self):
        """Setup HUD panels"""
        padding = 20
        
        # Top-left panel (FPS, performance)
        self.fps_panel = GlassPanel(
            padding, padding,
            200, 100,
            color=(30, 30, 40, 180),
            border_color=(255, 255, 255, 80)
        )
        self.panels.append(self.fps_panel)
        
        # Top-center panel (Current expression)
        self.expression_panel = GlassPanel(
            self.screen_width // 2 - 150, padding,
            300, 150,
            color=(30, 30, 40, 200),
            border_color=(255, 255, 255, 100)
        )
        self.panels.append(self.expression_panel)
        
        # Top-right panel (Confidence, stats)
        self.stats_panel = GlassPanel(
            self.screen_width - 220, padding,
            200, 120,
            color=(30, 30, 40, 180),
            border_color=(255, 255, 255, 80)
        )
        self.panels.append(self.stats_panel)
    
    def draw(self, screen):
        """Draw all HUD panels"""
        for panel in self.panels:
            panel.draw(screen)
    
    def draw_text(self, screen, text, x, y, color=(255, 255, 255), 
                  font_size=16, centered=False, panel_index=None):
        """Draw text with optional centering"""
        font = pygame.font.SysFont('arial', font_size)
        text_surf = font.render(text, True, color)
        
        if centered:
            text_rect = text_surf.get_rect(center=(x, y))
            x = text_rect.x
            y = text_rect.y
        
        screen.blit(text_surf, (x, y))