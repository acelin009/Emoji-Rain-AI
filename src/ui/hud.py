"""
Main HUD rendering
"""

import pygame
from typing import Optional, Tuple
from .glass import GlassHUD, GlassPanel
from ..core.logger import setup_logger


class HUD:
    """Main HUD for the application"""
    
    def __init__(self, screen_width: int, screen_height: int, config):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.config = config
        self.glass_hud = GlassHUD(screen_width, screen_height, config)
        self.logger = setup_logger('HUD')
        
        # State
        self.current_emoji = '😐'
        self.expression_name = 'neutral'
        self.confidence = 0.0
        self.fps = 0
        self.particle_count = 0
        self.ear = (0, 0)
        self.mar = 0.0
        self.smile_score = 0.0
    
    def update(self, emoji: str, expression: str, confidence: float,
               fps: float, particle_count: int, features: dict):
        """Update HUD state"""
        self.current_emoji = emoji
        self.expression_name = expression
        self.confidence = confidence
        self.fps = fps
        self.particle_count = particle_count
        
        # Update features
        self.ear = features.get('ear', (0, 0))
        self.mar = features.get('mar', 0.0)
        self.smile_score = features.get('smile_score', 0.0)
    
    def draw(self, screen):
        """Draw the HUD"""
        # Draw glass panels
        self.glass_hud.draw(screen)
        
        # Draw content on panels
        self._draw_expression_panel(screen)
        self._draw_stats_panel(screen)
        self._draw_fps_panel(screen)
        
        # Draw big emoji in center (optional)
        self._draw_big_emoji(screen)
    
    def _draw_expression_panel(self, screen):
        """Draw the main expression panel"""
        panel = self.glass_hud.expression_panel
        center_x = panel.x + panel.width // 2
        center_y = panel.y + panel.height // 2
        
        # Draw big emoji
        font_size = 60
        font = pygame.font.SysFont('segoeuiemoji', font_size)
        emoji_surf = font.render(self.current_emoji, True, (255, 255, 255))
        emoji_rect = emoji_surf.get_rect(center=(center_x, center_y - 10))
        screen.blit(emoji_surf, emoji_rect)
        
        # Draw expression name
        font = pygame.font.SysFont('arial', 14)
        name_surf = font.render(self.expression_name.upper(), True, (200, 200, 255))
        name_rect = name_surf.get_rect(center=(center_x, panel.y + panel.height - 25))
        screen.blit(name_surf, name_rect)
        
        # Draw confidence bar
        bar_x = panel.x + 20
        bar_y = panel.y + panel.height - 10
        bar_width = panel.width - 40
        bar_height = 4
        
        # Background
        pygame.draw.rect(screen, (60, 60, 80), (bar_x, bar_y, bar_width, bar_height))
        
        # Fill
        fill_width = int(bar_width * self.confidence)
        color = (100, 255, 100) if self.confidence > 0.7 else (255, 200, 50)
        pygame.draw.rect(screen, color, (bar_x, bar_y, fill_width, bar_height))
    
    def _draw_stats_panel(self, screen):
        """Draw the stats panel"""
        panel = self.glass_hud.stats_panel
        x = panel.x + 15
        y = panel.y + 15
        
        # Draw stats
        font = pygame.font.SysFont('monospace', 12)
        lines = [
            f"EAR: {self.ear[0]:.3f} / {self.ear[1]:.3f}",
            f"MAR: {self.mar:.3f}",
            f"Smile: {self.smile_score:.2f}",
            f"Particles: {self.particle_count}"
        ]
        
        for i, line in enumerate(lines):
            text_surf = font.render(line, True, (200, 200, 255, 180))
            screen.blit(text_surf, (x, y + i * 20))
    
    def _draw_fps_panel(self, screen):
        """Draw the FPS panel"""
        panel = self.glass_hud.fps_panel
        x = panel.x + 15
        y = panel.y + 15
        
        font = pygame.font.SysFont('monospace', 18)
        fps_text = f"FPS: {self.fps:.1f}"
        color = (100, 255, 100) if self.fps > 25 else (255, 200, 50)
        text_surf = font.render(fps_text, True, color)
        screen.blit(text_surf, (x, y))
        
        # Performance bar
        bar_y = y + 35
        bar_width = panel.width - 30
        bar_height = 4
        
        # Performance based on FPS
        performance = min(self.fps / 30.0, 1.0)
        color = (100, 255, 100) if performance > 0.8 else (255, 200, 50)
        
        pygame.draw.rect(screen, (60, 60, 80), (x, bar_y, bar_width, bar_height))
        pygame.draw.rect(screen, color, (x, bar_y, int(bar_width * performance), bar_height))
    
    def _draw_big_emoji(self, screen):
        """Draw a large emoji in the center (subtle)"""
        # Only draw if we have high confidence
        if self.confidence < 0.6:
            return
        
        # Draw transparent large emoji in background
        font_size = 150
        font = pygame.font.SysFont('segoeuiemoji', font_size)
        emoji_surf = font.render(self.current_emoji, True, (255, 255, 255, 30))
        emoji_surf.set_alpha(30)
        
        rect = emoji_surf.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
        screen.blit(emoji_surf, rect)