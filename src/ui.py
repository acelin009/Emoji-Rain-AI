"""
Heads-up display (HUD) overlay.

Draws the glassmorphism stats panel (FPS, current expression, confidence,
particle count, landmark count, detection status, camera status) onto the
composited frame. Built on top of :class:`~src.renderer.Renderer`'s
drawing primitives so all visual styling stays consistent and centrally
themeable via :class:`~src.config.UIConfig`.
"""

from __future__ import annotations

from PIL import Image

from src.app_state import AppState, CameraStatus, DetectionStatus
from src.config import UIConfig
from src.emoji_engine import EmojiStyle
from src.renderer import Renderer


class UIOverlay:
    """Renders the application's on-screen stats HUD.

    Example:
        overlay = UIOverlay(ui_config, renderer)
        overlay.draw(canvas, app_state, emoji_style)
    """

    def __init__(self, config: UIConfig, renderer: Renderer) -> None:
        self._config = config
        self._renderer = renderer

    def draw(self, canvas: Image.Image, state: AppState, style: EmojiStyle) -> None:
        """Draws the full HUD onto ``canvas`` in place."""
        margin_x, margin_y = self._config.hud_top_left_margin
        width = self._config.hud_width
        rows = self._build_rows(state)
        row_height = self._config.font_size_body + 10
        header_height = self._config.font_size_title + 22
        height = header_height + row_height * len(rows) + self._config.panel_padding

        self._renderer.draw_glass_panel(canvas, (margin_x, margin_y), (width, height))

        pad = self._config.panel_padding
        cursor_y = margin_y + pad

        title_text = f"{style.glyph}  {self._config.window_title}"
        self._renderer.draw_text(
            canvas,
            (margin_x + pad, cursor_y),
            title_text,
            size=self._config.font_size_title,
            color=self._config.text_color_primary,
            bold=True,
        )
        cursor_y += header_height

        for label, value, color in rows:
            self._renderer.draw_text(
                canvas,
                (margin_x + pad, cursor_y),
                label,
                size=self._config.font_size_small,
                color=self._config.text_color_secondary,
            )
            value_width, _ = self._renderer.measure_text(
                value, size=self._config.font_size_body, bold=True
            )
            value_x = margin_x + width - pad - value_width
            self._renderer.draw_text(
                canvas,
                (value_x, cursor_y - 2),
                value,
                size=self._config.font_size_body,
                color=color,
                bold=True,
            )
            cursor_y += row_height

    def _build_rows(self, state: AppState):
        cfg = self._config
        rows = []

        if cfg.show_fps:
            fps_color = cfg.success_color if state.fps >= 24 else cfg.danger_color
            rows.append(("FPS", f"{state.fps:.1f}", fps_color))

        rows.append(("EXPRESSION", state.current_expression.value, cfg.accent_color))
        rows.append(
            ("CONFIDENCE", f"{state.expression_confidence * 100:.0f}%", cfg.text_color_primary)
        )
        rows.append(("PARTICLES", str(state.particle_count), cfg.text_color_primary))
        rows.append(("LANDMARKS", str(state.landmark_count), cfg.text_color_primary))

        detection_color = (
            cfg.success_color
            if state.detection_status == DetectionStatus.FACE_DETECTED
            else cfg.danger_color
        )
        rows.append(("DETECTION", state.detection_status.value, detection_color))

        camera_color = (
            cfg.success_color if state.camera_status == CameraStatus.CONNECTED else cfg.danger_color
        )
        rows.append(("CAMERA", state.camera_status.value, camera_color))

        return rows
