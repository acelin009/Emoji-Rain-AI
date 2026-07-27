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

    def draw_debug_panel(self, canvas: Image.Image, state: AppState) -> None:
        """Draws the debug feature panel with raw geometric values."""
        if not self._config.show_debug_features:
            return

        margin_x, margin_y = self._config.hud_top_left_margin
        panel_x = margin_x + self._config.hud_width + 20
        width = self._config.debug_panel_width

        if state.debug_features is None:
            rows = [("No face detected", "", self._config.danger_color)]
        else:
            f = state.debug_features
            rows = [
                ("EAR L/R", f"{f.ear_left:.3f} / {f.ear_right:.3f}", self._config.text_color_primary),
                ("EAR AVG", f"{f.ear_avg:.3f}", self._config.text_color_primary),
                ("MAR", f"{f.mar:.3f}", self._config.text_color_primary),
                ("MOUTH OPEN", f"{f.mouth_open:.3f}", self._config.text_color_primary),
                ("MOUTH WIDTH", f"{f.mouth_width:.3f}", self._config.text_color_primary),  # NEW
                ("SMILE", f"{f.smile:.3f}", self._config.accent_color if f.smile > 0.04 else self._config.text_color_secondary),
                ("EYEBROW", f"{f.eyebrow:.3f}", self._config.text_color_primary),
                ("TEETH", f"{'YES' if f.teeth_visible else 'NO'}", self._config.success_color if f.teeth_visible else self._config.text_color_secondary),
                ("TONGUE", f"{'YES' if f.tongue_visible else 'NO'}", self._config.success_color if f.tongue_visible else self._config.text_color_secondary),
            ]

        if state.debug_raw_candidate is not None:
            rows.append(("RAW CANDIDATE", state.debug_raw_candidate.value, self._config.accent_color))

        row_height = self._config.font_size_body + 8
        header_height = self._config.font_size_title + 22
        height = header_height + row_height * len(rows) + self._config.panel_padding

        self._renderer.draw_glass_panel(canvas, (panel_x, margin_y), (width, height))

        pad = self._config.panel_padding
        cursor_y = margin_y + pad

        self._renderer.draw_text(
            canvas,
            (panel_x + pad, cursor_y),
            "FEATURES",
            size=self._config.font_size_title,
            color=self._config.accent_color,
            bold=True,
        )
        cursor_y += header_height

        for label, value, color in rows:
            self._renderer.draw_text(
                canvas,
                (panel_x + pad, cursor_y),
                label,
                size=self._config.font_size_small,
                color=self._config.text_color_secondary,
            )
            value_width, _ = self._renderer.measure_text(
                value, size=self._config.font_size_body, bold=True
            )
            value_x = panel_x + width - pad - value_width
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