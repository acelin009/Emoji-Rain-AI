"""
Frame rendering / compositing layer.

Responsible for turning the raw OpenCV (BGR, numpy) camera frame plus the
current particle population into the final composited image the user sees:
color emoji glyphs, drop shadows, and glassmorphism panels. OpenCV has no
native support for drawing color emoji glyphs or alpha-blended rounded
rectangles, so this module uses Pillow (PIL) for all high-level drawing and
converts back to a numpy BGR array only once per frame for display.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import cache
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.config import FONTS_DIR, ParticleConfig, UIConfig
from src.logger import get_logger
from src.particle import Particle

logger = get_logger(__name__)

# Common system locations for color/monochrome emoji-capable fonts, checked
# in order. If none are found, a bundled/default font is used as a
# best-effort fallback (glyphs may render as empty boxes on some systems -
# see README "Emoji Font Setup").
_CANDIDATE_EMOJI_FONT_PATHS: Tuple[str, ...] = (
    str(FONTS_DIR / "NotoColorEmoji.ttf"),
    str(FONTS_DIR / "Seguiemj.ttf"),
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/noto/NotoColorEmoji.ttf",
    "C:/Windows/Fonts/seguiemj.ttf",
    "/System/Library/Fonts/Apple Color Emoji.ttc",
)

_CANDIDATE_UI_FONT_PATHS: Tuple[str, ...] = (
    str(FONTS_DIR / "Inter-Regular.ttf"),
    str(FONTS_DIR / "Roboto-Regular.ttf"),
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "/System/Library/Fonts/SFNS.ttf",
)

_CANDIDATE_UI_BOLD_FONT_PATHS: Tuple[str, ...] = (
    str(FONTS_DIR / "Inter-Bold.ttf"),
    str(FONTS_DIR / "Roboto-Bold.ttf"),
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
    "/System/Library/Fonts/SFNSBold.ttf",
)


@cache
def _load_font(candidates: Tuple[str, ...], size: int) -> ImageFont.FreeTypeFont:
    """Loads the first available font from ``candidates`` at ``size``,
    falling back to Pillow's built-in default font if none exist."""
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    logger.warning(
        "No suitable font found among candidates %s; using Pillow default font. "
        "See README 'Emoji Font Setup' to enable full-color emoji rendering.",
        candidates,
    )
    return ImageFont.load_default()


def get_emoji_font(size: int) -> ImageFont.FreeTypeFont:
    """Returns a (cached) font capable of rendering emoji glyphs at the
    given pixel size."""
    return _load_font(_CANDIDATE_EMOJI_FONT_PATHS, size)


def get_ui_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Returns a (cached) UI font at the given pixel size."""
    candidates = _CANDIDATE_UI_BOLD_FONT_PATHS if bold else _CANDIDATE_UI_FONT_PATHS
    return _load_font(candidates, size)


class Renderer:
    """Composites particles and UI chrome onto each camera frame.

    Example:
        renderer = Renderer(particle_config, ui_config)
        canvas = renderer.begin_frame(bgr_frame)
        renderer.draw_particles(canvas, particle_system.particles)
        bgr_out = renderer.finish_frame(canvas)
    """

    def __init__(self, particle_config: ParticleConfig, ui_config: UIConfig) -> None:
        self._particle_config = particle_config
        self._ui_config = ui_config

    def begin_frame(self, frame_bgr: np.ndarray) -> Image.Image:
        """Converts a BGR numpy frame into an RGBA PIL image ready for
        layered drawing."""
        rgb = frame_bgr[:, :, ::-1]
        return Image.fromarray(rgb, mode="RGB").convert("RGBA")

    def finish_frame(self, canvas: Image.Image) -> np.ndarray:
        """Converts the composited RGBA canvas back into a BGR numpy array
        suitable for ``cv2.imshow``."""
        rgb = canvas.convert("RGB")
        array = np.array(rgb)
        return array[:, :, ::-1].copy()

    def draw_particles(self, canvas: Image.Image, particles: Iterable[Particle]) -> None:
        """Draws every particle onto ``canvas`` as a rotated, scaled,
        alpha-blended emoji glyph."""
        for particle in particles:
            self._draw_single_particle(canvas, particle)

    def _draw_single_particle(self, canvas: Image.Image, particle: Particle) -> None:
        if particle.opacity <= 0.01:
            return

        font_size = max(8, int(self._particle_config.base_font_size * particle.scale))
        font = get_emoji_font(font_size)

        # Render the glyph onto its own small transparent tile so it can be
        # independently rotated and alpha-composited without disturbing the
        # rest of the canvas.
        tile_size = int(font_size * 1.8)
        tile = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))
        tile_draw = ImageDraw.Draw(tile)
        try:
            tile_draw.text(
                (tile_size / 2, tile_size / 2),
                particle.emoji,
                font=font,
                anchor="mm",
                embedded_color=True,
            )
        except TypeError:
            # Older Pillow versions lack `embedded_color`; degrade gracefully.
            tile_draw.text((tile_size / 2, tile_size / 2), particle.emoji, font=font, anchor="mm")

        if particle.rotation:
            tile = tile.rotate(particle.rotation, resample=Image.BICUBIC, expand=False)

        alpha = tile.split()[-1].point(lambda a: int(a * particle.opacity))
        tile.putalpha(alpha)

        paste_x = int(particle.x - tile_size / 2)
        paste_y = int(particle.y - tile_size / 2)
        canvas.alpha_composite(tile, dest=(paste_x, paste_y))

    def draw_glass_panel(
        self,
        canvas: Image.Image,
        top_left: Tuple[int, int],
        size: Tuple[int, int],
    ) -> None:
        """Draws a semi-transparent, rounded-rectangle "glassmorphism"
        panel at ``top_left`` with the given ``size`` (width, height)."""
        cfg = self._ui_config
        x, y = top_left
        width, height = size

        panel = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)
        draw.rounded_rectangle(
            [(0, 0), (width - 1, height - 1)],
            radius=cfg.panel_corner_radius,
            fill=cfg.panel_background_rgba,
            outline=cfg.panel_border_rgba,
            width=1,
        )
        canvas.alpha_composite(panel, dest=(x, y))

    def draw_text(
        self,
        canvas: Image.Image,
        position: Tuple[int, int],
        text: str,
        size: int,
        color: Tuple[int, int, int],
        bold: bool = False,
    ) -> Tuple[int, int]:
        """Draws UI text at ``position`` and returns its rendered
        ``(width, height)`` for simple manual layout."""
        font = get_ui_font(size, bold=bold)
        draw = ImageDraw.Draw(canvas)
        draw.text(position, text, font=font, fill=(*color, 255))
        bbox = draw.textbbox(position, text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    def measure_text(self, text: str, size: int, bold: bool = False) -> Tuple[int, int]:
        """Measures rendered text size without drawing it."""
        font = get_ui_font(size, bold=bold)
        scratch = Image.new("RGBA", (1, 1))
        draw = ImageDraw.Draw(scratch)
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
