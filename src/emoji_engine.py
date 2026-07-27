"""
Bridges the expression-analysis layer and the particle-animation layer.

:class:`EmojiEngine` owns the mapping from a recognized
:class:`~src.expression_analyzer.Expression` to the emoji glyph and accent
color that should currently be "raining", and exposes a small, explicit
API that the particle system and UI consume. Keeping this mapping in its
own module means the emoji vocabulary (and thus the app's personality) can
be changed without touching classification or animation logic.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import EmojiMappingConfig
from src.expression_analyzer import Expression
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class EmojiStyle:
    """The emoji glyph and accent color currently associated with an
    expression."""

    glyph: str
    color: tuple


class EmojiEngine:
    """Tracks the "active" emoji that the particle system should spawn,
    and logs transitions when the underlying expression changes.

    Example:
        engine = EmojiEngine(config.emoji)
        engine.update(expression_result.expression)
        style = engine.current_style
    """

    def __init__(self, config: EmojiMappingConfig) -> None:
        self._config = config
        self._current_expression: Expression = Expression.NEUTRAL
        self._current_style = self._resolve_style(Expression.NEUTRAL)

    def update(self, expression: Expression) -> EmojiStyle:
        """Updates the active expression, returning the resulting emoji
        style. Spawning of the *previous* emoji naturally stops the moment
        the caller starts requesting the new glyph from
        :attr:`current_style`; already-airborne particles are left to
        finish their own lifetime/fade animation undisturbed.
        """
        if expression != self._current_expression:
            logger.info(
                "EmojiEngine switching glyph: %s (%s) -> %s (%s)",
                self._current_expression.value,
                self._current_style.glyph,
                expression.value,
                self._resolve_style(expression).glyph,
            )
            self._current_expression = expression
            self._current_style = self._resolve_style(expression)
        return self._current_style

    def _resolve_style(self, expression: Expression) -> EmojiStyle:
        glyph = self._config.mapping.get(expression.value, "\U0001f610")
        color = self._config.accent_colors.get(expression.value, (200, 200, 200))
        return EmojiStyle(glyph=glyph, color=color)

    @property
    def current_style(self) -> EmojiStyle:
        return self._current_style

    @property
    def current_expression(self) -> Expression:
        return self._current_expression
