"""
Application-wide logging configuration.

Provides a single :func:`get_logger` factory that returns a fully
configured :class:`logging.Logger` with a colorized console handler and an
optional rotating file handler. All modules in the project should obtain
their logger via ``get_logger(__name__)`` rather than instantiating
``logging`` directly, to guarantee consistent formatting and output
destinations across the codebase.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

from src.config import DEFAULT_CONFIG, LOGS_DIR

_CONFIGURED = False


class _ColorFormatter(logging.Formatter):
    """A minimal ANSI color formatter for readable console output.

    Falls back gracefully (no color codes) when the output stream does not
    support ANSI escape sequences, e.g. when redirected to a file.
    """

    _COLORS = {
        logging.DEBUG: "\033[36m",  # cyan
        logging.INFO: "\033[32m",  # green
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",  # red
        logging.CRITICAL: "\033[41m",  # red background
    }
    _RESET = "\033[0m"

    def __init__(self, use_color: bool) -> None:
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        if not self._use_color:
            return formatted
        color = self._COLORS.get(record.levelno, "")
        return f"{color}{formatted}{self._RESET}" if color else formatted


def _configure_root_logger() -> None:
    """Configures the root ``emoji_rain_ai`` logger exactly once."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    cfg = DEFAULT_CONFIG.logging
    root = logging.getLogger("emoji_rain_ai")
    root.setLevel(logging.DEBUG)
    root.propagate = False

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(getattr(logging, cfg.console_level, logging.INFO))
    use_color = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    console_handler.setFormatter(_ColorFormatter(use_color=use_color))
    root.addHandler(console_handler)

    if cfg.log_to_file:
        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            file_path = LOGS_DIR / cfg.log_file_name
            file_handler = RotatingFileHandler(
                filename=str(file_path),
                maxBytes=cfg.max_bytes,
                backupCount=cfg.backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(getattr(logging, cfg.file_level, logging.DEBUG))
            file_handler.setFormatter(_ColorFormatter(use_color=False))
            root.addHandler(file_handler)
        except OSError:
            # Filesystem may be read-only in some environments (e.g. CI
            # sandboxes); logging to console only is an acceptable fallback.
            root.warning("Could not create log file; continuing with console logging only.")

    _CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Returns a configured child logger.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A :class:`logging.Logger` instance namespaced under
        ``emoji_rain_ai``.
    """
    _configure_root_logger()
    if name:
        return logging.getLogger(f"emoji_rain_ai.{name}")
    return logging.getLogger("emoji_rain_ai")
