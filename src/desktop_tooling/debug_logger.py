"""
Simple file logger for debug mode.

Writes timestamped entries to a .log file.
- Production (PyInstaller): logs next to the .exe
- Development: logs to project root
"""
import os
import sys
import logging
import traceback
from datetime import datetime


_logger: logging.Logger | None = None


def init_logger(enabled: bool = False) -> None:
    """Initialize the debug logger. Call once at startup."""
    global _logger

    if not enabled:
        _logger = None
        return

    # Always write to user home — easy to find regardless of frozen/dev
    log_dir = os.path.expanduser('~')
    log_path = os.path.join(log_dir, 'pdf_classifier.log')

    _logger = logging.getLogger('pdf_classifier')
    _logger.setLevel(logging.DEBUG)
    _logger.handlers.clear()

    handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-7s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    _logger.addHandler(handler)

    _logger.info('=' * 60)
    _logger.info(f'Session started — {datetime.now().isoformat()}')
    _logger.info(f'Log file: {log_path}')
    _logger.info(f'Frozen: {getattr(sys, "frozen", False)}')
    _logger.info('=' * 60)


def debug(msg: str) -> None:
    """Log a debug message."""
    if _logger:
        _logger.debug(msg)


def info(msg: str) -> None:
    """Log an info message."""
    if _logger:
        _logger.info(msg)


def error(msg: str, exc: Exception = None) -> None:
    """Log an error with optional traceback."""
    if _logger:
        _logger.error(msg)
        if exc:
            _logger.error(traceback.format_exc())


def log_phase(phase: str, status: str, **extra) -> None:
    """Log a progress phase transition."""
    if _logger:
        parts = f'[{phase}] {status}'
        if extra:
            details = ', '.join(f'{k}={v}' for k, v in extra.items())
            parts += f' — {details}'
        _logger.info(parts)
