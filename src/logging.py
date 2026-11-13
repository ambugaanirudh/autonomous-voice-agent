"""Structured logging setup using loguru.

Logs go to both the console (colorized) and a rotating log file under
logs/agent.log. Configured from config.yaml (app.log_level).
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logging(log_level: str = "INFO", log_dir: str | Path = "logs") -> None:
    """Configure loguru sinks: colorized console + rotating file."""
    # Remove the default sink so we control output entirely.
    logger.remove()

    # Console sink: colorized, human-readable.
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        colorize=True,
    )

    # File sink: rotating, retains last 5 files, 10 MB each.
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_path / "agent.log",
        level=log_level,
        rotation="10 MB",
        retention="5 days",
        enqueue=True,  # thread-safe for async / multi-threaded use
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )

    logger.debug("Logging initialized at level {level}", level=log_level)