"""structlog configuration from settings."""

from __future__ import annotations

import logging
import sys

import structlog

from reactive_lob_sim.settings import LoggingSettings


def configure_logging(cfg: LoggingSettings) -> None:
    level = logging.getLevelNamesMapping().get(cfg.level.upper(), logging.INFO)
    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer() if cfg.json_logs else structlog.dev.ConsoleRenderer(colors=False)
    )
    structlog.configure(
        processors=[structlog.processors.add_log_level, structlog.processors.TimeStamper(fmt="iso"), renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(sys.stderr),
    )
