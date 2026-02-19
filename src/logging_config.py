"""Minimal structured logging setup. Call setup_logging() once at entry point."""
import logging
import sys

import structlog


def setup_logging(level: str = "INFO", json_output: bool = False):
    """Configure structlog. Call once in scripts/main entry points."""
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.format_exc_info,
    ]

    if json_output:
        shared_processors.append(structlog.processors.JSONRenderer())
    else:
        shared_processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=shared_processors,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )


def get_logger(name: str):
    """Get a structlog logger bound with module name."""
    return structlog.get_logger(name)
