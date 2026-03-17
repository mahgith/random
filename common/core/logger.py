"""
STRUCTURED LOGGER
=================
Configures structlog for two modes:
  - Local dev:   coloured ConsoleRenderer (human-readable)
  - Vertex AI:   JSONRenderer with GCP severity field (Cloud Logging compatible)

Cloud Logging picks up the "severity" field automatically and indexes it,
so you can filter by level in GCP without extra log sinks.

Usage (inside any module or KFP component function):
    from common.core.logger import get_logger
    logger = get_logger("my-component")
    logger.info("Step complete", rows=1234, direction="inbound")
"""

import logging
import structlog

from common.core.settings import get_settings


def _add_gcp_severity(logger, method_name: str, event_dict: dict) -> dict:
    """Maps structlog level names to GCP Cloud Logging severity strings."""
    level = event_dict.get("level")
    if level:
        event_dict["severity"] = level.upper()
    return event_dict


def setup_logging() -> None:
    """Initialises the structlog pipeline. Safe to call multiple times."""
    settings = get_settings()

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.LOG_JSON_FORMAT:
        # Vertex AI / Production — output one JSON object per log line
        renderer = structlog.processors.JSONRenderer()
        processors = shared_processors + [_add_gcp_severity, renderer]
    else:
        # Local dev — coloured, readable output
        renderer = structlog.dev.ConsoleRenderer(colors=True, pad_event=False)
        processors = shared_processors + [renderer]

    structlog.configure(
        processors=processors,
        logger_factory=structlog.PrintLoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.LOG_LEVEL)
        ),
        cache_logger_on_first_use=True,
    )


_configured = False


def get_logger(name: str):
    """
    Returns a named structlog logger, ensuring setup runs exactly once.

    Args:
        name: Logical name for the logger (shown as a field in the log output).

    Returns:
        A bound structlog logger instance.
    """
    global _configured
    if not _configured:
        setup_logging()
        _configured = True
    return structlog.get_logger(name)
