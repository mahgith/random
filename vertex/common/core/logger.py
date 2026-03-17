import logging
import structlog
from common.core.settings import get_settings

def add_gcp_severity(logger, method_name, event_dict):
    """
    Maps structlog levels to 'severity' for Google Cloud Logging compatibility.
    """
    level = event_dict.get("level")
    if level:
        event_dict["severity"] = level.upper()
    return event_dict

def setup_logging():
    """
    Initializes the structured logging pipeline.
    """
    settings = get_settings()
    
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.LOG_JSON_FORMAT:
        # Vertex AI / Production mode
        processors.append(structlog.processors.TimeStamper(fmt="iso"))
        processors.append(add_gcp_severity)
        renderer = structlog.processors.JSONRenderer()
    else:
        # Local development mode
        processors.append(structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"))
        renderer = structlog.dev.ConsoleRenderer(colors=True, pad_event=False)

    structlog.configure(
        processors=processors + [renderer],
        logger_factory=structlog.PrintLoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.LOG_LEVEL)
        ),
        cache_logger_on_first_use=True,
    )

def get_logger(name: str):
    """
    Returns a named logger. Ensures setup is only called once.
    """
    if not structlog.is_configured():
        setup_logging()
    return structlog.get_logger(name)