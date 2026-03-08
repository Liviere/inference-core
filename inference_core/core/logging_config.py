import logging
import logging.handlers
import os
import sys
from datetime import datetime
from logging.config import dictConfig

from pythonjsonlogger.json import JsonFormatter as BaseJsonFormatter

from .config import get_settings


class JsonFormatter(BaseJsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super(JsonFormatter, self).add_fields(log_record, record, message_dict)
        if not log_record.get("timestamp"):
            log_record["timestamp"] = record.created
        if log_record.get("level"):
            log_record["level"] = log_record["level"].upper()
        else:
            log_record["level"] = record.levelname
        # Prepend a human-readable date field at the very start of the record.
        existing = dict(log_record)
        log_record.clear()
        log_record["date"] = datetime.fromtimestamp(record.created).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
        log_record.update(existing)


def _build_file_handler(
    log_file_path: str, log_level: str
) -> logging.handlers.TimedRotatingFileHandler:
    """Create a pre-configured TimedRotatingFileHandler writing JSON records."""
    handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_file_path,
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    handler.setFormatter(
        JsonFormatter(fmt="%(timestamp)s %(level)s %(name)s %(message)s %(lineno)d")
    )
    handler.setLevel(log_level)
    return handler


def shutdown_logging() -> None:
    """Flush and close every FileHandler attached to the root logger and all named loggers.

    Should be called at process shutdown (FastAPI lifespan end, Celery worker
    shutdown) to ensure buffered records are flushed and OS file descriptors
    are released cleanly.
    """
    loggers_to_flush: list[logging.Logger] = [logging.root]
    for name in list(logging.root.manager.loggerDict):
        lgr = logging.getLogger(name)
        if lgr:
            loggers_to_flush.append(lgr)

    for lgr in loggers_to_flush:
        for handler in list(lgr.handlers):
            if isinstance(handler, logging.FileHandler):
                try:
                    handler.flush()
                    handler.close()
                    lgr.removeHandler(handler)
                except Exception:
                    pass


def setup_logging(service: str = "app") -> None:
    """Configure application logging with safe fallback.

    Attaches a console handler (human-readable, uvicorn-styled) and a
    rotating JSON file handler writing to ``logs/<service>.log``.

    If the file handler cannot be configured (e.g. permission denied when the
    container runs as non-root and a host directory is mounted with restrictive
    ownership), the configuration gracefully degrades to console-only logging
    instead of aborting application startup.

    Args:
        service: Logical service name used as the log file stem.
                 Use ``"api"`` for the FastAPI process and ``"celery"`` for
                 Celery workers so that each service writes to a dedicated file.
    """
    settings = get_settings()
    log_level = "DEBUG" if settings.debug else "INFO"

    log_dir = os.path.join("logs")
    log_file_path = os.path.join(log_dir, f"{service}.log")

    # Try to create the logs directory if it doesn't exist.
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception as e:  # pragma: no cover - extremely rare
        # Fall back silently; we'll still attempt console logging.
        print(
            f"[logging] Unable to create log directory '{log_dir}': {e}",
            file=sys.stderr,
        )

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(asctime)s | %(name)s:%(lineno)d | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "stream": sys.stdout,
            },
        },
        "loggers": {
            "uvicorn": {
                "handlers": ["console"],
                "level": log_level,
                "propagate": False,
            },
            "fastapi": {
                "handlers": ["console"],
                "level": log_level,
                "propagate": False,
            },
            "app": {
                "handlers": ["console"],
                "level": log_level,
                "propagate": False,
            },
        },
        "root": {
            "handlers": ["console"],
            "level": log_level,
        },
    }

    dictConfig(logging_config)

    # Attach the rotating file handler separately so that a failure (e.g.
    # permission denied) degrades gracefully to console-only without touching
    # the already-applied dictConfig.
    try:
        file_handler = _build_file_handler(log_file_path, log_level)
        for logger_name in ("uvicorn", "fastapi", "app"):
            logging.getLogger(logger_name).addHandler(file_handler)
        logging.root.addHandler(file_handler)
    except Exception as e:
        logging.getLogger(__name__).warning(
            "File logging disabled; falling back to console only (%s)", e
        )
