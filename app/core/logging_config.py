import logging
import os
import sys
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


def setup_logging():
    """Configure application logging with safe fallback.

    If the file handler cannot be configured (e.g. permission denied when the
    container runs as non-root and a host directory is mounted with restrictive
    ownership), the configuration gracefully degrades to console-only logging
    instead of aborting application startup.
    """
    settings = get_settings()
    log_level = "DEBUG" if settings.debug else "INFO"

    log_dir = os.path.join("logs")
    log_file_path = os.path.join(log_dir, "app.log")

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
            "json": {
                "()": JsonFormatter,
                "format": "%(timestamp)s %(level)s %(name)s %(message)s %(lineno)d",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "stream": sys.stdout,
            },
            "file": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "formatter": "json",
                "filename": log_file_path,
                "when": "midnight",
                "interval": 1,
                "backupCount": 30,
                "encoding": "utf-8",
            },
        },
        "loggers": {
            "uvicorn": {
                "handlers": ["console", "file"],
                "level": log_level,
                "propagate": False,
            },
            "fastapi": {
                "handlers": ["console", "file"],
                "level": log_level,
                "propagate": False,
            },
            "app": {
                "handlers": ["console", "file"],
                "level": log_level,
                "propagate": False,
            },
        },
        "root": {
            "handlers": ["console", "file"],
            "level": log_level,
        },
    }

    try:
        dictConfig(logging_config)
    except Exception as e:
        # Remove file handler and retry with console-only.
        for logger_name in list(logging_config.get("loggers", {}).keys()):
            handlers = logging_config["loggers"][logger_name].get("handlers", [])
            logging_config["loggers"][logger_name]["handlers"] = [
                h for h in handlers if h != "file"
            ]
        # Root logger
        logging_config["root"]["handlers"] = [
            h for h in logging_config["root"]["handlers"] if h != "file"
        ]
        logging_config["handlers"].pop("file", None)
        dictConfig(logging_config)
        logging.getLogger(__name__).warning(
            "File logging disabled; falling back to console only (%s)", e
        )
