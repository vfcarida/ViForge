"""
ViForge Structured and Rich Logging Utilities.
"""

import json
import logging
import sys
from typing import Any, Dict, Optional
from rich.console import Console
from rich.logging import RichHandler

console = Console()


class JSONFormatter(logging.Formatter):
    """Format logs as structured JSON strings for CloudWatch / MLOps logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "experiment_id"):
            log_obj["experiment_id"] = record.experiment_id
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


def setup_logger(
    name: str = "viforge",
    level: int = logging.INFO,
    json_output: bool = False,
) -> logging.Logger:
    """Configure a named logger with Rich console output or structured JSON output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    if json_output:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
    else:
        handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
            markup=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = setup_logger()
