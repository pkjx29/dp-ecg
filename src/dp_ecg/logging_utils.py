"""Structured logging: human-readable lines on stderr, JSON lines in a file."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"time": self.formatTime(record), "level": record.levelname,
                   "logger": record.name, "message": record.getMessage()}
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def get_logger(name: str = "dp_ecg", log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        logger.addHandler(console)
        logger.propagate = False
    if log_file is not None and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(_JsonFormatter())
        logger.addHandler(fh)
    return logger
