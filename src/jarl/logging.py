"""Logging configuration for the jarl package."""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any, TextIO

LOG_FORMAT = "[%(asctime)s] [%(filename)s:%(lineno)d] %(levelname)s - %(message)s"
"""Standard log format used across console and file handlers."""

LOG_DATE_FORMAT = "%m-%d %H:%M:%S"
"""Date format for log timestamps."""


def get_console(name: str = "jarl") -> logging.Logger:
    """Get the console logger."""
    console = logging.getLogger(name)
    console.setLevel(logging.INFO)
    console.propagate = False
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setLevel(logging.INFO)
    consoleHandler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    memoryHandler = logging.handlers.MemoryHandler(100, logging.ERROR, consoleHandler)
    console.addHandler(memoryHandler)

    def info(msg: Any, *args: Any, flush: bool = True, **kwargs: Any) -> None:
        """Log an info message with flush option."""
        if console.isEnabledFor(logging.INFO):
            console._log(logging.INFO, msg, args, stacklevel=2, **kwargs)
        if flush:
            console.handlers[0].flush()

    console.info = info
    return console


def redirect_console_to_stderr() -> None:
    """Point the package console logger at stderr.

    Stdio hosts (MCP) keep stdout for JSON-RPC; training logs must not share it.
    """
    for handler in logging.getLogger("jarl").handlers:
        _redirect_handler_stream(handler, sys.stderr)


def _redirect_handler_stream(handler: logging.Handler, stream: TextIO) -> None:
    """Retarget a stream handler, including a `MemoryHandler` target."""
    if isinstance(handler, logging.handlers.MemoryHandler) and handler.target is not None:
        _redirect_handler_stream(handler.target, stream)
        return
    if isinstance(handler, logging.StreamHandler):
        handler.acquire()
        try:
            handler.stream = stream
        finally:
            handler.release()


def attach_file_handler(logger_name: str, path: str | Path) -> logging.FileHandler:
    """Create and attach a file handler to the named logger.

    Args:
        logger_name: Name of the logger to attach to.
        path: File path for the log output.

    Returns:
        The created `FileHandler`.
    """
    handler = logging.FileHandler(path)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    logging.getLogger(logger_name).addHandler(handler)
    return handler


console = get_console("jarl")
"""Package-level console logger."""
