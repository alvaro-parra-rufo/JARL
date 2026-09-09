"""Filesystem I/O helpers."""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["write_text_atomic"]


def write_text_atomic(path: str | Path, content: str) -> Path:
    """Write text to a file atomically via a temporary sibling file.

    Args:
        path: Destination file path.
        content: Text content to persist.

    Returns:
        The destination path written to.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, destination)
    return destination
