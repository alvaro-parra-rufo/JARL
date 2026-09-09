"""Identifier helpers."""

from __future__ import annotations

import uuid

__all__ = ["short_uuid"]


def short_uuid(length: int = 8) -> str:
    """Generate a short hex UUID string.

    Args:
        length: Number of hex characters to return (max 32).

    Returns:
        Lowercase hex string of the requested length.
    """
    return uuid.uuid4().hex[:length]
