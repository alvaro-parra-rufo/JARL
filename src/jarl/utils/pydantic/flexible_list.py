"""Comma-tolerant list annotations for LLM / tool payloads."""

from __future__ import annotations

from typing import Annotated

from pydantic import BeforeValidator

__all__ = [
    "FlexibleList",
    "parse_flexible_list",
]


def parse_flexible_list(value: object) -> object:
    """Normalize list-like inputs before element-wise Pydantic validation.

    Accepted shapes:

    - ``None`` (for optional fields)
    - ``list`` / ``tuple`` (passed through as a list)
    - ``str`` split on commas (items must not contain commas)

    A blank string becomes ``[]``. Otherwise each segment is stripped and empty
    segments are preserved so callers can reject them if needed.
    """
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        if not value.strip():
            return []
        return [part.strip() for part in value.split(",")]
    msg = (
        "Expected a list or a comma-separated string "
        f"(items must not contain commas); got {type(value).__name__}: {value!r}."
    )
    raise ValueError(msg)


type FlexibleList[T] = Annotated[list[T], BeforeValidator(parse_flexible_list)]
"""``list[T]`` that also accepts comma-separated strings (e.g. ``FlexibleList[str]``)."""
