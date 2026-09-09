"""Compact dictionary projection for operation responses."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

__all__ = ["compact_value", "dataclass_to_compact_dict"]


def compact_value(value: object) -> object:
    """Recursively convert a value into a JSON-friendly structure.

    Args:
        value: Arbitrary object to serialize.

    Returns:
        JSON-friendly value with dataclasses expanded and ``Path`` values
        converted to POSIX strings.
    """
    if value is None:
        return None
    if isinstance(value, Path):
        serialized: object = value.as_posix()
    elif isinstance(value, Enum):
        serialized = value.value
    elif is_dataclass(value):
        serialized = {
            field.name: compact_value(getattr(value, field.name))
            for field in fields(value)
            if getattr(value, field.name) is not None
        }
    elif isinstance(value, dict):
        serialized = {key: compact_value(item) for key, item in value.items() if item is not None}
    elif isinstance(value, (list, tuple)):
        serialized = [compact_value(item) for item in value]
    else:
        serialized = value
    return serialized


def dataclass_to_compact_dict(
    instance: object,
    *,
    include: frozenset[str] | None = None,
) -> dict[str, object]:
    """Serialize a dataclass to a compact dictionary.

    Args:
        instance: Dataclass instance to serialize.
        include: Optional subset of field names to keep.

    Returns:
        JSON-friendly dictionary with ``None`` values removed.
    """
    if not is_dataclass(instance):
        msg = "dataclass_to_compact_dict expects a dataclass instance."
        raise TypeError(msg)

    compact: dict[str, object] = {}
    for field in fields(instance):
        if include is not None and field.name not in include:
            continue
        raw_value: Any = getattr(instance, field.name)
        if raw_value is None:
            continue
        compact[field.name] = compact_value(raw_value)
    return compact
