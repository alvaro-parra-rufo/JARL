"""Nested dictionary helpers."""

from __future__ import annotations

from typing import Any

__all__ = ["deep_merge_dicts", "flatten_dict", "unflatten_dict"]


def flatten_dict(
    d: dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> dict[str, Any]:
    """Flatten a nested dictionary into dot-separated keys.

    Args:
        d: Nested dictionary to flatten.
        parent_key: Prefix for keys (used in recursion).
        sep: Separator between key levels.

    Returns:
        Flat dictionary with composite keys like `a.b.c`.

    Example:
        ```python
        flatten_dict({"a": {"b": 1, "c": {"d": 2}}})
        # {"a.b": 1, "a.c.d": 2}
        ```
    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: dict[str, Any], sep: str = ".") -> dict[str, Any]:
    """Expand a flat dictionary with dot-separated keys into a nested dictionary.

    Inverse of `flatten_dict` for dict-only leaf values. Non-dotted keys and
    nested dict values are preserved; overlapping paths are deep-merged.

    Args:
        d: Flat or partially flat dictionary.
        sep: Separator between key levels.

    Returns:
        Nested dictionary.

    Example:
        ```python
        unflatten_dict({"a.b": 1, "a.c": 2, "flat": 3})
        # {"a": {"b": 1, "c": 2}, "flat": 3}
        ```
    """
    result: dict[str, Any] = {}
    for key, value in d.items():
        if sep in key:
            parts = key.split(sep)
            target = result
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            leaf = parts[-1]
            if leaf in target and isinstance(target[leaf], dict) and isinstance(value, dict):
                target[leaf] = deep_merge_dicts(target[leaf], value)
            else:
                target[leaf] = value
        elif isinstance(value, dict):
            if key in result and isinstance(result[key], dict):
                result[key] = deep_merge_dicts(result[key], value)
            else:
                result[key] = value
        else:
            result[key] = value
    return result


def deep_merge_dicts(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge `updates` into a copy of `base`.

    Args:
        base: Base nested dictionary.
        updates: Dictionary whose values override or merge into `base`.

    Returns:
        New dictionary with merged contents.
    """
    merged = dict(base)
    for key, value in updates.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged
