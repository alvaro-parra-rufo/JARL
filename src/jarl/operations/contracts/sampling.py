"""Deterministic downsampling for operation read payloads."""

from __future__ import annotations

__all__ = ["downsample_indices"]


def downsample_indices(count: int, max_points: int) -> list[int]:
    """Return uniformly spaced indices for deterministic series downsampling.

    Args:
        count: Number of source points.
        max_points: Maximum number of indices to return.

    Returns:
        Sorted indices in ``[0, count)`` with length ``min(count, max_points)``.
    """
    if count <= 0 or max_points <= 0:
        return []
    if count <= max_points:
        return list(range(count))
    if max_points == 1:
        return [0]
    last = count - 1
    return [round(index * last / (max_points - 1)) for index in range(max_points)]
