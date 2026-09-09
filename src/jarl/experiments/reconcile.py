"""Recovery helpers for stale experiment nodes."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ReconcileReport"]


@dataclass(frozen=True, slots=True)
class ReconcileReport:
    """Outcome of a stale-node reconciliation pass."""

    examined: tuple[str, ...]
    would_interrupt: tuple[str, ...]
    interrupted: tuple[str, ...]
