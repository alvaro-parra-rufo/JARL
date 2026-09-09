"""Validation helpers for explicit metric key requests."""

from __future__ import annotations

from jarl.agents.ppo.metric_schema import suggest_ppo_metric_names

__all__ = ["validate_requested_metric_keys"]


def validate_requested_metric_keys(
    available: tuple[str, ...],
    requested: list[str],
) -> None:
    """Reject unknown metric keys using logged names and the PPO metric catalog.

    Args:
        available: Metric names present in the node ``metrics.jsonl`` file.
        requested: Explicit metric keys from the caller.

    Raises:
        ValueError: When any requested key is absent from ``available``.
    """
    if not requested:
        return
    available_set = set(available)
    unknown = sorted({key for key in requested if key not in available_set})
    if not unknown:
        return
    msg = f"Unknown metric keys: {', '.join(unknown)}."
    if not available:
        msg = f"{msg} No metrics are logged for this node yet."
    else:
        msg = f"{msg} Available keys: {', '.join(available)}."
    hints = _metric_key_hints(unknown, available_set)
    if hints:
        msg = f"{msg} Suggestions: {'; '.join(hints)}."
    raise ValueError(msg)


def _metric_key_hints(unknown: list[str], available: set[str]) -> list[str]:
    """Return operator-facing hints derived from ``suggest_ppo_metric_names``."""
    hints: list[str] = []
    for key in unknown:
        suggestions = suggest_ppo_metric_names(key)
        if not suggestions:
            continue
        if len(suggestions) == 1:
            canonical = suggestions[0]
            suffix = "" if canonical in available else " (not logged on this node)"
            hints.append(f"`{key}` → `{canonical}`{suffix}")
            continue
        options = ", ".join(f"`{name}`" for name in suggestions)
        hints.append(f"`{key}` → one of {options}")
    return hints
