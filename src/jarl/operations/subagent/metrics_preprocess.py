"""Objective metric feature extraction for agentic metrics analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "METRIC_KEY_ALIASES",
    "PERCENTILE_LEVELS",
    "MetricsFeaturePayload",
    "alias_metric_key",
    "build_metrics_features",
]

METRIC_KEY_ALIASES: dict[str, str] = {
    "eval/episode_return": "eval_return",
    "eval/episode_length": "eval_length",
    "rollout/episode_return": "rollout_return",
    "rollout/episode_length": "rollout_length",
    "loss/policy_gradient_loss": "loss_policy",
    "loss/critic_loss": "loss_critic",
    "lr/learning_rate": "learning_rate",
}
"""Stable short aliases for common metric keys in feature ids."""

PERCENTILE_LEVELS: tuple[int, ...] = (25, 50, 75, 90)
"""Descriptive percentiles emitted for each series and window (p25/p50/p75/p90)."""


@dataclass(frozen=True, slots=True)
class MetricsFeaturePayload:
    """Objective feature map produced before LLM interpretation."""

    node_id: str
    features: dict[str, float | int]
    feature_ids: tuple[str, ...]
    focus: str = ""


def alias_metric_key(metric_key: str) -> str:
    """Return a stable short alias for ``metric_key`` used in feature ids."""
    if metric_key in METRIC_KEY_ALIASES:
        return METRIC_KEY_ALIASES[metric_key]
    return metric_key.replace("/", "_").replace(".", "_")


def build_metrics_features(
    node_id: str,
    series: dict[str, list[tuple[int, float]]],
    *,
    focus: str = "",
) -> MetricsFeaturePayload:
    """Build objective features with stable ids from downsampled series.

    Computes only descriptive statistics and windowed aggregates. Does not emit
    qualitative conclusions such as trends or spike flags.
    """
    features: dict[str, float | int] = {}
    for metric_key, points in series.items():
        alias = alias_metric_key(metric_key)
        features[f"{alias}.n_points"] = len(points)
        if not points:
            continue
        values = [value for _, value in points]
        steps = [step for step, _ in points]
        features[f"{alias}.first.step"] = steps[0]
        features[f"{alias}.first.value"] = values[0]
        features[f"{alias}.last.step"] = steps[-1]
        features[f"{alias}.last.value"] = values[-1]
        features[f"{alias}.min"] = min(values)
        features[f"{alias}.max"] = max(values)
        features[f"{alias}.mean"] = _mean(values)
        features[f"{alias}.std"] = _std(values)
        _add_percentiles(features, alias, values)
        features[f"{alias}.delta_abs"] = values[-1] - values[0]
        if values[0] != 0:
            features[f"{alias}.delta_rel"] = (values[-1] - values[0]) / abs(values[0])
        mid_index = len(values) // 2
        features[f"{alias}.mid.step"] = steps[mid_index]
        features[f"{alias}.mid.value"] = values[mid_index]
        for window_name, window_values in _window_slices(values).items():
            if not window_values:
                continue
            prefix = f"{alias}.window_{window_name}"
            features[f"{prefix}.mean"] = _mean(window_values)
            features[f"{prefix}.std"] = _std(window_values)
            features[f"{prefix}.min"] = min(window_values)
            features[f"{prefix}.max"] = max(window_values)
            _add_percentiles(features, prefix, window_values)
    feature_ids = tuple(sorted(features))
    return MetricsFeaturePayload(
        node_id=node_id,
        features=features,
        feature_ids=feature_ids,
        focus=focus,
    )


_MIN_WINDOW_POINTS = 3
"""Minimum points before splitting into three non-degenerate windows."""

_MIN_STD_POINTS = 2
"""Minimum points required to compute a non-zero standard deviation."""

_PERCENTILE_MAX_LEVEL = 100
"""Upper inclusive bound for percentile levels."""


def _window_slices(values: list[float]) -> dict[str, list[float]]:
    n = len(values)
    if n == 0:
        return {"initial": [], "middle": [], "final": []}
    if n < _MIN_WINDOW_POINTS:
        return {"initial": values[:1], "middle": values[:], "final": values[-1:]}
    third = max(1, n // 3)
    return {
        "initial": values[:third],
        "middle": values[third : n - third],
        "final": values[n - third :],
    }


def _add_percentiles(
    features: dict[str, float | int],
    prefix: str,
    values: list[float],
) -> None:
    """Attach ``p25``/``p50``/``p75``/``p90`` under ``prefix``."""
    for level in PERCENTILE_LEVELS:
        features[f"{prefix}.p{level}"] = _percentile(values, level)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) < _MIN_STD_POINTS:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _percentile(values: list[float], level: int) -> float:
    """Return the linear-interpolation percentile of ``values`` at ``level``.

    ``level`` is in ``[0, 100]``. Uses the common ``index = p * (n - 1)`` rule.
    """
    if not values:
        msg = "Cannot compute percentile of an empty series."
        raise ValueError(msg)
    if level < 0 or level > _PERCENTILE_MAX_LEVEL:
        msg = f"Percentile level must be in [0, {_PERCENTILE_MAX_LEVEL}], got {level}."
        raise ValueError(msg)
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (level / float(_PERCENTILE_MAX_LEVEL)) * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
