"""JIT-safe metric schemas and packing helpers for PPO full-JAX training."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

__all__ = [
    "PPO_EVAL_METRIC_SCHEMA",
    "PPO_LOGGED_METRIC_NAMES",
    "PPO_TRAIN_METRIC_SCHEMA",
    "MetricSchema",
    "assemble_metric_bundle",
    "pack_device_metrics",
    "suggest_ppo_metric_names",
]

_TRAIN_DEVICE_METRIC_NAMES: tuple[str, ...] = (
    "rollout/episode_return",
    "rollout/episode_length",
    "rollout/value_mean",
    "rollout/return_mean",
    "rollout/advantage_std",
    "loss/policy_gradient_loss",
    "loss/critic_loss",
    "loss/entropy_loss",
    "policy_ratio/approx_kl",
    "policy_ratio/clip_fraction",
    "policy_ratio/max",
    "gradients/policy_grad_norm",
    "gradients/critic_grad_norm",
    "lr/learning_rate",
    "v_value/explained_variance",
    "policy/std_dev",
)

_TRAIN_HOST_METRIC_NAMES: tuple[str, ...] = (
    "time/sps",
    "steps/nr_env_steps",
    "steps/nr_rollout_updates",
    "steps/nr_optimizer_updates",
    "steps/nr_updates",
)

_EVAL_DEVICE_METRIC_NAMES: tuple[str, ...] = (
    "eval/episode_return",
    "eval/episode_length",
)


@dataclass(frozen=True, slots=True)
class MetricSchema:
    """Fixed-order metric names for JIT device payloads and host extensions.

    ``device_names`` are packed inside compiled loops. ``host_names`` are
    appended on the host before logging (for example throughput and step
    counters).
    """

    device_names: tuple[str, ...]
    host_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject duplicate or overlapping metric names."""
        _validate_unique_names(self.device_names, "device_names")
        _validate_unique_names(self.host_names, "host_names")
        overlap = set(self.device_names) & set(self.host_names)
        if overlap:
            msg = f"Metric names cannot appear in both device and host schemas: {sorted(overlap)}"
            raise ValueError(msg)

    @property
    def all_names(self) -> tuple[str, ...]:
        """Device names followed by host names in logging order."""
        return self.device_names + self.host_names


def pack_device_metrics(
    values: Mapping[str, jnp.ndarray | float],
    schema: MetricSchema,
) -> jnp.ndarray:
    """Pack device-side metrics into a fixed-order JAX array.

    Args:
        values: Metric values keyed by ``schema.device_names``.
        schema: Schema defining the packed order.

    Returns:
        ``float32`` vector with one entry per ``schema.device_names`` entry.
    """
    missing = set(schema.device_names) - set(values)
    if missing:
        msg = f"Missing device metric values for: {sorted(missing)}"
        raise KeyError(msg)
    extra = set(values) - set(schema.device_names)
    if extra:
        msg = f"Unexpected device metric values for: {sorted(extra)}"
        raise KeyError(msg)
    return jnp.array([values[name] for name in schema.device_names], dtype=jnp.float32)


def assemble_metric_bundle(
    device_body: np.ndarray,
    host_values: Mapping[str, float],
    schema: MetricSchema,
) -> np.ndarray:
    """Merge a device payload with host metrics following ``schema.all_names``.

    Args:
        device_body: Packed device metrics in ``schema.device_names`` order.
        host_values: Host metrics keyed by ``schema.host_names``.
        schema: Schema defining the final logging order.

    Returns:
        ``float64`` vector with one entry per ``schema.all_names`` entry.
    """
    array = np.asarray(device_body, dtype=np.float64).reshape(-1)
    if array.shape[0] != len(schema.device_names):
        msg = f"Expected {len(schema.device_names)} device metric values, got {array.shape[0]}."
        raise ValueError(msg)
    missing_host = set(schema.host_names) - set(host_values)
    if missing_host:
        msg = f"Missing host metric values for: {sorted(missing_host)}"
        raise KeyError(msg)
    extra_host = set(host_values) - set(schema.host_names)
    if extra_host:
        msg = f"Unexpected host metric values for: {sorted(extra_host)}"
        raise KeyError(msg)
    device_by_name = dict(zip(schema.device_names, array, strict=True))
    merged = {**device_by_name, **{name: float(host_values[name]) for name in schema.host_names}}
    return np.array([merged[name] for name in schema.all_names], dtype=np.float64)


def _validate_unique_names(names: tuple[str, ...], field_name: str) -> None:
    if len(names) == len(set(names)):
        return
    duplicates = sorted({name for name in names if names.count(name) > 1})
    msg = f"Duplicate metric names in {field_name}: {duplicates}"
    raise ValueError(msg)


PPO_TRAIN_METRIC_SCHEMA = MetricSchema(
    device_names=_TRAIN_DEVICE_METRIC_NAMES,
    host_names=_TRAIN_HOST_METRIC_NAMES,
)
"""Shared train-phase schema for PPO and PPO-GRU full-JAX trainers."""

PPO_EVAL_METRIC_SCHEMA = MetricSchema(device_names=_EVAL_DEVICE_METRIC_NAMES)
"""Eval-phase schema for PPO and PPO-GRU full-JAX trainers."""

PPO_LOGGED_METRIC_NAMES: frozenset[str] = frozenset(
    (*PPO_TRAIN_METRIC_SCHEMA.all_names, *PPO_EVAL_METRIC_SCHEMA.all_names),
)
"""Canonical metric names written by PPO full-JAX trainers."""


def suggest_ppo_metric_names(invalid: str) -> tuple[str, ...]:
    """Return catalog metric names that may match a mistyped ``metrics.jsonl`` key."""
    if invalid in PPO_LOGGED_METRIC_NAMES:
        return (invalid,)
    rewritten = invalid.replace("charts/", "rollout/", 1)
    if rewritten != invalid and rewritten in PPO_LOGGED_METRIC_NAMES:
        return (rewritten,)
    tail = invalid.rsplit("/", maxsplit=1)[-1]
    matches = sorted({name for name in PPO_LOGGED_METRIC_NAMES if name.endswith(tail) or tail in name})
    return tuple(matches)
