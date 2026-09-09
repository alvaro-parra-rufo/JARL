"""Sparse training overrides derived from a full run config snapshot."""

from __future__ import annotations

from typing import Any, Literal

from jarl.training.config import RLRunConfig
from jarl.utils import flatten_dict

OverridePolicy = Literal["retrain", "fork"]

_OVERRIDE_KEY_HINTS: dict[str, str] = {
    "algorithm.eval_frequency": "algorithm.evaluation_and_save_frequency",
    "algorithm.nr_envs": "environment.nr_envs",
    "algorithm.entropy_coefficient": "algorithm.entropy_coef",
    "env_id": "environment.env_id",
    "total_timesteps": "algorithm.total_timesteps",
    "eval_frequency": "algorithm.evaluation_and_save_frequency",
    "entropy_coefficient": "algorithm.entropy_coef",
    "entropy_coef": "algorithm.entropy_coef",
    "nr_envs": "environment.nr_envs",
    "nr_steps": "algorithm.nr_steps",
    "minibatch": "algorithm.minibatch_size",
    "learning_rate": "algorithm.learning_rate",
    "gae_lambda": "algorithm.gae_lambda",
    "clip_range": "algorithm.clip_range",
    "gamma": "algorithm.gamma",
}

__all__ = [
    "config_override_key_hint",
    "ensure_validated_config_overrides",
    "merge_validated_config_overrides",
    "mutable_overrides_from_config",
    "retrain_overrides_from_config",
    "validate_config_overrides",
    "validate_fork_overrides",
    "validate_retrain_overrides",
]


def config_override_key_hint(invalid_key: str) -> str | None:
    """Return the canonical dotted override path for a common invalid or shorthand key."""
    return _OVERRIDE_KEY_HINTS.get(invalid_key)


def mutable_overrides_from_config(config: RLRunConfig) -> dict[str, Any]:
    """Return mutable config paths for training or resuming an existing node."""
    flat = flatten_dict(config.model_dump(), sep=".")
    return {path: flat[path] for path in RLRunConfig.RETRAIN_MUTABLE_PATHS}


def retrain_overrides_from_config(config: RLRunConfig) -> dict[str, Any]:
    """Extract and validate sparse overrides for re-train or resume."""
    overrides = mutable_overrides_from_config(config)
    validate_config_overrides(overrides, policy="retrain")
    return overrides


def validate_config_overrides(
    overrides: dict[str, Any],
    *,
    policy: OverridePolicy,
) -> None:
    """Validate sparse override keys against the mutable path policy."""
    _validate_override_policy(overrides, policy=policy)


def validate_retrain_overrides(overrides: dict[str, Any]) -> None:
    """Validate overrides for re-train or resume on an existing node."""
    validate_config_overrides(overrides, policy="retrain")


def validate_fork_overrides(overrides: dict[str, Any]) -> None:
    """Validate overrides for fork or extend relative to a parent node."""
    validate_config_overrides(overrides, policy="fork")


def merge_validated_config_overrides(
    base_config: RLRunConfig,
    overrides: dict[str, Any] | None,
    *,
    policy: OverridePolicy,
) -> RLRunConfig | None:
    """Validate override keys and return ``base_config`` merged with ``overrides``.

    Rollout batch divisibility and other ``RLRunConfig`` validators run on the
    merged config, not on the sparse override dict alone.
    """
    if not overrides:
        return None
    validate_config_overrides(overrides, policy=policy)
    return base_config.apply_overrides(overrides)


def ensure_validated_config_overrides(
    base_config: RLRunConfig,
    overrides: dict[str, Any] | None,
    *,
    policy: OverridePolicy,
) -> None:
    """Validate sparse overrides against ``base_config`` without returning the merge.

    Use when callers still apply or persist the sparse ``overrides`` dict elsewhere
    (for example ``jarl.training.runner``) but need the same final-config checks as
    ``merge_validated_config_overrides``.
    """
    merge_validated_config_overrides(base_config, overrides, policy=policy)


def _allowed_override_paths(policy: OverridePolicy) -> frozenset[str]:
    if policy == "retrain":
        return RLRunConfig.RETRAIN_MUTABLE_PATHS
    return RLRunConfig.RETRAIN_MUTABLE_PATHS | RLRunConfig.FORK_EXTRA_MUTABLE_PATHS


def _validate_override_policy(overrides: dict[str, Any], *, policy: OverridePolicy) -> None:
    _reject_immutable_overrides(overrides)
    allowed = _allowed_override_paths(policy)
    extra = sorted(set(overrides) - allowed)
    if not extra:
        return
    if policy == "retrain":
        msg = f"Overrides fuera de RETRAIN_MUTABLE_PATHS: {', '.join(extra)}."
    else:
        msg = f"Overrides no permitidos en fork/extend: {', '.join(extra)}."
    hints = _override_key_hints(extra, policy=policy)
    if hints:
        msg = f"{msg} Sugerencias: {'; '.join(hints)}."
    raise ValueError(msg)


def _override_key_hints(keys: list[str], *, policy: OverridePolicy) -> list[str]:
    """Return operator-facing alias hints for common invalid override keys."""
    hints: list[str] = []
    for key in keys:
        canonical = _OVERRIDE_KEY_HINTS.get(key)
        if canonical is None:
            continue
        if canonical == "environment.nr_envs" and policy == "fork":
            hints.append(f"`{key}` → `{canonical}` (no permitido en fork/extend; solo `environment.env_id` extra)")
            continue
        hints.append(f"`{key}` → `{canonical}`")
    return hints


def _reject_immutable_overrides(overrides: dict[str, Any]) -> None:
    blocked = sorted(set(overrides) & RLRunConfig.immutable_paths())
    if blocked:
        msg = f"Config diff touches immutable paths: {', '.join(blocked)}"
        raise ValueError(msg)
