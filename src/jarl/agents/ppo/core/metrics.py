"""Post-optimization metric helpers for PPO full-JAX."""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = [
    "aggregate_optimization_metrics",
    "compute_explained_variance",
    "compute_policy_std_dev",
    "compute_rollout_target_stats",
]


def compute_explained_variance(
    returns: jax.Array,
    values: jax.Array,
) -> jax.Array:
    """Compute value-function explained variance for a rollout batch."""
    return 1.0 - jnp.var(returns - values) / (jnp.var(returns) + 1e-8)


def compute_rollout_target_stats(
    values: jax.Array,
    returns: jax.Array,
    advantages: jax.Array,
) -> dict[str, jax.Array]:
    """Summarize rollout value targets before minibatch normalization."""
    return {
        "rollout/value_mean": jnp.mean(values),
        "rollout/return_mean": jnp.mean(returns),
        "rollout/advantage_std": jnp.std(advantages),
    }


def aggregate_optimization_metrics(
    metrics: dict[str, jax.Array],
) -> dict[str, jax.Array]:
    """Average optimization metrics while preserving max policy ratio across minibatches."""
    ratio_max = jnp.max(metrics["policy_ratio/max"])
    averaged = {key: jnp.mean(value) for key, value in metrics.items() if key != "policy_ratio/max"}
    averaged["policy_ratio/max"] = ratio_max
    return averaged


def compute_policy_std_dev(
    *,
    is_discrete: bool,
    policy_logstd: jax.Array | None = None,
) -> jax.Array:
    """Return policy standard deviation metric logged during training."""
    if is_discrete:
        return jnp.array(0.0, dtype=jnp.float32)
    if policy_logstd is None:
        msg = "policy_logstd is required for continuous action spaces."
        raise ValueError(msg)
    return jnp.mean(jnp.exp(policy_logstd))
