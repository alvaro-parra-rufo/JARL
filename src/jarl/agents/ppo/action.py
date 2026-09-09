"""Action sampling and post-processing for PPO full-JAX."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

__all__ = [
    "ProcessedActionFn",
    "continuous_log_prob",
    "discrete_log_prob",
    "make_processed_action_fn",
    "sample_continuous_action",
    "sample_discrete_action",
]

ProcessedActionFn = Callable[[jax.Array], jax.Array]
"""Callable that maps raw policy actions to environment actions."""


def sample_discrete_action(
    key: jax.Array,
    logits: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Sample discrete actions and their log-probabilities.

    Args:
        key: PRNG key.
        logits: Unnormalized action logits with shape ``(..., n_actions)``.

    Returns:
        Tuple of sampled actions and per-batch log-probabilities.
    """
    action = jax.random.categorical(key, logits, axis=-1)
    log_prob = jnp.take_along_axis(jax.nn.log_softmax(logits), action[:, None], axis=1).squeeze(1)
    return action, log_prob


def discrete_log_prob(logits: jax.Array, action: jax.Array) -> jax.Array:
    """Compute log-probability of discrete actions under a categorical policy."""
    log_probs = jax.nn.log_softmax(logits)
    return log_probs[jnp.arange(action.shape[0]), action.astype(jnp.int32)]


def sample_continuous_action(
    key: jax.Array,
    action_mean: jax.Array,
    action_logstd: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Sample continuous actions from a diagonal Gaussian policy.

    Args:
        key: PRNG key.
        action_mean: Mean with shape ``(batch, action_dim)``.
        action_logstd: Log standard deviation broadcastable to the mean.

    Returns:
        Tuple of sampled actions and per-batch log-probabilities.
    """
    action_std = jnp.exp(action_logstd)
    action = action_mean + action_std * jax.random.normal(key, shape=action_mean.shape)
    log_prob = continuous_log_prob(action, action_mean, action_logstd)
    return action, log_prob


def continuous_log_prob(
    action: jax.Array,
    action_mean: jax.Array,
    action_logstd: jax.Array,
) -> jax.Array:
    """Compute per-batch log-probability under a diagonal Gaussian policy."""
    action_std = jnp.exp(action_logstd)
    per_dim = -0.5 * ((action - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd
    return per_dim.sum(axis=-1)


def make_processed_action_fn(
    *,
    action_clipping_and_rescaling: bool,
    action_low: jax.Array,
    action_high: jax.Array,
) -> ProcessedActionFn:
    """Build the environment action post-processor for continuous control.

    Args:
        action_clipping_and_rescaling: When ``True``, clip to ``[-1, 1]`` and affine-map to bounds.
        action_low: Lower bound of the environment action space.
        action_high: Upper bound of the environment action space.

    Returns:
        JIT-compiled action post-processor.
    """
    if action_clipping_and_rescaling:

        def clip_and_scale(action: jax.Array) -> jax.Array:
            clipped = jnp.clip(action, -1.0, 1.0)
            return action_low + (0.5 * (clipped + 1.0) * (action_high - action_low))

        return jax.jit(clip_and_scale)
    return jax.jit(lambda action: action)
