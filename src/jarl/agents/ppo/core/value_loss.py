"""Clipped PPO value loss shared by feedforward and GRU trainers."""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = [
    "clipped_value_loss",
    "pure_value_loss",
]


def pure_value_loss(
    *,
    value_prediction: jax.Array,
    return_target: jax.Array,
) -> jax.Array:
    """Unclipped per-sample value loss (legacy MSE)."""
    return_target = jax.lax.stop_gradient(return_target)
    return 0.5 * (value_prediction - return_target) ** 2


def clipped_value_loss(
    *,
    value_prediction: jax.Array,
    old_value: jax.Array,
    return_target: jax.Array,
    clip_range: float,
) -> jax.Array:
    """Per-sample clipped PPO value loss.

    Compares squared error against ``return_target`` for the raw prediction and for
    a prediction clipped to ``old_value ± clip_range``, then returns half the
    element-wise maximum (standard PPO value clipping).
    """
    old_value = jax.lax.stop_gradient(old_value)
    return_target = jax.lax.stop_gradient(return_target)
    value_clipped = old_value + jnp.clip(
        value_prediction - old_value,
        -clip_range,
        clip_range,
    )
    unclipped_loss = (value_prediction - return_target) ** 2
    clipped_loss = (value_clipped - return_target) ** 2
    return 0.5 * jnp.maximum(unclipped_loss, clipped_loss)
