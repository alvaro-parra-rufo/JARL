"""Rollout batch reshaping helpers."""

from __future__ import annotations

import jax.numpy as jnp

__all__ = ["flatten_rollout_tensor"]


def flatten_rollout_tensor(
    tensor: jnp.ndarray,
    trailing_shape: tuple[int, ...],
) -> jnp.ndarray:
    """Flatten rollout time and parallel-env axes into a single batch axis.

    Args:
        tensor: Array with shape ``(nr_steps, nr_envs, *trailing_shape)``.
        trailing_shape: Trailing dimensions after the rollout axes (for example observation shape).

    Returns:
        Array with shape ``(nr_steps * nr_envs, *trailing_shape)``.
    """
    return tensor.reshape((-1, *trailing_shape))
