"""Minibatch construction and advantage normalization."""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = [
    "make_minibatch_indices",
    "normalize_advantages",
]


def normalize_advantages(advantages: jax.Array) -> jax.Array:
    """Standardize advantages for a minibatch."""
    return (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)


def make_minibatch_indices(
    key: jax.Array,
    *,
    batch_size: int,
    nr_epochs: int,
    nr_minibatches: int,
    minibatch_size: int,
) -> jax.Array:
    """Build shuffled minibatch index rows for PPO optimization.

    Build shuffled minibatch indices for PPO full-JAX optimization.

    Args:
        key: PRNG key.
        batch_size: Total rollout batch size ``nr_envs * nr_steps``.
        nr_epochs: PPO epochs per rollout.
        nr_minibatches: Minibatches per epoch.
        minibatch_size: Environment steps per minibatch.

    Returns:
        Integer array with shape ``(nr_epochs * nr_minibatches, minibatch_size)``.
    """
    indices = jnp.tile(jnp.arange(batch_size), (nr_epochs, 1))
    shuffled = jax.random.permutation(key, indices, axis=1, independent=True)
    return shuffled.reshape((nr_epochs * nr_minibatches, minibatch_size))
