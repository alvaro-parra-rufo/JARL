"""Environment-index minibatch construction for PPO-GRU."""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = [
    "make_gru_minibatch_env_indices",
]


def make_gru_minibatch_env_indices(
    key: jax.Array,
    *,
    nr_envs: int,
    nr_epochs: int,
    nr_minibatches: int,
    nr_minibatch_envs: int,
) -> jax.Array:
    """Build shuffled environment-index rows for recurrent PPO optimization.

    Build shuffled environment indices for PPO-GRU full-JAX optimization.

    Args:
        key: PRNG key.
        nr_envs: Number of parallel training environments.
        nr_epochs: PPO epochs per rollout.
        nr_minibatches: Minibatches per epoch.
        nr_minibatch_envs: Environments per minibatch.

    Returns:
        Integer array with shape ``(nr_epochs * nr_minibatches, nr_minibatch_envs)``.
    """
    env_indices = jnp.tile(jnp.arange(nr_envs), (nr_epochs, 1))
    shuffled = jax.random.permutation(key, env_indices, axis=1, independent=True)
    return shuffled.reshape((nr_epochs * nr_minibatches, nr_minibatch_envs))
