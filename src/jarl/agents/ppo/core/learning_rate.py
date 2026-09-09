"""Learning-rate schedule helpers for PPO full-JAX trainers."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

__all__ = [
    "linear_annealed_learning_rate",
    "make_linear_annealed_learning_rate_schedule",
]


def linear_annealed_learning_rate(
    count: jax.Array,
    *,
    base_learning_rate: float,
    nr_updates: int,
    nr_minibatches: int,
    nr_epochs: int,
) -> jax.Array:
    """Compute a linearly annealed learning rate with a non-negative floor.

    Args:
        count: Optimizer step counter from ``optax.inject_hyperparams``.
        base_learning_rate: Initial Adam learning rate.
        nr_updates: Planned rollout updates for the run.
        nr_minibatches: Minibatches per PPO epoch.
        nr_epochs: PPO epochs per rollout.

    Returns:
        Annealed learning rate, clamped to zero once the rollout budget is exceeded.
    """
    optimizer_steps_per_rollout = nr_minibatches * nr_epochs
    rollout_index = count // optimizer_steps_per_rollout
    fraction = 1.0 - rollout_index / nr_updates
    return base_learning_rate * jnp.maximum(fraction, 0.0)


def make_linear_annealed_learning_rate_schedule(
    *,
    base_learning_rate: float,
    nr_updates: int,
    nr_minibatches: int,
    nr_epochs: int,
) -> Callable[[jax.Array], jax.Array]:
    """Build an Optax-compatible linear annealing schedule with a zero floor."""
    return lambda count: linear_annealed_learning_rate(
        count,
        base_learning_rate=base_learning_rate,
        nr_updates=nr_updates,
        nr_minibatches=nr_minibatches,
        nr_epochs=nr_epochs,
    )
