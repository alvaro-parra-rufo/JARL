"""Reference GAE helpers for PPO rollout regression tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = [
    "reference_gae_leaky_terminations_only",
    "reference_gae_with_episode_masks",
]


def reference_gae_with_episode_masks(
    rewards: jax.Array,
    values: jax.Array,
    next_values: jax.Array,
    terminations: jax.Array,
    truncations: jax.Array,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """Reference GAE with TimeLimit bootstrap and ``done``-cut carry."""
    bootstrap_mask = terminations.astype(jnp.float32)
    gae_carry_mask = (terminations | truncations).astype(jnp.float32)
    delta = rewards + gamma * next_values * (1.0 - bootstrap_mask) - values
    init_advantage = delta[-1]
    nr_steps = delta.shape[0]

    if nr_steps == 1:
        advantages = init_advantage[None]
    else:
        expected_advantages = [init_advantage]
        for step_index in range(nr_steps - 2, -1, -1):
            previous = expected_advantages[-1]
            expected_advantages.append(
                delta[step_index] + gamma * gae_lambda * (1.0 - gae_carry_mask[step_index]) * previous
            )
        advantages = jnp.stack(list(reversed(expected_advantages)), axis=0)

    returns = advantages + values
    return advantages, returns


def reference_gae_leaky_terminations_only(
    rewards: jax.Array,
    values: jax.Array,
    next_values: jax.Array,
    terminations: jax.Array,
    *,
    gamma: float,
    gae_lambda: float,
) -> jax.Array:
    """Legacy reference: ``terminations`` mask both bootstrap and GAE carry."""
    termination_mask = terminations.astype(jnp.float32)
    delta = rewards + gamma * next_values * (1.0 - termination_mask) - values
    init_advantage = delta[-1]
    nr_steps = delta.shape[0]

    if nr_steps == 1:
        return init_advantage[None]

    expected_advantages = [init_advantage]
    for step_index in range(nr_steps - 2, -1, -1):
        previous = expected_advantages[-1]
        expected_advantages.append(
            delta[step_index] + gamma * gae_lambda * (1.0 - termination_mask[step_index]) * previous
        )
    return jnp.stack(list(reversed(expected_advantages)), axis=0)
