"""Generalized advantage estimation for PPO full-JAX.

Episode boundaries use two masks: bootstrap TD suppresses ``next_values`` only on
``terminations``; GAE carry is cut on ``terminations | truncations`` when
``truncations`` is supplied (TimeLimit convention). Omitting ``truncations``
preserves legacy carry-on-truncation behaviour.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ["compute_gae_advantages"]


def compute_gae_advantages(
    rewards: jax.Array,
    values: jax.Array,
    next_values: jax.Array,
    terminations: jax.Array,
    *,
    truncations: jax.Array | None = None,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """Compute GAE advantages and returns for a fixed-length rollout batch.

    Bootstrap TD error suppresses ``next_values`` only on ``terminations`` (true
    episode ends). Truncation (time limits) may still bootstrap from
    ``actual_next_observation``, matching the Gymnasium TimeLimit convention.

    GAE carry is cut on any episode boundary: ``terminations | truncations`` when
    ``truncations`` is provided, otherwise only on ``terminations``.

    Args:
        rewards: Step rewards with leading rollout-time dimension ``T``.
        values: Value predictions at each step, same shape as ``rewards``.
        next_values: Value predictions for ``actual_next_observation`` at each step.
        terminations: ``True`` when the step ended with a terminal transition.
        truncations: ``True`` when the step ended by time limit. When omitted, only
            ``terminations`` cut GAE carry (legacy behaviour).
        gamma: Discount factor.
        gae_lambda: GAE lambda.

    Returns:
        Tuple ``(advantages, returns)`` with the same shape as ``rewards``.
    """
    bootstrap_mask = terminations.astype(jnp.float32)
    gae_carry_mask = bootstrap_mask if truncations is None else (terminations | truncations).astype(jnp.float32)

    delta = rewards + gamma * next_values * (1.0 - bootstrap_mask) - values
    init_advantage = delta[-1]
    nr_steps = delta.shape[0]

    def compute_advantage(carry: tuple[jax.Array], step_index: jax.Array) -> tuple[tuple[jax.Array], jax.Array]:
        previous_advantage = carry[0]
        advantage = delta[step_index] + gamma * gae_lambda * (1.0 - gae_carry_mask[step_index]) * previous_advantage
        return (advantage,), advantage

    if nr_steps == 1:
        advantages = init_advantage[None]
    else:
        _, reversed_advantages = jax.lax.scan(
            compute_advantage,
            (init_advantage,),
            jnp.arange(nr_steps - 2, -1, -1),
            unroll=True,
        )
        advantages = jnp.concatenate([reversed_advantages[::-1], init_advantage[None]], axis=0)

    returns = advantages + values
    return advantages, returns
