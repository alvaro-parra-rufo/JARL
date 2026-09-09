"""Shared Navix rollout helpers for PPO feedforward and PPO-GRU tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jarl.envs.navix import NavixFullJITState, make_navix_full_jit_env

__all__ = [
    "collect_feedforward_rollout_gae_inputs",
    "collect_navix_rollout_masks",
]


def collect_navix_rollout_masks(
    *,
    max_episode_steps: int,
    nr_envs: int,
    nr_steps: int,
    seed: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Collect episode masks from a Navix rollout scan."""
    env = make_navix_full_jit_env(
        env_id="Navix-Empty-5x5-v0",
        max_episode_steps=max_episode_steps,
    )

    def rollout_step(
        env_state: NavixFullJITState,
        _unused: None,
    ) -> tuple[NavixFullJITState, tuple[jax.Array, jax.Array, jax.Array]]:
        next_state = env.step(env_state, jnp.zeros((nr_envs,), dtype=jnp.int32))
        terminations = next_state.terminated
        truncations = next_state.truncated
        done = terminations | truncations
        return next_state, (terminations, truncations, done)

    reset_keys = jax.random.split(jax.random.PRNGKey(seed), nr_envs)
    init_state = env.reset(reset_keys, eval_mode=False)
    _, batch = jax.lax.scan(rollout_step, init_state, None, nr_steps)
    terminations, truncations, dones = batch
    return terminations, truncations, dones


def collect_feedforward_rollout_gae_inputs(
    *,
    max_episode_steps: int,
    nr_envs: int,
    nr_steps: int,
    seed: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Collect tensors unpacked like ``feedforward_loop`` before the GAE call."""
    env = make_navix_full_jit_env(
        env_id="Navix-Empty-5x5-v0",
        max_episode_steps=max_episode_steps,
    )

    def rollout_step(
        env_state: NavixFullJITState,
        _unused: None,
    ) -> tuple[NavixFullJITState, tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]]:
        observation = env_state.next_observation
        env_state = env.step(env_state, jnp.zeros((nr_envs,), dtype=jnp.int32))
        transition = (
            observation,
            env_state.actual_next_observation,
            env_state.reward,
            jnp.zeros((nr_envs,), dtype=jnp.float32),
            env_state.terminated,
            env_state.truncated,
        )
        return env_state, transition

    reset_keys = jax.random.split(jax.random.PRNGKey(seed), nr_envs)
    init_state = env.reset(reset_keys, eval_mode=False)
    _, batch = jax.lax.scan(rollout_step, init_state, None, nr_steps)
    _states, next_states, rewards, values, terminations, truncations = batch
    return rewards, values, terminations, truncations, next_states
