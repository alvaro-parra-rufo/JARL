"""Feedforward PPO rollout tests for GAE truncation masks (parity with GRU)."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from jarl.agents.ppo.core.gae import compute_gae_advantages
from tests.helpers.ppo_gae_reference import (
    reference_gae_leaky_terminations_only,
    reference_gae_with_episode_masks,
)
from tests.helpers.ppo_rollout_helpers import (
    collect_feedforward_rollout_gae_inputs,
    collect_navix_rollout_masks,
)


class TestFeedforwardRolloutEpisodeBoundaries:
    """Feedforward rollouts must pass ``truncations`` into GAE like ``feedforward_loop``."""

    @pytest.mark.slow
    def test_navix_rollout_records_truncations_without_terminal_flag(self) -> None:
        terminations, truncations, dones = collect_navix_rollout_masks(
            max_episode_steps=2,
            nr_envs=4,
            nr_steps=8,
            seed=0,
        )

        assert jnp.any(truncations)
        assert jnp.any(dones)
        assert jnp.any(truncations & jnp.logical_not(terminations))

    def test_feedforward_rollout_gae_uses_truncation_mask_in_time_major_batch(self) -> None:
        """Same contract as GRU: batch ``[T, nr_envs]`` + ``truncations`` for GAE."""
        rewards = jnp.array([[0.0, 0.0], [0.0, 0.0], [100.0, 1.0]])
        values = jnp.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        next_values = jnp.array([[0.0, 0.0], [5.0, 0.0], [0.0, 0.0]])
        terminations = jnp.array([[False, False], [False, False], [False, False]])
        truncations = jnp.array([[False, False], [True, False], [False, False]])

        advantages, _ = compute_gae_advantages(
            rewards,
            values,
            next_values,
            terminations,
            truncations=truncations,
            gamma=1.0,
            gae_lambda=1.0,
        )
        expected_advantages, _ = reference_gae_with_episode_masks(
            rewards,
            values,
            next_values,
            terminations,
            truncations,
            gamma=1.0,
            gae_lambda=1.0,
        )
        leaky_advantages = reference_gae_leaky_terminations_only(
            rewards,
            values,
            next_values,
            terminations,
            gamma=1.0,
            gae_lambda=1.0,
        )

        leaky_truncated_advantage = 105.0

        assert jnp.allclose(advantages, expected_advantages, atol=1e-6)
        assert float(advantages[1, 0]) == pytest.approx(5.0, rel=1e-5)
        assert float(leaky_advantages[1, 0]) == pytest.approx(leaky_truncated_advantage, rel=1e-5)
        assert float(advantages[1, 0]) != pytest.approx(leaky_truncated_advantage, rel=1e-3)

    @pytest.mark.slow
    def test_feedforward_loop_unpack_produces_truncations_for_gae(self) -> None:
        """Navix rollout via feedforward transition layout must feed non-empty truncations."""
        rewards, values, terminations, truncations, next_states = collect_feedforward_rollout_gae_inputs(
            max_episode_steps=2,
            nr_envs=4,
            nr_steps=8,
            seed=1,
        )

        assert rewards.shape == terminations.shape == truncations.shape == next_states.shape[:2]
        assert jnp.any(truncations)

        trunc_positions = jnp.argwhere(truncations)
        assert trunc_positions.shape[0] > 0
        time_index, env_index = trunc_positions[0]
        assert int(time_index) + 1 < rewards.shape[0]
        # Navix rewards are often zero; inject a post-truncation spike so leaky
        # carry differs from the fixed mask at the truncation step.
        rewards = rewards.at[time_index + 1, env_index].set(100.0)

        next_values = jnp.zeros_like(values)
        advantages_fixed, _ = compute_gae_advantages(
            rewards,
            values,
            next_values,
            terminations,
            truncations=truncations,
            gamma=0.99,
            gae_lambda=0.95,
        )
        advantages_leaky = reference_gae_leaky_terminations_only(
            rewards,
            values,
            next_values,
            terminations,
            gamma=0.99,
            gae_lambda=0.95,
        )
        expected_advantages, _ = reference_gae_with_episode_masks(
            rewards,
            values,
            next_values,
            terminations,
            truncations,
            gamma=0.99,
            gae_lambda=0.95,
        )

        assert jnp.allclose(advantages_fixed, expected_advantages, atol=1e-6)
        fixed_value = float(advantages_fixed[time_index, env_index])
        leaky_value = float(advantages_leaky[time_index, env_index])
        assert fixed_value != pytest.approx(leaky_value, rel=1e-3)
