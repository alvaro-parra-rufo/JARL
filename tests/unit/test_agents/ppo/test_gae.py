"""Regression tests for GAE episode-boundary masks (Fase 1 critic stability)."""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import pytest

import jarl.agents.ppo as ppo_pkg
import jarl.agents.ppo.core as ppo_core
from jarl.agents.ppo.core.gae import compute_gae_advantages
from tests.helpers.ppo_gae_reference import (
    reference_gae_leaky_terminations_only,
    reference_gae_with_episode_masks,
)


def _reference_gae_with_episode_masks(
    rewards: jax.Array,
    values: jax.Array,
    next_values: jax.Array,
    terminations: jax.Array,
    truncations: jax.Array,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    return reference_gae_with_episode_masks(
        rewards,
        values,
        next_values,
        terminations,
        truncations,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )


def _reference_gae_leaky_terminations_only(
    rewards: jax.Array,
    values: jax.Array,
    next_values: jax.Array,
    terminations: jax.Array,
    *,
    gamma: float,
    gae_lambda: float,
) -> jax.Array:
    return reference_gae_leaky_terminations_only(
        rewards,
        values,
        next_values,
        terminations,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )


class TestGAEEpisodeBoundaries:
    """GAE must not leak advantages across truncation boundaries."""

    @pytest.mark.parametrize(
        ("gamma", "gae_lambda"),
        [
            pytest.param(1.0, 1.0, id="identity-discount"),
            pytest.param(0.99, 0.95, id="ppo-defaults"),
        ],
    )
    def test_truncation_bootstraps_next_value_but_cuts_gae_carry(
        self,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Truncation at ``t=1`` must not import advantage mass from the next episode at ``t=2``."""
        rewards = jnp.array([[0.0], [0.0], [100.0]])
        values = jnp.array([[0.0], [0.0], [0.0]])
        next_values = jnp.array([[0.0], [5.0], [0.0]])
        terminations = jnp.array([[False], [False], [False]])
        truncations = jnp.array([[False], [True], [False]])

        advantages, returns = compute_gae_advantages(
            rewards,
            values,
            next_values,
            terminations,
            truncations=truncations,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        expected_advantages, expected_returns = _reference_gae_with_episode_masks(
            rewards,
            values,
            next_values,
            terminations,
            truncations,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        leaky_advantages = _reference_gae_leaky_terminations_only(
            rewards,
            values,
            next_values,
            terminations,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        truncated_step_delta = gamma * float(next_values[1, 0]) - float(values[1, 0])
        next_episode_tail_advantage = float(
            rewards[2, 0] + gamma * float(next_values[2, 0]) * (1.0 - float(terminations[2, 0])) - values[2, 0]
        )
        leaky_advantage_at_truncation = truncated_step_delta + gamma * gae_lambda * next_episode_tail_advantage

        assert advantages.shape == rewards.shape
        assert jnp.allclose(advantages, expected_advantages, atol=1e-6)
        assert jnp.allclose(returns, expected_returns, atol=1e-6)
        assert float(advantages[1, 0]) == pytest.approx(truncated_step_delta, rel=1e-5)
        assert float(leaky_advantages[1, 0]) == pytest.approx(leaky_advantage_at_truncation, rel=1e-5)
        assert leaky_advantage_at_truncation != pytest.approx(truncated_step_delta, rel=1e-3)
        assert float(advantages[1, 0]) != pytest.approx(leaky_advantage_at_truncation, rel=1e-3)

    def test_termination_suppresses_bootstrap_and_cuts_gae_carry(self) -> None:
        """True termination must ignore ``next_values`` and stop carry."""
        rewards = jnp.array([[0.0], [1.0], [50.0]])
        values = jnp.array([[0.0], [0.5], [0.0]])
        next_values = jnp.array([[0.0], [999.0], [0.0]])
        terminations = jnp.array([[False], [True], [False]])
        truncations = jnp.array([[False], [False], [False]])

        advantages, returns = compute_gae_advantages(
            rewards,
            values,
            next_values,
            terminations,
            truncations=truncations,
            gamma=0.99,
            gae_lambda=0.95,
        )
        expected_advantages, expected_returns = _reference_gae_with_episode_masks(
            rewards,
            values,
            next_values,
            terminations,
            truncations,
            gamma=0.99,
            gae_lambda=0.95,
        )

        assert jnp.allclose(advantages, expected_advantages, atol=1e-6)
        assert jnp.allclose(returns, expected_returns, atol=1e-6)
        assert float(advantages[1, 0]) == pytest.approx(0.5, rel=1e-5)

    def test_truncation_only_env_does_not_contaminate_parallel_env(self) -> None:
        """One env truncating must not change carry semantics for a continuing env."""
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
        expected_advantages, _ = _reference_gae_with_episode_masks(
            rewards,
            values,
            next_values,
            terminations,
            truncations,
            gamma=1.0,
            gae_lambda=1.0,
        )
        leaky_advantages = _reference_gae_leaky_terminations_only(
            rewards,
            values,
            next_values,
            terminations,
            gamma=1.0,
            gae_lambda=1.0,
        )

        correct_truncated_advantage = 5.0
        leaky_truncated_advantage = 5.0 + 100.0

        assert float(advantages[1, 0]) == pytest.approx(correct_truncated_advantage, rel=1e-5)
        assert float(leaky_advantages[1, 0]) == pytest.approx(leaky_truncated_advantage, rel=1e-5)
        assert float(advantages[1, 0]) != pytest.approx(leaky_truncated_advantage, rel=1e-3)
        assert float(advantages[1, 1]) == pytest.approx(float(expected_advantages[1, 1]), rel=1e-5)
        assert float(advantages[2, 1]) == pytest.approx(1.0, rel=1e-5)


class TestGAEPublicAPI:
    """Public export and backward-compatible signature for ``compute_gae_advantages``."""

    def test_compute_gae_advantages_exported_without_api_surface_change(self) -> None:
        assert "compute_gae_advantages" in ppo_pkg.__all__
        assert "compute_gae_advantages" in ppo_core.__all__
        assert ppo_pkg.compute_gae_advantages is compute_gae_advantages

    def test_truncations_is_optional_keyword_only_parameter(self) -> None:
        signature = inspect.signature(compute_gae_advantages)
        truncations = signature.parameters["truncations"]
        assert truncations.default is None
        assert truncations.kind is inspect.Parameter.KEYWORD_ONLY

    def test_legacy_call_without_truncations_matches_terminations_only_carry(self) -> None:
        rewards = jnp.array([[1.0], [0.0]])
        values = jnp.array([[0.0], [0.0]])
        next_values = jnp.array([[0.0], [0.0]])
        terminations = jnp.array([[False], [True]])

        advantages, returns = compute_gae_advantages(
            rewards,
            values,
            next_values,
            terminations,
            gamma=1.0,
            gae_lambda=1.0,
        )

        assert advantages.shape == rewards.shape
        assert returns.shape == rewards.shape
        assert float(advantages[0, 0]) == pytest.approx(1.0, rel=1e-6)
        assert float(advantages[1, 0]) == pytest.approx(0.0, rel=1e-6)
