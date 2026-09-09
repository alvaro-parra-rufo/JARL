"""Tests for PPO linear learning-rate annealing (Fase 4)."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from jarl.agents.ppo.core.learning_rate import (
    linear_annealed_learning_rate,
    make_linear_annealed_learning_rate_schedule,
)


@pytest.fixture()
def schedule_kwargs() -> dict[str, float | int]:
    """Small schedule for deterministic LR checks."""
    return {
        "base_learning_rate": 1.0,
        "nr_updates": 4,
        "nr_minibatches": 2,
        "nr_epochs": 2,
    }


class TestLinearAnnealedLearningRate:
    """Annealed LR stays non-negative when optimizer steps exceed the planned budget."""

    def test_learning_rate_starts_at_base_value(self, schedule_kwargs: dict[str, float | int]) -> None:
        learning_rate = linear_annealed_learning_rate(jnp.array(0, dtype=jnp.int32), **schedule_kwargs)

        assert float(learning_rate) == pytest.approx(1.0, rel=1e-6)

    def test_learning_rate_decays_linearly_within_budget(self, schedule_kwargs: dict[str, float | int]) -> None:
        optimizer_steps_per_rollout = int(schedule_kwargs["nr_minibatches"] * schedule_kwargs["nr_epochs"])
        midpoint_count = jnp.array(optimizer_steps_per_rollout, dtype=jnp.int32)

        learning_rate = linear_annealed_learning_rate(midpoint_count, **schedule_kwargs)

        assert float(learning_rate) == pytest.approx(0.75, rel=1e-6)

    def test_learning_rate_is_non_negative_after_budget_exceeded(self, schedule_kwargs: dict[str, float | int]) -> None:
        optimizer_steps_per_rollout = int(schedule_kwargs["nr_minibatches"] * schedule_kwargs["nr_epochs"])
        extra_rollouts = 3
        over_budget_count = jnp.array(
            optimizer_steps_per_rollout * (int(schedule_kwargs["nr_updates"]) + extra_rollouts),
            dtype=jnp.int32,
        )

        learning_rate = linear_annealed_learning_rate(over_budget_count, **schedule_kwargs)

        assert float(learning_rate) == pytest.approx(0.0, abs=1e-6)
        assert float(learning_rate) >= 0.0

    def test_make_linear_annealed_learning_rate_schedule_matches_helper(
        self,
        schedule_kwargs: dict[str, float | int],
    ) -> None:
        schedule = make_linear_annealed_learning_rate_schedule(**schedule_kwargs)
        count = jnp.array(6, dtype=jnp.int32)

        assert float(schedule(count)) == pytest.approx(
            float(linear_annealed_learning_rate(count, **schedule_kwargs)),
            rel=1e-6,
        )
