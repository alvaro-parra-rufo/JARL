"""Tests for remaining timestep budget on resume."""

from __future__ import annotations

from jarl.training.config import AlgorithmConfig, EnvironmentConfig, RLRunConfig
from jarl.training.resume_budget import apply_remaining_timestep_budget, remaining_timesteps


class TestResumeBudget:
    """Tests for resume timestep budget helpers."""

    def test_remaining_timesteps_never_negative(self) -> None:
        config = RLRunConfig(algorithm=AlgorithmConfig(total_timesteps=100))
        assert remaining_timesteps(config, 150) == 0

    def test_apply_remaining_timestep_budget_recomputes_schedule(self) -> None:
        config = RLRunConfig(
            environment=EnvironmentConfig(nr_envs=2),
            algorithm=AlgorithmConfig(
                total_timesteps=256,
                nr_steps=4,
                minibatch_size=8,
                evaluation_and_save_frequency=32,
            ),
        )
        adjusted, schedule = apply_remaining_timestep_budget(config, 96)

        assert adjusted.algorithm.total_timesteps == 160
        assert schedule.requested_total_timesteps == 160
