"""Tests for derived RL training schedule computation."""

from __future__ import annotations

import pytest

from jarl.training.config import AlgorithmConfig, EnvironmentConfig, RLRunConfig
from jarl.training.schedule import (
    TrainingSchedule,
    apply_training_schedule,
    compute_training_schedule,
    format_training_schedule_message,
)


def _reference_build_config_schedule() -> TrainingSchedule:
    """Expected schedule for ``train_ppo_full_jax.build_config``-aligned inputs."""
    environment = EnvironmentConfig(nr_envs=64)
    algorithm = AlgorithmConfig(
        total_timesteps=5_000_000,
        nr_steps=128,
        nr_epochs=4,
        minibatch_size=2048,
        evaluation_and_save_frequency=65_536,
    )

    return compute_training_schedule(environment, algorithm)


class TestComputeTrainingSchedule:
    """Tests for ``compute_training_schedule``."""

    def test_matches_train_ppo_full_jax_build_config(self) -> None:
        schedule = _reference_build_config_schedule()

        assert schedule == TrainingSchedule(
            requested_total_timesteps=5_000_000,
            actual_total_timesteps=4_980_736,
            actual_rollout_updates=608,
            actual_optimizer_updates=9_728,
            batch_size=8_192,
            nr_minibatches=4,
            effective_evaluation_and_save_frequency=65_536,
        )

    @pytest.mark.parametrize(
        ("nr_envs", "nr_steps", "total_timesteps", "minibatch_size", "eval_frequency", "expected"),
        [
            pytest.param(
                8,
                128,
                8192,
                1024,
                -1,
                TrainingSchedule(
                    requested_total_timesteps=8192,
                    actual_total_timesteps=8192,
                    actual_rollout_updates=8,
                    actual_optimizer_updates=32,
                    batch_size=1024,
                    nr_minibatches=1,
                    effective_evaluation_and_save_frequency=8192,
                ),
                id="auto-eval-small-smoke",
            ),
            pytest.param(
                1,
                1,
                1,
                1,
                1,
                TrainingSchedule(
                    requested_total_timesteps=1,
                    actual_total_timesteps=1,
                    actual_rollout_updates=1,
                    actual_optimizer_updates=4,
                    batch_size=1,
                    nr_minibatches=1,
                    effective_evaluation_and_save_frequency=1,
                ),
                id="minimum-env-and-steps",
            ),
        ],
    )
    def test_parametrized_schedules(
        self,
        nr_envs: int,
        nr_steps: int,
        total_timesteps: int,
        minibatch_size: int,
        eval_frequency: int,
        expected: TrainingSchedule,
    ) -> None:
        environment = EnvironmentConfig(nr_envs=nr_envs)
        algorithm = AlgorithmConfig(
            total_timesteps=total_timesteps,
            nr_steps=nr_steps,
            nr_epochs=4,
            minibatch_size=minibatch_size,
            evaluation_and_save_frequency=eval_frequency,
        )

        schedule = compute_training_schedule(environment, algorithm)

        assert schedule == expected

    def test_rejects_evaluation_frequency_not_divisible_by_batch_size(self) -> None:
        environment = EnvironmentConfig(nr_envs=8)
        algorithm = AlgorithmConfig(
            nr_steps=128,
            minibatch_size=1024,
            evaluation_and_save_frequency=1000,
        )

        with pytest.raises(ValueError, match="evaluation_and_save_frequency"):
            compute_training_schedule(environment, algorithm)

    def test_rejects_minibatch_not_dividing_batch_size(self) -> None:
        environment = EnvironmentConfig(nr_envs=8)
        algorithm = AlgorithmConfig(nr_steps=128, minibatch_size=1000)

        with pytest.raises(ValueError, match="minibatch_size"):
            compute_training_schedule(environment, algorithm)


class TestApplyTrainingSchedule:
    """Tests for ``apply_training_schedule``."""

    def test_fills_algorithm_derived_fields(self) -> None:
        config = RLRunConfig(
            environment=EnvironmentConfig(nr_envs=8),
            algorithm=AlgorithmConfig(
                total_timesteps=8192,
                nr_steps=128,
                nr_epochs=4,
                minibatch_size=1024,
                evaluation_and_save_frequency=-1,
            ),
        )

        updated = apply_training_schedule(config)

        assert updated.algorithm.requested_total_timesteps == 8192
        assert updated.algorithm.actual_total_timesteps == 8192
        assert updated.algorithm.actual_rollout_updates == 8
        assert updated.algorithm.actual_optimizer_updates == 32
        assert updated.algorithm.batch_size == 1024
        assert updated.algorithm.nr_minibatches == 1
        assert updated.algorithm.effective_evaluation_and_save_frequency == 8192

    def test_does_not_mutate_input_config(self) -> None:
        config = RLRunConfig(
            environment=EnvironmentConfig(nr_envs=8),
            algorithm=AlgorithmConfig(total_timesteps=8192, nr_steps=128, minibatch_size=1024),
        )

        apply_training_schedule(config)

        assert config.algorithm.batch_size is None


class TestFormatTrainingScheduleMessage:
    """Tests for schedule log formatting."""

    def test_includes_primary_counters(self) -> None:
        schedule = _reference_build_config_schedule()

        message = format_training_schedule_message(schedule)

        assert "batch_size=8192" in message
        assert "actual_rollout_updates=608" in message
        assert "effective_evaluation_and_save_frequency=65536" in message
