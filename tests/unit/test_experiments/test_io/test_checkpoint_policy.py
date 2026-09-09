"""Tests for checkpoint save policy evaluation."""

from __future__ import annotations

import pytest

from jarl.experiments.io.checkpoint_policy import CheckpointPolicyState, should_save_checkpoint
from jarl.experiments.run_config import CheckpointConfig


class TestShouldSaveCheckpoint:
    """Tests for pure checkpoint policy evaluation."""

    @pytest.mark.parametrize(
        ("step", "interval", "expected"),
        [
            pytest.param(10, 5, True, id="step_interval_hit"),
            pytest.param(11, 5, False, id="step_interval_miss"),
        ],
    )
    def test_step_interval(self, step: int, interval: int, expected: bool) -> None:
        config = CheckpointConfig(save_interval_steps=interval)
        state = CheckpointPolicyState()

        result = should_save_checkpoint(
            step,
            config=config,
            state=state,
            total_steps=None,
            elapsed_seconds=None,
            monotonic_now=0.0,
        )

        assert result is expected

    def test_time_interval(self) -> None:
        config = CheckpointConfig(save_interval_steps=10_000, save_interval_seconds=30.0)
        state = CheckpointPolicyState()

        assert should_save_checkpoint(
            1,
            config=config,
            state=state,
            total_steps=None,
            elapsed_seconds=0.0,
            monotonic_now=0.0,
        )
        assert should_save_checkpoint(
            2,
            config=config,
            state=state,
            total_steps=None,
            elapsed_seconds=30.0,
            monotonic_now=30.0,
        )

    def test_progress_fractions(self) -> None:
        config = CheckpointConfig(
            save_interval_steps=10_000,
            save_at_progress_fractions=[0.5, 1.0],
        )
        state = CheckpointPolicyState()

        assert not should_save_checkpoint(
            40,
            config=config,
            state=state,
            total_steps=100,
            elapsed_seconds=None,
            monotonic_now=0.0,
        )
        assert should_save_checkpoint(
            50,
            config=config,
            state=state,
            total_steps=100,
            elapsed_seconds=None,
            monotonic_now=1.0,
        )
        assert should_save_checkpoint(
            100,
            config=config,
            state=state,
            total_steps=100,
            elapsed_seconds=None,
            monotonic_now=2.0,
        )
