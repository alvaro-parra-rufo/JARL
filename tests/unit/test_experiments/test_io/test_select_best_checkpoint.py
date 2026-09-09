"""Tests for canonical best-checkpoint selection."""

from __future__ import annotations

from jarl.experiments.io.checkpoints import (
    CHECKPOINT_BEST_LENGTH_METRIC,
    CHECKPOINT_BEST_RETURN_METRIC,
    CheckpointRecord,
    CheckpointStatus,
    select_best_checkpoint,
)


def _record(
    step: int,
    *,
    episode_return: float | None = None,
    episode_length: float | None = None,
    status: CheckpointStatus = CheckpointStatus.SAVED,
) -> CheckpointRecord:
    metrics: dict[str, float] = {}
    if episode_return is not None:
        metrics[CHECKPOINT_BEST_RETURN_METRIC] = episode_return
    if episode_length is not None:
        metrics[CHECKPOINT_BEST_LENGTH_METRIC] = episode_length
    return CheckpointRecord.for_step(
        node_step=step,
        checkpoint_step=step,
        metrics=metrics,
        status=status,
    )


class TestSelectBestCheckpoint:
    def test_prefers_higher_return_over_latest_step(self) -> None:
        records = (
            _record(20, episode_return=0.9, episode_length=12.0),
            _record(30, episode_return=0.4, episode_length=40.0),
        )

        best = select_best_checkpoint(records)

        assert best is not None
        assert best.checkpoint_step == 20

    def test_return_tie_prefers_shorter_episode(self) -> None:
        records = (
            _record(10, episode_return=1.0, episode_length=20.0),
            _record(20, episode_return=1.0, episode_length=8.0),
        )

        best = select_best_checkpoint(records)

        assert best is not None
        assert best.checkpoint_step == 20

    def test_return_and_length_tie_prefers_higher_step(self) -> None:
        records = (
            _record(10, episode_return=1.0, episode_length=5.0),
            _record(30, episode_return=1.0, episode_length=5.0),
        )

        best = select_best_checkpoint(records)

        assert best is not None
        assert best.checkpoint_step == 30

    def test_missing_length_loses_length_tie(self) -> None:
        records = (
            _record(10, episode_return=1.0),
            _record(20, episode_return=1.0, episode_length=5.0),
        )

        best = select_best_checkpoint(records)

        assert best is not None
        assert best.checkpoint_step == 20

    def test_ignores_failed_and_missing_return(self) -> None:
        records = (
            _record(5, episode_return=2.0, status=CheckpointStatus.FAILED),
            _record(10),
            _record(15, episode_return=0.5, episode_length=3.0),
        )

        best = select_best_checkpoint(records)

        assert best is not None
        assert best.checkpoint_step == 15

    def test_empty_returns_none(self) -> None:
        assert select_best_checkpoint(()) is None
