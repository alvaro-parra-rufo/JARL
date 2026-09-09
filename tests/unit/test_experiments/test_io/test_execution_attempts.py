"""Tests for execution attempt registry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarl.experiments.io.execution_attempts import (
    ExecutionAttemptRecord,
    ExecutionAttemptRegistry,
    ExecutionAttemptStatus,
)


class TestExecutionAttemptRecord:
    """Tests for execution attempt record serialization."""

    @pytest.mark.parametrize(
        ("resume_of_attempt_id", "checkpoint_step_at_start", "checkpoint_step_at_end"),
        [
            pytest.param(None, None, None, id="minimal"),
            pytest.param(1, 3, 5, id="resume_with_checkpoints"),
        ],
    )
    def test_round_trip_preserves_fields(
        self,
        resume_of_attempt_id: int | None,
        checkpoint_step_at_start: int | None,
        checkpoint_step_at_end: int | None,
    ) -> None:
        record = ExecutionAttemptRecord(
            attempt_id=2,
            started_at="2026-06-21T12:00:00Z",
            status=ExecutionAttemptStatus.FAILED,
            ended_at="2026-06-21T12:01:00Z",
            checkpoint_step_at_start=checkpoint_step_at_start,
            checkpoint_step_at_end=checkpoint_step_at_end,
            error_type="RuntimeError",
            error_message="boom",
            resume_of_attempt_id=resume_of_attempt_id,
        )

        restored = ExecutionAttemptRecord.from_dict(record.to_dict())

        assert restored == record


class TestExecutionAttemptRegistry:
    """Tests for execution_attempts.json persistence."""

    def test_open_attempt_assigns_monotonic_ids(self, tmp_path: Path) -> None:
        registry = ExecutionAttemptRegistry(tmp_path / "execution_attempts.json")

        first = registry.open_attempt()
        second = registry.open_attempt(resume_of_attempt_id=first.attempt_id, checkpoint_step_at_start=3)

        assert first.attempt_id == 1
        assert second.attempt_id == 2
        assert first.status == ExecutionAttemptStatus.STARTED
        assert second.resume_of_attempt_id == 1
        assert second.checkpoint_step_at_start == 3

    def test_close_attempt_updates_terminal_fields(self, tmp_path: Path) -> None:
        registry = ExecutionAttemptRegistry(tmp_path / "execution_attempts.json")
        opened = registry.open_attempt()

        closed = registry.close_attempt(
            opened.attempt_id,
            status=ExecutionAttemptStatus.FAILED,
            checkpoint_step_at_end=7,
            error_type="ValueError",
            error_message="bad step",
        )

        assert closed.status == ExecutionAttemptStatus.FAILED
        assert closed.ended_at is not None
        assert closed.checkpoint_step_at_end == 7
        assert closed.error_type == "ValueError"
        assert closed.error_message == "bad step"

    def test_latest_closed_attempt_id_skips_started(self, tmp_path: Path) -> None:
        registry = ExecutionAttemptRegistry(tmp_path / "execution_attempts.json")
        first = registry.open_attempt()
        registry.close_attempt(first.attempt_id, status=ExecutionAttemptStatus.INTERRUPTED)
        registry.open_attempt(resume_of_attempt_id=first.attempt_id)

        assert registry.latest_closed_attempt_id() == first.attempt_id

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        registry_path = tmp_path / "execution_attempts.json"
        registry = ExecutionAttemptRegistry(registry_path)
        opened = registry.open_attempt(resume_of_attempt_id=None, checkpoint_step_at_start=1)
        registry.close_attempt(
            opened.attempt_id,
            status=ExecutionAttemptStatus.COMPLETED,
            checkpoint_step_at_end=4,
        )
        registry.save()

        restored = ExecutionAttemptRegistry(registry_path)
        restored.load()
        records = restored.list()

        assert len(records) == 1
        assert records[0].attempt_id == 1
        assert records[0].status == ExecutionAttemptStatus.COMPLETED
        assert records[0].checkpoint_step_at_start == 1
        assert records[0].checkpoint_step_at_end == 4
        assert json.loads(registry_path.read_text(encoding="utf-8"))["attempts"][0]["status"] == "completed"

    def test_load_missing_file_is_empty(self, tmp_path: Path) -> None:
        registry = ExecutionAttemptRegistry(tmp_path / "execution_attempts.json")

        registry.load()

        assert registry.list() == []
        assert registry.latest() is None
        assert registry.latest_closed_attempt_id() is None
