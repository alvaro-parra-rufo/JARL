"""Execution attempt registry for node training runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from jarl.metadata import now_iso
from jarl.utils import write_text_atomic

__all__ = [
    "ExecutionAttemptRecord",
    "ExecutionAttemptRegistry",
    "ExecutionAttemptStatus",
]


class ExecutionAttemptStatus(StrEnum):
    """Outcome status for one training execution attempt."""

    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class ExecutionAttemptRecord:
    """One persisted training execution attempt for a node.

    Args:
        attempt_id: Monotonic attempt identifier within the node.
        started_at: ISO-8601 timestamp when the attempt opened.
        status: Attempt outcome status.
        ended_at: ISO-8601 timestamp when the attempt closed.
        checkpoint_step_at_start: Restorable checkpoint used when resuming.
        checkpoint_step_at_end: Latest saved checkpoint when the attempt closed.
        error_type: Exception class name when the attempt failed.
        error_message: Exception message when the attempt failed.
        resume_of_attempt_id: Prior attempt resumed by this attempt, if any.
    """

    attempt_id: int
    started_at: str
    status: ExecutionAttemptStatus
    ended_at: str | None = None
    checkpoint_step_at_start: int | None = None
    checkpoint_step_at_end: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    resume_of_attempt_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the attempt record to a JSON-serializable dictionary."""
        return {
            "attempt_id": self.attempt_id,
            "started_at": self.started_at,
            "status": self.status.value,
            "ended_at": self.ended_at,
            "checkpoint_step_at_start": self.checkpoint_step_at_start,
            "checkpoint_step_at_end": self.checkpoint_step_at_end,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "resume_of_attempt_id": self.resume_of_attempt_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ExecutionAttemptRecord:
        """Build an attempt record from a parsed JSON object.

        Args:
            data: Parsed JSON object for one attempt entry.

        Returns:
            Parsed `ExecutionAttemptRecord`.
        """
        return cls(
            attempt_id=int(data["attempt_id"]),  # type: ignore[arg-type]
            started_at=str(data["started_at"]),
            status=ExecutionAttemptStatus(str(data["status"])),
            ended_at=str(data["ended_at"]) if data.get("ended_at") is not None else None,
            checkpoint_step_at_start=(
                int(data["checkpoint_step_at_start"])  # type: ignore[arg-type]
                if data.get("checkpoint_step_at_start") is not None
                else None
            ),
            checkpoint_step_at_end=(
                int(data["checkpoint_step_at_end"])  # type: ignore[arg-type]
                if data.get("checkpoint_step_at_end") is not None
                else None
            ),
            error_type=str(data["error_type"]) if data.get("error_type") is not None else None,
            error_message=str(data["error_message"]) if data.get("error_message") is not None else None,
            resume_of_attempt_id=(
                int(data["resume_of_attempt_id"])  # type: ignore[arg-type]
                if data.get("resume_of_attempt_id") is not None
                else None
            ),
        )


class ExecutionAttemptRegistry:
    """In-memory execution attempt index backed by ``execution_attempts.json``.

    Args:
        path: Registry file path (typically ``execution_attempts.json``).
    """

    def __init__(self, path: str | Path) -> None:
        """Initialize an empty or disk-backed execution attempt registry."""
        self._path = Path(path)
        self._records: dict[int, ExecutionAttemptRecord] = {}

    @property
    def path(self) -> Path:
        """Registry file path."""
        return self._path

    def load(self) -> None:
        """Load attempt records from disk when the file exists."""
        self._records.clear()
        if not self._path.exists():
            return
        data = json.loads(self._path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            msg = f"Execution attempt registry must be a JSON object: {self._path}"
            raise TypeError(msg)
        attempts = data.get("attempts", [])
        if not isinstance(attempts, list):
            msg = f"Execution attempt registry attempts must be a JSON list: {self._path}"
            raise TypeError(msg)
        for item in attempts:
            if not isinstance(item, dict):
                msg = f"Execution attempt entries must be JSON objects: {self._path}"
                raise TypeError(msg)
            record = ExecutionAttemptRecord.from_dict(item)
            self._records[record.attempt_id] = record

    def save(self) -> Path:
        """Persist the registry atomically to disk.

        Returns:
            Path to the written registry file.
        """
        payload = {
            "attempts": [record.to_dict() for record in self._sorted_records()],
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        return write_text_atomic(self._path, json.dumps(payload, indent=2, default=str))

    def list(self) -> list[ExecutionAttemptRecord]:
        """Return all attempt records in ascending attempt order."""
        return self._sorted_records()

    def latest(self) -> ExecutionAttemptRecord | None:
        """Return the most recent attempt record, if any."""
        records = self._sorted_records()
        if not records:
            return None
        return records[-1]

    def latest_closed_attempt_id(self) -> int | None:
        """Return the latest attempt id that is no longer ``started``."""
        for record in reversed(self._sorted_records()):
            if record.status != ExecutionAttemptStatus.STARTED:
                return record.attempt_id
        return None

    def open_attempt(
        self,
        *,
        resume_of_attempt_id: int | None = None,
        checkpoint_step_at_start: int | None = None,
    ) -> ExecutionAttemptRecord:
        """Create and register a new started attempt.

        Args:
            resume_of_attempt_id: Prior attempt resumed by this run.
            checkpoint_step_at_start: Restorable checkpoint selected for resume.

        Returns:
            Newly opened attempt record.
        """
        attempt_id = self._next_attempt_id()
        record = ExecutionAttemptRecord(
            attempt_id=attempt_id,
            started_at=now_iso(),
            status=ExecutionAttemptStatus.STARTED,
            resume_of_attempt_id=resume_of_attempt_id,
            checkpoint_step_at_start=checkpoint_step_at_start,
        )
        self._records[attempt_id] = record
        return record

    def close_attempt(
        self,
        attempt_id: int,
        *,
        status: ExecutionAttemptStatus,
        checkpoint_step_at_end: int | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> ExecutionAttemptRecord:
        """Close an open attempt with a terminal status.

        Args:
            attempt_id: Attempt identifier to close.
            status: Terminal attempt status.
            checkpoint_step_at_end: Latest saved checkpoint at close time.
            error_type: Exception class name for failed attempts.
            error_message: Exception message for failed attempts.

        Returns:
            Updated attempt record.

        Raises:
            KeyError: If the attempt record does not exist.
        """
        existing = self._records[attempt_id]
        updated = ExecutionAttemptRecord(
            attempt_id=existing.attempt_id,
            started_at=existing.started_at,
            status=status,
            ended_at=now_iso(),
            checkpoint_step_at_start=existing.checkpoint_step_at_start,
            checkpoint_step_at_end=checkpoint_step_at_end,
            error_type=error_type,
            error_message=error_message,
            resume_of_attempt_id=existing.resume_of_attempt_id,
        )
        self._records[attempt_id] = updated
        return updated

    def _next_attempt_id(self) -> int:
        if not self._records:
            return 1
        return max(self._records) + 1

    def _sorted_records(self) -> list[ExecutionAttemptRecord]:
        return [self._records[attempt_id] for attempt_id in sorted(self._records)]
