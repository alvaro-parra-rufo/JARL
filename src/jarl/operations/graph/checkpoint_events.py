"""Read checkpoint registry and execution attempts for a node."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.io.checkpoints import CheckpointRecord
from jarl.experiments.io.execution_attempts import ExecutionAttemptRecord
from jarl.operations.contracts.compact import dataclass_to_compact_dict
from jarl.training.config import RLRunConfig

__all__ = [
    "CheckpointEvent",
    "CheckpointEventsRequest",
    "CheckpointEventsResponse",
    "ExecutionAttemptEvent",
    "checkpoint_events",
]


@dataclass(frozen=True, slots=True)
class CheckpointEvent:
    """Compact checkpoint registry entry."""

    checkpoint_step: int
    status: str
    node_step: int
    metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class ExecutionAttemptEvent:
    """Compact execution attempt entry."""

    attempt_id: int
    status: str
    started_at: str
    ended_at: str | None = None
    checkpoint_step_at_start: int | None = None
    checkpoint_step_at_end: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    resume_of_attempt_id: int | None = None


@dataclass(frozen=True, slots=True)
class CheckpointEventsRequest:
    """Inputs for reading checkpoint and attempt events."""

    node_id: str
    checkpoint_step: int | None = None


@dataclass(frozen=True, slots=True)
class CheckpointEventsResponse:
    """Outcome of ``checkpoint_events``."""

    node_id: str
    checkpoints: tuple[CheckpointEvent, ...]
    execution_attempts: tuple[ExecutionAttemptEvent, ...]

    _compact_include: ClassVar[frozenset[str] | None] = None

    def to_compact_dict(self) -> dict[str, object]:
        """Return a compact projection for tools and agents."""
        return {
            "node_id": self.node_id,
            "checkpoints": [dataclass_to_compact_dict(item) for item in self.checkpoints],
            "execution_attempts": [dataclass_to_compact_dict(item) for item in self.execution_attempts],
        }


def checkpoint_events(
    graph: ExperimentGraph[RLRunConfig],
    request: CheckpointEventsRequest,
) -> CheckpointEventsResponse:
    """Return checkpoint registry entries and execution attempts for a node."""
    workspace = graph.get_node(request.node_id)
    checkpoints = [_checkpoint_event(record) for record in workspace.list_checkpoints()]
    if request.checkpoint_step is not None:
        checkpoints = [item for item in checkpoints if item.checkpoint_step == request.checkpoint_step]
    attempts = tuple(_attempt_event(record) for record in workspace.list_execution_attempts())
    return CheckpointEventsResponse(
        node_id=request.node_id,
        checkpoints=tuple(checkpoints),
        execution_attempts=attempts,
    )


def _checkpoint_event(record: CheckpointRecord) -> CheckpointEvent:
    return CheckpointEvent(
        checkpoint_step=record.checkpoint_step,
        status=record.status.value,
        node_step=record.node_step,
        metrics=dict(record.metrics),
    )


def _attempt_event(record: ExecutionAttemptRecord) -> ExecutionAttemptEvent:
    return ExecutionAttemptEvent(
        attempt_id=record.attempt_id,
        status=record.status.value,
        started_at=record.started_at,
        ended_at=record.ended_at,
        checkpoint_step_at_start=record.checkpoint_step_at_start,
        checkpoint_step_at_end=record.checkpoint_step_at_end,
        error_type=record.error_type,
        error_message=record.error_message,
        resume_of_attempt_id=record.resume_of_attempt_id,
    )
