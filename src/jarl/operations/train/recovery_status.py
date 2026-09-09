"""Diagnose whether a failed or interrupted node can resume training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.io.execution_attempts import ExecutionAttemptRecord
from jarl.experiments.node import RESUMABLE_NODE_STATUSES, NodeWorkspace
from jarl.operations.contracts.compact import dataclass_to_compact_dict
from jarl.training.config import RLRunConfig

__all__ = [
    "LatestAttemptSummary",
    "RecommendedAction",
    "RecoveryStatusRequest",
    "RecoveryStatusResponse",
    "recovery_status",
]

RecommendedAction = Literal["train_resume", "none"]
"""Closed enum for the structured recovery recommendation."""


@dataclass(frozen=True, slots=True)
class LatestAttemptSummary:
    """Compact view of the most recent execution attempt."""

    attempt_id: int
    status: str
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class RecoveryStatusRequest:
    """Inputs for diagnosing train recovery on a node."""

    node_id: str | None = None


@dataclass(frozen=True, slots=True)
class RecoveryStatusResponse:
    """Deterministic recovery facts and structured action."""

    node_id: str
    node_status: str
    latest_attempt: LatestAttemptSummary | None
    resumable_checkpoint_step: int | None
    can_resume: bool
    recommended_action: RecommendedAction

    _compact_include: ClassVar[frozenset[str] | None] = None

    def to_compact_dict(self) -> dict[str, object]:
        """Return a compact projection for tools and agents."""
        return dataclass_to_compact_dict(self, include=self._compact_include)


def recovery_status(
    graph: ExperimentGraph[RLRunConfig],
    request: RecoveryStatusRequest,
) -> RecoveryStatusResponse:
    """Return resume eligibility and a closed recommended action for a node."""
    workspace = graph.get_node(request.node_id) if request.node_id else graph.current_node
    return _build_response(workspace)


def _build_response(workspace: NodeWorkspace) -> RecoveryStatusResponse:
    status_value = workspace.status.value
    resume_record = workspace.resolve_resume_checkpoint_record()
    resumable_step = resume_record.checkpoint_step if resume_record is not None else None
    can_resume = workspace.status in RESUMABLE_NODE_STATUSES and resumable_step is not None
    action: RecommendedAction = "train_resume" if can_resume else "none"
    latest = workspace.latest_execution_attempt()

    return RecoveryStatusResponse(
        node_id=workspace.id,
        node_status=status_value,
        latest_attempt=_attempt_summary(latest),
        resumable_checkpoint_step=resumable_step,
        can_resume=can_resume,
        recommended_action=action,
    )


def _attempt_summary(attempt: ExecutionAttemptRecord | None) -> LatestAttemptSummary | None:
    if attempt is None:
        return None
    return LatestAttemptSummary(
        attempt_id=attempt.attempt_id,
        status=attempt.status.value,
        error_type=attempt.error_type,
        error_message=attempt.error_message,
    )
