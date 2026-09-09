"""Read checkpoint overview or filtered candidates for a node."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.io.checkpoints import (
    CHECKPOINT_ALIAS_BEST,
    CHECKPOINT_ALIAS_FINAL,
    CHECKPOINT_ALIAS_LATEST,
    CHECKPOINT_BEST_LENGTH_METRIC,
    CHECKPOINT_BEST_RETURN_METRIC,
    CheckpointRecord,
)
from jarl.experiments.node import NodeWorkspace
from jarl.operations.contracts.compact import dataclass_to_compact_dict
from jarl.operations.contracts.constants import CHECKPOINTS_DEFAULT_LIMIT, CHECKPOINTS_MAX_LIMIT
from jarl.training.config import RLRunConfig

__all__ = [
    "CHECKPOINT_ALIAS_ORDER",
    "CheckpointItem",
    "CheckpointsRequest",
    "CheckpointsResponse",
    "checkpoints",
]

CHECKPOINT_ALIAS_ORDER = (
    CHECKPOINT_ALIAS_BEST,
    CHECKPOINT_ALIAS_LATEST,
    CHECKPOINT_ALIAS_FINAL,
)
"""Stable alias order in checkpoint payloads (``best`` first)."""

CheckpointsMode = Literal["overview", "search"]


@dataclass(frozen=True, slots=True)
class CheckpointsRequest:
    """Inputs for reading node checkpoints.

    Without search filters the response is an overview of aliases. Any of
    ``step_min``, ``step_max``, ``sort_by``, ``limit``, or ``offset`` selects
    search mode and returns a bounded candidate page.
    """

    node_id: str | None = None
    step_min: int | None = None
    step_max: int | None = None
    sort_by: str | None = None
    sort_descending: bool = True
    limit: int | None = None
    offset: int = 0


@dataclass(frozen=True, slots=True)
class CheckpointItem:
    """Compact checkpoint row for agent payloads."""

    checkpoint_step: int
    status: str
    node_step: int
    aliases: tuple[str, ...] = ()
    eval_return: float | None = None
    eval_length: float | None = None


@dataclass(frozen=True, slots=True)
class CheckpointsResponse:
    """Outcome of ``checkpoints``."""

    node_id: str
    mode: CheckpointsMode
    checkpoint_count: int
    latest: CheckpointItem | None = None
    final: CheckpointItem | None = None
    best: CheckpointItem | None = None
    candidates: tuple[CheckpointItem, ...] = ()
    returned_count: int = 0
    offset: int = 0
    has_more: bool = False
    sort_by: str | None = None
    sort_descending: bool = True

    _compact_include: ClassVar[frozenset[str] | None] = None

    def to_compact_dict(self) -> dict[str, object]:
        """Return a compact projection for tools and agents."""
        return dataclass_to_compact_dict(self, include=self._compact_include)


def checkpoints(
    graph: ExperimentGraph[RLRunConfig],
    request: CheckpointsRequest,
) -> CheckpointsResponse:
    """Return an alias overview or a filtered checkpoint page for one node."""
    _validate_request(request)
    workspace = graph.get_node(request.node_id) if request.node_id else graph.current_node
    records = workspace.list_checkpoints()
    alias_steps = _alias_steps(workspace)
    items_by_step = {record.checkpoint_step: _checkpoint_item(record, alias_steps=alias_steps) for record in records}
    latest = _alias_item(workspace, CHECKPOINT_ALIAS_LATEST, items_by_step)
    final = _alias_item(workspace, CHECKPOINT_ALIAS_FINAL, items_by_step)
    best = _alias_item(workspace, CHECKPOINT_ALIAS_BEST, items_by_step)
    if not _is_search_mode(request):
        return CheckpointsResponse(
            node_id=workspace.id,
            mode="overview",
            checkpoint_count=len(records),
            latest=latest,
            final=final,
            best=best,
        )
    limit = request.limit if request.limit is not None else CHECKPOINTS_DEFAULT_LIMIT
    filtered = _filter_records(records, step_min=request.step_min, step_max=request.step_max)
    sort_by = request.sort_by or "checkpoint_step"
    ordered = _sort_records(filtered, sort_by=sort_by, descending=request.sort_descending)
    page = ordered[request.offset : request.offset + limit]
    candidates = tuple(_checkpoint_item(record, alias_steps=alias_steps) for record in page)
    return CheckpointsResponse(
        node_id=workspace.id,
        mode="search",
        checkpoint_count=len(records),
        latest=latest,
        final=final,
        best=best,
        candidates=candidates,
        returned_count=len(candidates),
        offset=request.offset,
        has_more=request.offset + limit < len(ordered),
        sort_by=sort_by,
        sort_descending=request.sort_descending,
    )


def _validate_request(request: CheckpointsRequest) -> None:
    if request.step_min is not None and request.step_max is not None and request.step_min > request.step_max:
        msg = f"step_min ({request.step_min}) must be <= step_max ({request.step_max})."
        raise ValueError(msg)
    if request.offset < 0:
        msg = f"offset must be >= 0, got {request.offset}."
        raise ValueError(msg)
    if request.limit is not None and request.limit <= 0:
        msg = f"limit must be > 0, got {request.limit}."
        raise ValueError(msg)
    if request.limit is not None and request.limit > CHECKPOINTS_MAX_LIMIT:
        msg = f"limit must be <= {CHECKPOINTS_MAX_LIMIT}, got {request.limit}."
        raise ValueError(msg)


def _is_search_mode(request: CheckpointsRequest) -> bool:
    return (
        request.step_min is not None
        or request.step_max is not None
        or request.sort_by is not None
        or request.limit is not None
        or request.offset > 0
    )


def _alias_steps(workspace: NodeWorkspace) -> dict[str, int]:
    steps: dict[str, int] = {}
    for alias in CHECKPOINT_ALIAS_ORDER:
        record = workspace.resolve_checkpoint_alias(alias)
        if record is not None:
            steps[alias] = record.checkpoint_step
    return steps


def _alias_item(
    workspace: NodeWorkspace,
    alias: str,
    items_by_step: dict[int, CheckpointItem],
) -> CheckpointItem | None:
    record = workspace.resolve_checkpoint_alias(alias)
    if record is None:
        return None
    return items_by_step.get(record.checkpoint_step)


def _checkpoint_item(
    record: CheckpointRecord,
    *,
    alias_steps: dict[str, int],
) -> CheckpointItem:
    eval_return = record.metrics.get(CHECKPOINT_BEST_RETURN_METRIC)
    eval_length = record.metrics.get(CHECKPOINT_BEST_LENGTH_METRIC)
    aliases = tuple(alias for alias in CHECKPOINT_ALIAS_ORDER if alias_steps.get(alias) == record.checkpoint_step)
    return CheckpointItem(
        checkpoint_step=record.checkpoint_step,
        status=record.status.value,
        node_step=record.node_step,
        aliases=aliases,
        eval_return=float(eval_return) if eval_return is not None else None,
        eval_length=float(eval_length) if eval_length is not None else None,
    )


def _filter_records(
    records: list[CheckpointRecord],
    *,
    step_min: int | None,
    step_max: int | None,
) -> list[CheckpointRecord]:
    filtered = records
    if step_min is not None:
        filtered = [record for record in filtered if record.checkpoint_step >= step_min]
    if step_max is not None:
        filtered = [record for record in filtered if record.checkpoint_step <= step_max]
    return filtered


def _sort_records(
    records: list[CheckpointRecord],
    *,
    sort_by: str,
    descending: bool,
) -> list[CheckpointRecord]:
    def sort_key(record: CheckpointRecord) -> tuple[float, int]:
        if sort_by == "checkpoint_step":
            value = float(record.checkpoint_step)
        else:
            raw = record.metrics.get(sort_by)
            value = float(raw) if raw is not None else float("-inf")
        return (value, record.checkpoint_step)

    return sorted(records, key=sort_key, reverse=descending)
