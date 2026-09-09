"""Promote the best saved checkpoint for a metric."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.io.checkpoints import (
    CHECKPOINT_BEST_RETURN_METRIC,
    select_best_checkpoint,
)
from jarl.operations.contracts.compact import dataclass_to_compact_dict
from jarl.training.config import RLRunConfig

__all__ = ["PromoteRequest", "PromoteResponse", "promote"]


@dataclass(frozen=True, slots=True)
class PromoteRequest:
    """Inputs for promoting a checkpoint.

    ``metric`` is retained for API compatibility. Selection always uses the
    canonical ``best`` policy from `select_best_checkpoint`.
    """

    node_id: str
    metric: str = CHECKPOINT_BEST_RETURN_METRIC


@dataclass(frozen=True, slots=True)
class PromoteResponse:
    """Outcome of ``promote``."""

    node_id: str
    checkpoint_step: int | None
    metric: str
    metric_value: float | None = None

    _compact_include: ClassVar[frozenset[str] | None] = None

    def to_compact_dict(self) -> dict[str, object]:
        """Return a compact projection for tools and agents."""
        return dataclass_to_compact_dict(self, include=self._compact_include)


def promote(
    graph: ExperimentGraph[RLRunConfig],
    request: PromoteRequest,
) -> PromoteResponse:
    """Promote the ``best`` checkpoint alias using the canonical ranking policy."""
    workspace = graph.get_node(request.node_id)
    best_record = select_best_checkpoint(workspace.list_checkpoints())
    if best_record is None:
        graph.save()
        return PromoteResponse(
            node_id=request.node_id,
            checkpoint_step=None,
            metric=CHECKPOINT_BEST_RETURN_METRIC,
        )
    metric_value = float(best_record.metrics[CHECKPOINT_BEST_RETURN_METRIC])
    workspace.promote_checkpoint_best(
        best_record.checkpoint_step,
        metric_name=CHECKPOINT_BEST_RETURN_METRIC,
        metric_value=metric_value,
        reason="operations.promote",
    )
    graph.save()
    return PromoteResponse(
        node_id=request.node_id,
        checkpoint_step=best_record.checkpoint_step,
        metric=CHECKPOINT_BEST_RETURN_METRIC,
        metric_value=metric_value,
    )
