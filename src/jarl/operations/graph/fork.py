"""Fork a new branch from a parent node."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.io.checkpoints import CheckpointRef
from jarl.operations.contracts.compact import dataclass_to_compact_dict
from jarl.operations.graph._helpers import apply_config_overrides, resolve_parent
from jarl.training.config import RLRunConfig

__all__ = ["ForkRequest", "ForkResponse", "fork"]


@dataclass(frozen=True, slots=True)
class ForkRequest:
    """Inputs for forking a branch."""

    branch: str
    label: str = ""
    from_node: str | None = None
    config_overrides: dict[str, Any] | None = None
    from_checkpoint: CheckpointRef | None = None
    prepare: bool = False


@dataclass(frozen=True, slots=True)
class ForkResponse:
    """Outcome of ``fork``."""

    node_id: str
    branch: str
    parent_id: str
    status: str

    _compact_include: ClassVar[frozenset[str] | None] = None

    def to_compact_dict(self) -> dict[str, object]:
        """Return a compact projection for tools and agents."""
        return dataclass_to_compact_dict(self, include=self._compact_include)


def fork(
    graph: ExperimentGraph[RLRunConfig],
    request: ForkRequest,
) -> ForkResponse:
    """Fork a new branch from a parent node."""
    parent = resolve_parent(graph, request.from_node)
    parent_config = graph.resolve_config(parent)
    target_config = apply_config_overrides(parent_config, request.config_overrides)
    child = graph.fork(
        request.branch,
        from_node=parent,
        config=target_config,
        label=request.label,
        from_checkpoint=request.from_checkpoint,
        prepare=request.prepare,
    )
    graph.save()
    return ForkResponse(
        node_id=child.id,
        branch=child.branch,
        parent_id=parent.id,
        status=child.status.value,
    )
