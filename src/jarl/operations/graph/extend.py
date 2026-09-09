"""Extend a branch from its head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.io.checkpoints import CheckpointRef
from jarl.operations.contracts.compact import dataclass_to_compact_dict
from jarl.operations.graph._helpers import apply_config_overrides
from jarl.training.config import RLRunConfig

__all__ = ["ExtendRequest", "ExtendResponse", "extend"]


@dataclass(frozen=True, slots=True)
class ExtendRequest:
    """Inputs for extending a branch from its head."""

    label: str = ""
    branch: str | None = None
    config_overrides: dict[str, Any] | None = None
    from_checkpoint: CheckpointRef | None = None
    prepare: bool = False


@dataclass(frozen=True, slots=True)
class ExtendResponse:
    """Outcome of ``extend``."""

    node_id: str
    branch: str
    parent_id: str
    status: str

    _compact_include: ClassVar[frozenset[str] | None] = None

    def to_compact_dict(self) -> dict[str, object]:
        """Return a compact projection for tools and agents."""
        return dataclass_to_compact_dict(self, include=self._compact_include)


def extend(
    graph: ExperimentGraph[RLRunConfig],
    request: ExtendRequest,
) -> ExtendResponse:
    """Extend a branch from its current head."""
    target_branch = request.branch or graph.current_node.branch
    parent = graph.head(target_branch)
    parent_config = graph.resolve_config(parent)
    target_config = apply_config_overrides(parent_config, request.config_overrides)
    child = graph.extend(
        target_branch,
        config=target_config,
        label=request.label,
        from_checkpoint=request.from_checkpoint,
        prepare=request.prepare,
    )
    graph.save()
    return ExtendResponse(
        node_id=child.id,
        branch=child.branch,
        parent_id=parent.id,
        status=child.status.value,
    )
