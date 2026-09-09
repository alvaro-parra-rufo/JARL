"""Pin a checkpoint step on a node."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from jarl.experiments.graph import ExperimentGraph
from jarl.operations.contracts.compact import dataclass_to_compact_dict
from jarl.training.config import RLRunConfig

__all__ = ["PinRequest", "PinResponse", "pin"]


@dataclass(frozen=True, slots=True)
class PinRequest:
    """Checkpoint step to pin."""

    node_id: str
    checkpoint_step: int


@dataclass(frozen=True, slots=True)
class PinResponse:
    """Outcome of ``pin``."""

    node_id: str
    checkpoint_step: int
    pinned: bool = True

    _compact_include: ClassVar[frozenset[str] | None] = None

    def to_compact_dict(self) -> dict[str, object]:
        """Return a compact projection for tools and agents."""
        return dataclass_to_compact_dict(self, include=self._compact_include)


def pin(
    graph: ExperimentGraph[RLRunConfig],
    request: PinRequest,
) -> PinResponse:
    """Pin a checkpoint step on the target node."""
    graph.get_node(request.node_id).pin_checkpoint(request.checkpoint_step)
    graph.save()
    return PinResponse(node_id=request.node_id, checkpoint_step=request.checkpoint_step)
