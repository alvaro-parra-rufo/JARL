"""Checkout the current experiment node."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from jarl.experiments.graph import ExperimentGraph
from jarl.operations.contracts.compact import dataclass_to_compact_dict
from jarl.training.config import RLRunConfig

__all__ = ["CheckoutRequest", "CheckoutResponse", "checkout"]


@dataclass(frozen=True, slots=True)
class CheckoutRequest:
    """Target node for checkout."""

    node_id: str


@dataclass(frozen=True, slots=True)
class CheckoutResponse:
    """Outcome of ``checkout``."""

    node_id: str
    status: str
    branch: str

    _compact_include: ClassVar[frozenset[str] | None] = None

    def to_compact_dict(self) -> dict[str, object]:
        """Return a compact projection for tools and agents."""
        return dataclass_to_compact_dict(self, include=self._compact_include)


def checkout(
    graph: ExperimentGraph[RLRunConfig],
    request: CheckoutRequest,
) -> CheckoutResponse:
    """Move the current node pointer without forking."""
    graph.checkout(request.node_id)
    graph.save()
    workspace = graph.current_node
    return CheckoutResponse(
        node_id=workspace.id,
        status=workspace.status.value,
        branch=workspace.branch,
    )
