"""Build an enriched experiment or node summary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.summaries import (
    ExperimentSummary,
    NodeSummary,
    build_experiment_summary,
    config_highlights,
)
from jarl.operations.contracts.compact import dataclass_to_compact_dict
from jarl.training.config import RLRunConfig

__all__ = [
    "NodeSummaryEnriched",
    "SummaryRequest",
    "SummaryResponse",
    "summary",
]


@dataclass(frozen=True, slots=True)
class SummaryRequest:
    """Inputs for building an experiment summary."""

    node_id: str | None = None


@dataclass(frozen=True, slots=True)
class NodeSummaryEnriched:
    """Node summary enriched with config highlights."""

    summary: NodeSummary
    config_highlights: dict[str, str | int | float | bool]


@dataclass(frozen=True, slots=True)
class SummaryResponse:
    """Outcome of ``summary``."""

    experiment: ExperimentSummary
    nodes: tuple[NodeSummaryEnriched, ...]

    _compact_include: ClassVar[frozenset[str] | None] = None

    def to_compact_dict(self) -> dict[str, object]:
        """Return a compact projection for tools and agents."""
        experiment = dataclass_to_compact_dict(self.experiment, include=self._compact_include)
        nodes = [
            {
                **dataclass_to_compact_dict(node.summary, include=self._compact_include),
                "config_highlights": node.config_highlights,
            }
            for node in self.nodes
        ]
        return {"experiment": experiment, "nodes": nodes}


def summary(
    graph: ExperimentGraph[RLRunConfig],
    request: SummaryRequest,
) -> SummaryResponse:
    """Build an enriched summary for one node or the full experiment."""
    experiment = build_experiment_summary(graph)
    if request.node_id is None:
        node_summaries = experiment.nodes
    else:
        graph.get_node(request.node_id)
        node_summaries = [node for node in experiment.nodes if node.node_id == request.node_id]
    enriched = tuple(_enrich_node(graph, node_summary) for node_summary in node_summaries)
    return SummaryResponse(experiment=experiment, nodes=enriched)


def _enrich_node(
    graph: ExperimentGraph[RLRunConfig],
    node_summary: NodeSummary,
) -> NodeSummaryEnriched:
    workspace = graph.get_node(node_summary.node_id)
    resolved = graph.resolve_config(workspace)
    return NodeSummaryEnriched(
        summary=node_summary,
        config_highlights=config_highlights(resolved),
    )
