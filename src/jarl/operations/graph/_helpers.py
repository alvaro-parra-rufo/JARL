"""Internal helpers for graph operations."""

from __future__ import annotations

from typing import Any

from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.node import NodeWorkspace
from jarl.training.config import RLRunConfig
from jarl.training.run_overrides import merge_validated_config_overrides


def node_count(graph: ExperimentGraph[RLRunConfig]) -> int:
    """Return the number of nodes in the experiment graph."""
    return graph.as_networkx().number_of_nodes()


def resolve_parent(
    graph: ExperimentGraph[RLRunConfig],
    from_node: str | None,
) -> NodeWorkspace:
    """Resolve the parent workspace for fork operations."""
    if from_node is not None:
        return graph.get_node(from_node)
    return graph.current_node


def apply_config_overrides(
    base_config: RLRunConfig,
    config_overrides: dict[str, Any] | None,
) -> RLRunConfig | None:
    """Merge validated sparse overrides into ``base_config`` for fork/extend."""
    return merge_validated_config_overrides(
        base_config,
        config_overrides,
        policy="fork",
    )
