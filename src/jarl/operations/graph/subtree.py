"""Build a visible subtree payload for tree explorers and agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from jarl.experiments.feed import (
    VisibleSubtreePayload,
    build_visible_subtree,
    experiment_tree_root_id,
    node_depth_from_root,
    node_ids_within_depth,
    payload_to_json_dict,
    read_graph_revision,
)
from jarl.experiments.graph import ExperimentGraph
from jarl.operations.contracts.constants import SUBTREE_DEFAULT_DEPTH, SUBTREE_MAX_VISIBLE
from jarl.training.config import RLRunConfig

__all__ = ["SubtreeRequest", "SubtreeResponse", "subtree"]


@dataclass(frozen=True, slots=True)
class SubtreeRequest:
    """Inputs for building a visible subtree."""

    root_id: str | None = None
    depth: int = SUBTREE_DEFAULT_DEPTH
    metric_key: str = "rollout/episode_return"


@dataclass(frozen=True, slots=True)
class SubtreeResponse:
    """Outcome of ``subtree``."""

    payload: VisibleSubtreePayload

    _compact_include: ClassVar[frozenset[str] | None] = None

    def to_compact_dict(self) -> dict[str, object]:
        """Return a compact projection for tools and agents."""
        return payload_to_json_dict(self.payload)


def subtree(
    graph: ExperimentGraph[RLRunConfig],
    request: SubtreeRequest,
) -> SubtreeResponse:
    """Build a depth-limited subtree snapshot rooted at ``request.root_id``."""
    root_id = request.root_id or experiment_tree_root_id(graph)
    graph.get_node(root_id)
    revision = read_graph_revision(graph.layout.root).token()
    visible_ids = node_ids_within_depth(graph, root_id=root_id, max_depth=request.depth)
    expanded_ids = {
        node_id for node_id in visible_ids if node_depth_from_root(graph, node_id, root_id=root_id) < request.depth
    }
    payload = build_visible_subtree(
        graph,
        exp_dir=graph.layout.root,
        metric_key=request.metric_key,
        expanded_ids=expanded_ids,
        revision=revision,
        max_depth=request.depth,
        max_visible=SUBTREE_MAX_VISIBLE,
        root_id=root_id,
    )
    return SubtreeResponse(payload=payload)
