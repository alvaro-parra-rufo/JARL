"""Runner Lab graph feed helpers (session revision + tree explorer defaults)."""

from __future__ import annotations

from pathlib import Path

from jarl.experiments.feed import VisibleSubtreePayload, read_graph_revision
from jarl.experiments.feed import build_visible_subtree as _build_visible_subtree
from jarl.experiments.graph import ExperimentGraph
from jarl.training.config import RLRunConfig

DEFAULT_MAX_DEPTH = 4
"""Default visible depth for the Runner Lab tree explorer (root depth is zero)."""

MAX_TREE_DEPTH = 100
"""Maximum depth the tree explorer slider can request."""

DEFAULT_MAX_VISIBLE = 64
"""Default cap on nodes included in one tree explorer payload."""

__all__ = [
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MAX_VISIBLE",
    "MAX_TREE_DEPTH",
    "build_visible_subtree",
    "combined_graph_revision",
    "default_expanded_node_ids",
]


def combined_graph_revision(exp_dir: Path, session_revision: int) -> str:
    """Combine disk and in-session revision counters.

    Args:
        exp_dir: Experiment root directory.
        session_revision: Monotonic UI counter from ``session.graph_revision()``.

    Returns:
        Composite revision string for fragment cache invalidation.
    """
    disk = read_graph_revision(exp_dir)
    return f"{disk.token()}:{int(session_revision)}"


def default_expanded_node_ids(
    graph: ExperimentGraph[RLRunConfig],
    *,
    root_id: str,
    current_node_id: str,
) -> set[str]:
    """Return default expanded nodes: root plus ancestors of the current node.

    Args:
        graph: Loaded experiment graph.
        root_id: Tree root node id.
        current_node_id: Active node id in the experiment.

    Returns:
        Node ids that should start expanded in the explorer.
    """
    expanded = {root_id}
    if current_node_id == root_id:
        return expanded
    lineage = graph.get_lineage(current_node_id)
    expanded.update(workspace.id for workspace in lineage)
    return expanded


def build_visible_subtree(
    graph: ExperimentGraph[RLRunConfig],
    *,
    exp_dir: Path,
    metric_key: str,
    expanded_ids: set[str],
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_visible: int = DEFAULT_MAX_VISIBLE,
    search_filter: str = "",
    session_revision: int = 0,
) -> VisibleSubtreePayload:
    """Build the visible subtree payload for the Runner Lab tree explorer."""
    return _build_visible_subtree(
        graph,
        exp_dir=exp_dir,
        metric_key=metric_key,
        expanded_ids=expanded_ids,
        revision=combined_graph_revision(exp_dir, session_revision),
        max_depth=max_depth,
        max_visible=max_visible,
        search_filter=search_filter,
    )
