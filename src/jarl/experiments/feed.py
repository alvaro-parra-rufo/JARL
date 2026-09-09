"""Incremental experiment graph feed for tree explorers and agent tools."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.io.layout import ExperimentLayout
from jarl.experiments.node import NodeWorkspace
from jarl.experiments.summaries import config_highlights, latest_metric
from jarl.training.config import RLRunConfig

__all__ = [
    "DiskGraphRevision",
    "GraphNodePayload",
    "VisibleSubtreePayload",
    "build_visible_subtree",
    "experiment_tree_root_id",
    "node_depth_from_root",
    "node_ids_within_depth",
    "payload_to_json_dict",
    "read_graph_revision",
]


@dataclass(frozen=True, slots=True)
class DiskGraphRevision:
    """Lightweight on-disk fingerprint for an experiment tree."""

    updated_at: str
    node_count: int
    manifest_mtime_ns: int
    activity_mtime_ns: int

    def token(self) -> str:
        """Return a stable string token for cache keys."""
        return f"{self.updated_at}:{self.node_count}:{self.manifest_mtime_ns}:{self.activity_mtime_ns}"


@dataclass(frozen=True, slots=True)
class GraphNodePayload:
    """One node in the visible subtree feed."""

    id: str
    parent_id: str | None
    branch: str
    label: str
    status: str
    metric_value: float | None
    depth: int
    has_children: bool
    expanded: bool
    step: int
    train_return: float | None
    eval_return: float | None
    highlights: dict[str, str | int | float | bool]


@dataclass(frozen=True, slots=True)
class VisibleSubtreePayload:
    """JSON-serializable subtree snapshot for tree explorer components."""

    revision: str
    experiment_dir: str
    root_id: str
    current_node_id: str
    metric_key: str
    nodes: tuple[GraphNodePayload, ...]
    edges: tuple[tuple[str, str], ...]
    truncated: bool
    max_depth: int
    max_visible: int
    search_filter: str


def read_graph_revision(exp_dir: Path) -> DiskGraphRevision:
    """Read a disk fingerprint from ``experiment.json`` and node activity mtimes.

    Args:
        exp_dir: Experiment root directory.

    Returns:
        Revision snapshot used to detect structural and status/metric changes.
    """
    layout = ExperimentLayout(exp_dir)
    manifest_path = layout.manifest_path
    if not manifest_path.is_file():
        msg = f"Experiment manifest not found: {manifest_path}"
        raise FileNotFoundError(msg)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    execution_state = payload.get("execution_state", {})
    updated_at = str(execution_state.get("updated_at", ""))
    nodes = payload.get("nodes", [])
    return DiskGraphRevision(
        updated_at=updated_at,
        node_count=len(nodes),
        manifest_mtime_ns=manifest_path.stat().st_mtime_ns,
        activity_mtime_ns=_experiment_activity_mtime_ns(layout),
    )


def experiment_tree_root_id(graph: ExperimentGraph[RLRunConfig]) -> str:
    """Return the unique root node id for a tree-shaped experiment graph.

    Args:
        graph: Loaded experiment graph.

    Returns:
        Root node identifier.

    Raises:
        ValueError: If the graph has no nodes or multiple roots.
    """
    tree = graph.as_networkx()
    roots = [node_id for node_id, in_degree in tree.in_degree() if in_degree == 0]
    if not roots:
        msg = "Experiment graph has no nodes."
        raise ValueError(msg)
    if len(roots) > 1:
        msg = f"Experiment graph has multiple roots: {roots}"
        raise ValueError(msg)
    return roots[0]


def node_ids_within_depth(
    graph: ExperimentGraph[RLRunConfig],
    *,
    root_id: str,
    max_depth: int,
) -> set[str]:
    """Return node ids reachable from ``root_id`` up to ``max_depth``.

    Args:
        graph: Loaded experiment graph.
        root_id: Tree root node id.
        max_depth: Maximum depth from root (root depth is zero).

    Returns:
        Node ids included in a depth-limited subtree.
    """
    children = _children_by_parent(graph)
    included: set[str] = set()

    def visit(node_id: str, depth: int) -> None:
        if depth > max_depth:
            return
        included.add(node_id)
        for child_id in children.get(node_id, ()):
            visit(child_id, depth + 1)

    visit(root_id, 0)
    return included


def node_depth_from_root(
    graph: ExperimentGraph[RLRunConfig],
    node_id: str,
    *,
    root_id: str,
) -> int:
    """Return the depth of ``node_id`` relative to ``root_id`` (root depth is zero).

    Args:
        graph: Loaded experiment graph.
        node_id: Node to measure.
        root_id: Subtree root node id.

    Returns:
        Depth from ``root_id`` to ``node_id``.

    Raises:
        ValueError: If ``node_id`` is not reachable from ``root_id``.
    """
    if node_id == root_id:
        return 0
    lineage_ids = [workspace.id for workspace in graph.get_lineage(node_id)]
    try:
        root_index = lineage_ids.index(root_id)
    except ValueError as exc:
        msg = f"Node {node_id!r} is not under root {root_id!r}."
        raise ValueError(msg) from exc
    return len(lineage_ids) - root_index - 1


def build_visible_subtree(
    graph: ExperimentGraph[RLRunConfig],
    *,
    exp_dir: Path,
    metric_key: str,
    expanded_ids: set[str],
    revision: str,
    max_depth: int,
    max_visible: int,
    search_filter: str = "",
    root_id: str | None = None,
) -> VisibleSubtreePayload:
    """Build the visible subtree payload for a tree explorer.

    Args:
        graph: Loaded experiment graph.
        exp_dir: Experiment root directory.
        metric_key: Metric used for node coloring in the UI.
        expanded_ids: Node ids currently expanded by the operator.
        revision: Caller-provided revision token (disk-only or combined with session).
        max_depth: Maximum depth from root (root depth is zero).
        max_visible: Maximum number of nodes to include before truncation.
        search_filter: Optional case-insensitive id substring filter.
        root_id: Optional subtree root. Defaults to the experiment tree root.

    Returns:
        Visible subtree snapshot for the frontend component.
    """
    resolved_root_id = root_id or experiment_tree_root_id(graph)
    graph.get_node(resolved_root_id)
    current_node_id = graph.current_node.id
    children = _children_by_parent(graph)
    forced_ids = _search_matched_nodes_with_ancestors(
        graph,
        root_id=resolved_root_id,
        search_filter=search_filter,
    )
    visible_ids, truncated = _collect_visible_node_ids(
        root_id=resolved_root_id,
        children=children,
        expanded_ids=expanded_ids,
        forced_ids=forced_ids,
        max_depth=max_depth,
        max_visible=max_visible,
    )

    node_payloads: list[GraphNodePayload] = []
    edges: list[tuple[str, str]] = []
    for node_id in visible_ids:
        workspace = graph.get_node(node_id)
        depth = _node_depth(graph, node_id, root_id=resolved_root_id)
        child_ids = children.get(node_id, ())
        node_children = [child_id for child_id in child_ids if child_id in visible_ids]
        edges.extend((node_id, child_id) for child_id in node_children)
        node_payloads.append(
            _build_node_payload(
                graph=graph,
                workspace=workspace,
                metric_key=metric_key,
                depth=depth,
                has_children=bool(child_ids),
                expanded=node_id in expanded_ids,
            )
        )

    return VisibleSubtreePayload(
        revision=revision,
        experiment_dir=str(exp_dir.resolve()),
        root_id=resolved_root_id,
        current_node_id=current_node_id,
        metric_key=metric_key,
        nodes=tuple(node_payloads),
        edges=tuple(edges),
        truncated=truncated,
        max_depth=max_depth,
        max_visible=max_visible,
        search_filter=search_filter,
    )


def payload_to_json_dict(payload: VisibleSubtreePayload) -> dict[str, object]:
    """Convert a subtree payload into a JSON-serializable dictionary.

    Args:
        payload: Visible subtree snapshot.

    Returns:
        Dictionary suitable for custom tree explorer components.
    """
    return {
        "revision": payload.revision,
        "experiment_dir": payload.experiment_dir,
        "root_id": payload.root_id,
        "current_node_id": payload.current_node_id,
        "metric_key": payload.metric_key,
        "nodes": [asdict(node) for node in payload.nodes],
        "edges": [list(edge) for edge in payload.edges],
        "truncated": payload.truncated,
        "max_depth": payload.max_depth,
        "max_visible": payload.max_visible,
        "search_filter": payload.search_filter,
    }


def _experiment_activity_mtime_ns(layout: ExperimentLayout) -> int:
    nodes_dir = layout.nodes_dir
    if not nodes_dir.is_dir():
        return 0
    max_ns = 0
    for pattern in ("*/node.json", "*/metrics.jsonl"):
        for path in nodes_dir.glob(pattern):
            max_ns = max(max_ns, path.stat().st_mtime_ns)
    return max_ns


def _children_by_parent(graph: ExperimentGraph[RLRunConfig]) -> dict[str, tuple[str, ...]]:
    tree = graph.as_networkx()
    children: dict[str, list[str]] = {}
    for parent_id, child_id in tree.edges():
        children.setdefault(parent_id, []).append(child_id)
    return {parent_id: tuple(sorted(child_ids)) for parent_id, child_ids in children.items()}


def _node_depth(graph: ExperimentGraph[RLRunConfig], node_id: str, *, root_id: str) -> int:
    return node_depth_from_root(graph, node_id, root_id=root_id)


def _search_matched_nodes_with_ancestors(
    graph: ExperimentGraph[RLRunConfig],
    *,
    root_id: str,
    search_filter: str,
) -> set[str]:
    needle = search_filter.strip().casefold()
    if not needle:
        return set()
    matched: set[str] = set()
    for node_id in graph.as_networkx().nodes:
        if needle in node_id.casefold():
            matched.add(node_id)
            for workspace in graph.get_lineage(node_id):
                matched.add(workspace.id)
    matched.add(root_id)
    return matched


def _collect_visible_node_ids(
    *,
    root_id: str,
    children: Mapping[str, Sequence[str]],
    expanded_ids: set[str],
    forced_ids: set[str],
    max_depth: int,
    max_visible: int,
) -> tuple[list[str], bool]:
    visible_order: list[str] = []
    seen: set[str] = set()
    truncated = False

    def visit(node_id: str, depth: int) -> None:
        nonlocal truncated
        if truncated or node_id in seen:
            return
        if depth > max_depth and node_id not in forced_ids:
            return
        seen.add(node_id)
        visible_order.append(node_id)
        if len(visible_order) >= max_visible:
            truncated = True
            return
        if node_id not in expanded_ids and node_id not in forced_ids:
            return
        for child_id in children.get(node_id, ()):
            visit(child_id, depth + 1)

    visit(root_id, 0)
    if forced_ids:
        for node_id in sorted(forced_ids):
            if node_id not in seen and len(visible_order) < max_visible:
                visible_order.append(node_id)
                seen.add(node_id)
            elif node_id not in seen:
                truncated = True
    return visible_order, truncated


def _build_node_payload(
    *,
    graph: ExperimentGraph[RLRunConfig],
    workspace: NodeWorkspace,
    metric_key: str,
    depth: int,
    has_children: bool,
    expanded: bool,
) -> GraphNodePayload:
    metric_value = latest_metric(workspace, metric_key)
    highlights: dict[str, str | int | float | bool] = {}
    try:
        config = graph.resolve_config(workspace)
        highlights = config_highlights(config)
    except Exception:
        highlights = {}

    return GraphNodePayload(
        id=workspace.id,
        parent_id=workspace.node_metadata.parent_id,
        branch=workspace.branch,
        label=workspace.node_metadata.label,
        status=workspace.status.value,
        metric_value=metric_value,
        depth=depth,
        has_children=has_children,
        expanded=expanded,
        step=workspace.node_metadata.step,
        train_return=latest_metric(workspace, "rollout/episode_return"),
        eval_return=latest_metric(workspace, "eval/episode_return"),
        highlights=highlights,
    )
