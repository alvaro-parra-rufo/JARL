"""App-specific graph feed facade tests."""

from __future__ import annotations

from pathlib import Path

from jarl.app.lib.graph_feed import (
    build_visible_subtree,
    combined_graph_revision,
    default_expanded_node_ids,
)
from jarl.experiments.feed import experiment_tree_root_id, payload_to_json_dict
from jarl.experiments.graph import ExperimentGraph
from jarl.training.config import RLRunConfig


def _build_small_tree(exp_dir: Path) -> ExperimentGraph[RLRunConfig]:
    graph = ExperimentGraph(exp_dir, base_config=RLRunConfig())
    root = graph.create_root(RLRunConfig(), branch="main", label="root", prepare=True)
    child = graph.fork("alt", from_node=root, label="child", prepare=True)
    graph.checkout(child.id)
    graph.save()
    return ExperimentGraph.from_directory(exp_dir, config_cls=RLRunConfig)


def test_default_expanded_includes_current_lineage(tmp_path: Path) -> None:
    exp_dir = tmp_path / "exp"
    graph = _build_small_tree(exp_dir)
    root_id = experiment_tree_root_id(graph)
    current_id = graph.current_node.id

    expanded = default_expanded_node_ids(
        graph,
        root_id=root_id,
        current_node_id=current_id,
    )

    assert root_id in expanded
    assert current_id in expanded


def test_build_visible_subtree_merges_session_revision(tmp_path: Path) -> None:
    exp_dir = tmp_path / "exp"
    graph = _build_small_tree(exp_dir)
    root_id = experiment_tree_root_id(graph)

    payload = build_visible_subtree(
        graph,
        exp_dir=exp_dir,
        metric_key="eval/episode_return",
        expanded_ids={root_id},
        max_depth=2,
        max_visible=16,
        session_revision=3,
    )
    data = payload_to_json_dict(payload)

    assert combined_graph_revision(exp_dir, 3) in str(data["revision"])
