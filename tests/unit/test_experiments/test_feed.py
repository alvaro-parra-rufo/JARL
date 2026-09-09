"""Tests for ``jarl.experiments.feed``."""

from __future__ import annotations

from pathlib import Path

from jarl.experiments.feed import (
    build_visible_subtree,
    experiment_tree_root_id,
    node_ids_within_depth,
    payload_to_json_dict,
    read_graph_revision,
)
from jarl.experiments.graph import ExperimentGraph
from jarl.training.config import RLRunConfig


def _build_small_tree(exp_dir: Path) -> ExperimentGraph[RLRunConfig]:
    graph = ExperimentGraph(exp_dir, base_config=RLRunConfig())
    root = graph.create_root(RLRunConfig(), branch="main", label="root", prepare=True)
    child = graph.fork("alt", from_node=root, label="child", prepare=True)
    graph.checkout(child.id)
    graph.save()
    return ExperimentGraph.from_directory(exp_dir, config_cls=RLRunConfig)


class TestReadGraphRevision:
    """Disk revision fingerprint tests."""

    def test_revision_stable_for_unchanged_manifest(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "exp"
        _build_small_tree(exp_dir)

        first = read_graph_revision(exp_dir)
        second = read_graph_revision(exp_dir)

        assert first == second
        assert first.node_count == 2

    def test_revision_changes_when_tree_grows(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "exp"
        graph = _build_small_tree(exp_dir)
        before = read_graph_revision(exp_dir)

        graph.extend("alt", label="grandchild", prepare=True)
        graph.save()
        after = read_graph_revision(exp_dir)

        assert after.node_count == before.node_count + 1
        assert after.token() != before.token()


class TestVisibleSubtree:
    """Visible subtree builder tests."""

    def test_root_only_when_children_collapsed(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "exp"
        graph = _build_small_tree(exp_dir)
        root_id = experiment_tree_root_id(graph)

        payload = build_visible_subtree(
            graph,
            exp_dir=exp_dir,
            metric_key="rollout/episode_return",
            expanded_ids=set(),
            revision=read_graph_revision(exp_dir).token(),
            max_depth=3,
            max_visible=32,
        )

        assert [node.id for node in payload.nodes] == [root_id]
        assert payload.edges == ()

    def test_includes_children_when_parent_expanded(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "exp"
        graph = _build_small_tree(exp_dir)
        root_id = experiment_tree_root_id(graph)
        child_id = graph.branch_heads["alt"].id

        payload = build_visible_subtree(
            graph,
            exp_dir=exp_dir,
            metric_key="rollout/episode_return",
            expanded_ids={root_id, child_id},
            revision=read_graph_revision(exp_dir).token(),
            max_depth=3,
            max_visible=32,
        )

        node_ids = [node.id for node in payload.nodes]
        assert root_id in node_ids
        assert child_id in node_ids
        assert (root_id, child_id) in payload.edges

    def test_search_filter_includes_matching_branch(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "exp"
        graph = _build_small_tree(exp_dir)
        child_id = graph.branch_heads["alt"].id

        payload = build_visible_subtree(
            graph,
            exp_dir=exp_dir,
            metric_key="rollout/episode_return",
            expanded_ids=set(),
            revision=read_graph_revision(exp_dir).token(),
            max_depth=1,
            max_visible=32,
            search_filter=child_id[-6:],
        )

        node_ids = {node.id for node in payload.nodes}
        assert child_id in node_ids

    def test_truncates_when_max_visible_reached(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "exp"
        graph = _build_small_tree(exp_dir)
        root_id = experiment_tree_root_id(graph)

        payload = build_visible_subtree(
            graph,
            exp_dir=exp_dir,
            metric_key="rollout/episode_return",
            expanded_ids={root_id},
            revision=read_graph_revision(exp_dir).token(),
            max_depth=5,
            max_visible=1,
        )

        assert payload.truncated is True
        assert len(payload.nodes) == 1

    def test_respects_custom_root_id(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "exp"
        graph = _build_small_tree(exp_dir)
        child_id = graph.branch_heads["alt"].id

        payload = build_visible_subtree(
            graph,
            exp_dir=exp_dir,
            metric_key="rollout/episode_return",
            expanded_ids={child_id},
            revision=read_graph_revision(exp_dir).token(),
            max_depth=2,
            max_visible=32,
            root_id=child_id,
        )

        node_ids = {node.id for node in payload.nodes}

        assert payload.root_id == child_id
        assert child_id in node_ids
        assert experiment_tree_root_id(graph) not in node_ids

    def test_respects_custom_root_id_with_grandchildren(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "exp"
        graph = _build_small_tree(exp_dir)
        child_id = graph.branch_heads["alt"].id
        grandchild_id = graph.extend("alt", label="grandchild", prepare=True).id

        payload = build_visible_subtree(
            graph,
            exp_dir=exp_dir,
            metric_key="rollout/episode_return",
            expanded_ids={child_id, grandchild_id},
            revision=read_graph_revision(exp_dir).token(),
            max_depth=2,
            max_visible=32,
            root_id=child_id,
        )

        node_ids = {node.id for node in payload.nodes}
        depths = {node.id: node.depth for node in payload.nodes}

        assert node_ids == {child_id, grandchild_id}
        assert depths[child_id] == 0
        assert depths[grandchild_id] == 1

    def test_node_ids_within_depth_limits_reach(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "exp"
        graph = _build_small_tree(exp_dir)
        root_id = experiment_tree_root_id(graph)

        depth_zero = node_ids_within_depth(graph, root_id=root_id, max_depth=0)
        depth_all = node_ids_within_depth(graph, root_id=root_id, max_depth=5)

        assert depth_zero == {root_id}
        assert root_id in depth_all
        assert graph.branch_heads["alt"].id in depth_all


class TestPayloadSerialization:
    """JSON payload conversion tests."""

    def test_payload_to_json_dict_roundtrip_shape(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "exp"
        graph = _build_small_tree(exp_dir)
        root_id = experiment_tree_root_id(graph)
        revision = "disk-rev:3"

        payload = build_visible_subtree(
            graph,
            exp_dir=exp_dir,
            metric_key="eval/episode_return",
            expanded_ids={root_id},
            revision=revision,
            max_depth=2,
            max_visible=16,
        )
        data = payload_to_json_dict(payload)

        assert data["root_id"] == root_id
        assert data["metric_key"] == "eval/episode_return"
        assert isinstance(data["nodes"], list)
        assert data["revision"] == revision
