"""Tests for subtree operations with internal roots."""

from __future__ import annotations

from jarl.experiments.feed import experiment_tree_root_id
from jarl.experiments.graph import ExperimentGraph
from jarl.operations.graph.extend import ExtendRequest, extend
from jarl.operations.graph.fork import ForkRequest, fork
from jarl.operations.graph.subtree import SubtreeRequest, subtree
from jarl.training.config import RLRunConfig


class TestSubtreeRootId:
    def test_subtree_respects_internal_root(self, prepared_root: ExperimentGraph[RLRunConfig]) -> None:
        main_root_id = experiment_tree_root_id(prepared_root)
        child = fork(prepared_root, ForkRequest(branch="exp", label="child", prepare=True))

        payload = subtree(
            prepared_root,
            SubtreeRequest(root_id=child.node_id, depth=2),
        ).to_compact_dict()

        node_ids = {node["id"] for node in payload["nodes"]}

        assert payload["root_id"] == child.node_id
        assert child.node_id in node_ids
        assert main_root_id not in node_ids

    def test_subtree_internal_root_includes_grandchildren_at_depth_two(
        self,
        prepared_root: ExperimentGraph[RLRunConfig],
    ) -> None:
        child = fork(prepared_root, ForkRequest(branch="exp", label="child", prepare=True))
        grandchild = extend(prepared_root, ExtendRequest(branch="exp", label="grandchild", prepare=True))
        great_grandchild = extend(prepared_root, ExtendRequest(branch="exp", label="great", prepare=True))

        payload = subtree(
            prepared_root,
            SubtreeRequest(root_id=child.node_id, depth=2),
        ).to_compact_dict()

        node_ids = {node["id"] for node in payload["nodes"]}
        depths = {node["id"]: node["depth"] for node in payload["nodes"]}

        assert node_ids == {child.node_id, grandchild.node_id, great_grandchild.node_id}
        assert depths[child.node_id] == 0
        assert depths[grandchild.node_id] == 1
        assert depths[great_grandchild.node_id] == 2
