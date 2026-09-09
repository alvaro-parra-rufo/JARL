"""Tests for the built-in demo tree case."""

from __future__ import annotations

from pathlib import Path

from jarl.experiments.cases.builtin.demo_tree import CASE
from jarl.experiments.node import NodeStatus


class TestDemoTreeCase:
    def test_materializes_expected_tree_and_aliases(self, tmp_path: Path) -> None:
        context = CASE.materialize(tmp_path / "demo")

        graph = context.reload_graph()
        root_id = context.node_id("root")
        first_id = context.node_id("first")
        second_id = context.node_id("second")

        assert set(graph.as_networkx().edges) == {
            (root_id, first_id),
            (first_id, second_id),
        }
        assert graph.branch_heads["main"].id == root_id
        assert graph.branch_heads["exp"].id == second_id
        assert graph.current_node.id == second_id
        assert all(graph.get_node(node_id).status is NodeStatus.PREPARED for node_id in (root_id, first_id, second_id))
