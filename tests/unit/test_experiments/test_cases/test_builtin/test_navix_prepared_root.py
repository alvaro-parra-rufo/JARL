"""Tests for the built-in prepared Navix root case."""

from __future__ import annotations

from pathlib import Path

from jarl.experiments.cases.builtin.navix_prepared_root import CASE
from jarl.experiments.node import NodeStatus


class TestNavixPreparedRootCase:
    def test_materializes_prepared_root_with_stable_alias(self, tmp_path: Path) -> None:
        context = CASE.materialize(tmp_path / "prepared")

        graph = context.reload_graph()
        root = graph.get_node(context.node_id("root"))
        config = graph.resolve_config(root)

        assert graph.as_networkx().number_of_nodes() == 1
        assert root.status is NodeStatus.PREPARED
        assert root.branch == "main"
        assert graph.current_node.id == root.id
        assert config.environment.env_id == "Navix-Empty-5x5-v0"
