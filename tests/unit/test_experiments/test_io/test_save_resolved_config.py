"""Tests for resolved config persistence."""

from __future__ import annotations

from pathlib import Path

from jarl.config import BaseConfig
from jarl.experiments.io.config import save_resolved_config
from jarl.experiments.node import NodeMetadata, NodeWorkspace


class SampleConfig(BaseConfig):
    """Concrete config for IO config tests."""

    learning_rate: float = 0.1


class TestSaveResolvedConfig:
    """Tests for explicit resolved-config persistence."""

    def test_save_resolved_config_writes_config_json(self, tmp_path: Path) -> None:
        node_dir = tmp_path / "nodes" / "main_test_abc12345"
        workspace = NodeWorkspace(node_dir=node_dir, node_metadata=NodeMetadata(id="main_test_abc12345"))
        config = SampleConfig(learning_rate=0.05)

        path = workspace.save_resolved_config(config)

        assert path == workspace.config_path
        assert SampleConfig.load(path).learning_rate == 0.05

    def test_save_resolved_config_does_not_require_graph(self, tmp_path: Path) -> None:
        config = SampleConfig(learning_rate=0.02)
        target = tmp_path / "config.json"

        written = save_resolved_config(target, config)

        assert written == target
        assert SampleConfig.load(target).learning_rate == 0.02
