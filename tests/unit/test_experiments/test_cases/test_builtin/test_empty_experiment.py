"""Tests for the built-in empty experiment case."""

from __future__ import annotations

from pathlib import Path

from jarl.experiments.cases.builtin.empty_experiment import CASE
from jarl.experiments.manifest import MANIFEST_FILENAME


class TestEmptyExperimentCase:
    def test_materializes_valid_empty_graph(self, tmp_path: Path) -> None:
        context = CASE.materialize(tmp_path / "empty")

        graph = context.reload_graph()

        assert graph.as_networkx().number_of_nodes() == 0
        assert graph.layout.config_path.is_file()
        assert (context.experiment_dir / MANIFEST_FILENAME).is_file()
        assert dict(context.aliases) == {}
