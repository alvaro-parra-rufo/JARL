"""Tests for experiment case materialization contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarl.experiments.cases import ExperimentCaseContext
from jarl.experiments.cases.builtin.empty_experiment import EmptyExperimentCase
from jarl.training.config import RLRunConfig


class TestExperimentCaseContext:
    def test_registers_and_resolves_node_alias(self, tmp_path: Path) -> None:
        context = ExperimentCaseContext(tmp_path, RLRunConfig)

        context.register_alias("root", "node_123")

        assert context.node_id("root") == "node_123"
        assert dict(context.aliases) == {"root": "node_123"}

    def test_rejects_duplicate_node_alias(self, tmp_path: Path) -> None:
        context = ExperimentCaseContext(tmp_path, RLRunConfig)
        context.register_alias("root", "node_123")

        with pytest.raises(ValueError, match="already registered"):
            context.register_alias("root", "node_456")

    def test_unknown_node_alias_raises(self, tmp_path: Path) -> None:
        context = ExperimentCaseContext(tmp_path, RLRunConfig)

        with pytest.raises(KeyError):
            context.node_id("missing")


class TestExperimentCaseMaterialization:
    def test_materializes_into_new_directory(self, tmp_path: Path) -> None:
        case = EmptyExperimentCase()
        destination = tmp_path / "new"

        context = case.materialize(destination)

        assert context.experiment_dir == destination.resolve()
        assert context.reload_graph().as_networkx().number_of_nodes() == 0

    def test_materializes_into_existing_empty_directory(self, tmp_path: Path) -> None:
        case = EmptyExperimentCase()
        destination = tmp_path / "existing"
        destination.mkdir()

        context = case.materialize(destination)

        assert context.reload_graph().as_networkx().number_of_nodes() == 0

    def test_rejects_non_empty_destination_without_overwriting(self, tmp_path: Path) -> None:
        case = EmptyExperimentCase()
        destination = tmp_path / "occupied"
        destination.mkdir()
        marker = destination / "keep.txt"
        marker.write_text("keep", encoding="utf-8")

        with pytest.raises(ValueError, match="empty directory"):
            case.materialize(destination)

        assert marker.read_text(encoding="utf-8") == "keep"
