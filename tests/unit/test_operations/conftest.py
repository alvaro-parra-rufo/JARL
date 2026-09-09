"""Shared fixtures for ``jarl.operations`` tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarl.experiments.graph import ExperimentGraph
from jarl.training.config import RLRunConfig


@pytest.fixture()
def empty_graph(tmp_path: Path) -> ExperimentGraph[RLRunConfig]:
    """Empty experiment graph bound to a temp directory."""
    return ExperimentGraph(tmp_path / "exp", base_config=RLRunConfig())


@pytest.fixture()
def prepared_root(empty_graph: ExperimentGraph[RLRunConfig]) -> ExperimentGraph[RLRunConfig]:
    """Experiment graph with one prepared root node."""
    empty_graph.create_root(RLRunConfig(), branch="main", label="baseline", prepare=True)
    empty_graph.save()
    return empty_graph
