"""Shared fixtures for MCP stdio integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarl.experiments.graph import ExperimentGraph
from jarl.training.config import RLRunConfig


@pytest.fixture()
def prepared_experiment(tmp_path: Path) -> Path:
    """Experiment directory with one prepared root node."""
    exp_dir = tmp_path / "exp"
    graph = ExperimentGraph(exp_dir, base_config=RLRunConfig())
    graph.create_root(RLRunConfig(), branch="main", label="root", prepare=True)
    graph.save()
    return exp_dir
