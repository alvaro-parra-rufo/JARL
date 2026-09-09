"""Shared fixtures for ``jarl.agentic`` tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarl.agentic.tools.context import ToolContext
from jarl.agentic.workflow import AgenticWorkflow
from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.manifest import ExperimentManifest
from jarl.training.config import RLRunConfig
from jarl.utils.io import write_text_atomic


@pytest.fixture()
def prepared_experiment(tmp_path: Path) -> Path:
    """Experiment directory with one prepared root node."""
    exp_dir = tmp_path / "exp"
    graph = ExperimentGraph(exp_dir, base_config=RLRunConfig())
    graph.create_root(RLRunConfig(), branch="main", label="root", prepare=True)
    graph.save()
    return exp_dir


@pytest.fixture()
def empty_experiment(tmp_path: Path) -> Path:
    """Experiment directory with manifest and base config but no nodes."""
    exp_dir = tmp_path / "exp"
    config = RLRunConfig()
    graph = ExperimentGraph(exp_dir, base_config=config)
    config.save(graph.layout.config_path)
    write_text_atomic(
        graph.layout.manifest_path,
        ExperimentManifest().model_dump_json(indent=2),
    )
    return exp_dir


@pytest.fixture()
def tool_context(prepared_experiment: Path) -> ToolContext:
    """Tool context bound to a prepared experiment workflow."""
    workflow = AgenticWorkflow.from_experiment(prepared_experiment)
    return workflow.build_tool_context()
