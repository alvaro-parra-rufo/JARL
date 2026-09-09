"""Tests for DAG visualization helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

from jarl.config import BaseConfig
from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.viz import plot_dag


class SampleConfig(BaseConfig):
    """Concrete config for viz tests."""

    value: float = 1.0


@pytest.fixture()
def graph(tmp_path: Path) -> ExperimentGraph[SampleConfig]:
    """Graph with metrics on two nodes."""
    exp_dir = tmp_path / "exp"
    g: ExperimentGraph[SampleConfig] = ExperimentGraph(exp_dir)
    root = g.create_root(config=SampleConfig(), branch="main")
    step2 = g.extend("main", label="step2")
    root.save_metrics({"test_accuracy": 0.5})
    step2.save_metrics({"test_accuracy": 0.9})
    return g


class TestPlotDag:
    """Tests for plot_dag."""

    def test_returns_figure(self, graph: ExperimentGraph[SampleConfig]) -> None:
        result = plot_dag(graph)

        assert isinstance(result, Figure)
        plt.close(result)

    def test_empty_graph_raises(self, tmp_path: Path) -> None:
        graph: ExperimentGraph[SampleConfig] = ExperimentGraph(tmp_path / "empty")

        with pytest.raises(ValueError, match="no nodes"):
            plot_dag(graph)

    def test_plot_on_existing_axes(self, graph: ExperimentGraph[SampleConfig]) -> None:
        _, ax = plt.subplots()

        result = plot_dag(graph, ax=ax)

        assert result is ax
        plt.close(ax.figure)
