"""Tests for stale-node reconciliation on the experiment graph."""

from __future__ import annotations

from pathlib import Path

from jarl.config import BaseConfig
from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.node import NodeStatus


class SampleConfig(BaseConfig):
    """Minimal config for reconcile tests."""

    learning_rate: float = 0.01


class TestReconcileStaleNodes:
    """Tests for operator recovery of orphaned training nodes."""

    def test_dry_run_reports_without_mutating_disk(self, tmp_path: Path) -> None:
        graph: ExperimentGraph[SampleConfig] = ExperimentGraph(tmp_path / "exp")
        root = graph.create_root(config=SampleConfig(), branch="main", label="root")
        root._meta.status = NodeStatus.TRAINING
        root._save_metadata()
        graph.save()

        report = graph.reconcile_stale_nodes(dry_run=True)

        assert root.id in report.examined
        assert root.id in report.would_interrupt
        assert report.interrupted == ()
        reloaded = ExperimentGraph.from_directory(tmp_path / "exp", config_cls=SampleConfig).get_node(root.id)
        assert reloaded.status == NodeStatus.TRAINING

    def test_reconcile_marks_training_nodes_interrupted(self, tmp_path: Path) -> None:
        graph: ExperimentGraph[SampleConfig] = ExperimentGraph(tmp_path / "exp")
        root = graph.create_root(config=SampleConfig(), branch="main", label="root")
        root._meta.status = NodeStatus.TRAINING
        root._save_metadata()
        graph.save()

        report = graph.reconcile_stale_nodes(dry_run=False)

        assert root.id in report.interrupted
        reloaded = ExperimentGraph.from_directory(tmp_path / "exp", config_cls=SampleConfig).get_node(root.id)
        assert reloaded.status == NodeStatus.INTERRUPTED
