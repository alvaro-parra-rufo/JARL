"""Tests for extend-chain metric concatenation."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarl.config import BaseConfig
from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.metrics import concatenate_metric_snapshots, extend_chain_nodes
from jarl.experiments.node import NodeMetadata, NodeWorkspace
from tests.unit.test_experiments.test_metrics.test_snapshot import build_metrics_snapshot_from_steps


class TestLineageConcat:
    """Extend-chain metric concatenation."""

    def test_concatenate_offsets_child_steps_by_parent_checkpoint(self) -> None:
        parent_ws = NodeWorkspace(
            node_dir=Path("parent"),
            node_metadata=NodeMetadata(id="parent", branch="main"),
        )
        child_meta = NodeMetadata(id="child", branch="main", parent_id="parent", parent_checkpoint_step=98304)
        child_ws = NodeWorkspace(
            node_dir=Path("child"),
            node_metadata=child_meta,
            parent_checkpoint_step=98304,
        )
        parent_snap = build_metrics_snapshot_from_steps(
            {
                1024: {"rollout/episode_return": 0.1},
                98304: {"rollout/episode_return": 0.9},
            }
        )
        child_snap = build_metrics_snapshot_from_steps(
            {
                1024: {"rollout/episode_return": 0.95},
                49152: {"rollout/episode_return": 1.0},
            }
        )

        merged = concatenate_metric_snapshots([(parent_ws, parent_snap), (child_ws, child_snap)])

        assert 1024 in merged.by_step
        assert 98304 in merged.by_step
        assert 99328 in merged.by_step  # 98304 + 1024
        assert merged.by_step[99328]["rollout/episode_return"] == pytest.approx(0.95)
        assert merged.latest["rollout/episode_return"] == pytest.approx(1.0)
        assert merged.by_step[147456]["rollout/episode_return"] == pytest.approx(1.0)

    def test_extend_chain_nodes_keeps_same_branch_suffix(self, tmp_path: Path) -> None:
        class SampleConfig(BaseConfig):
            learning_rate: float = 0.1

        graph: ExperimentGraph[SampleConfig] = ExperimentGraph(tmp_path / "exp")
        root = graph.create_root(config=SampleConfig(), branch="main", label="base")
        mid = graph.extend("main", label="mid")
        graph.fork(branch="lr_fork", from_node=mid, label="fork_lr")

        chain = extend_chain_nodes(graph, mid.id)

        assert [workspace.id for workspace in chain] == [root.id, mid.id]
