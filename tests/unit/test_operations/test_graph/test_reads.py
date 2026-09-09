"""Tests for graph read operations."""

from __future__ import annotations

import pytest

from jarl.experiments.graph import ExperimentGraph
from jarl.operations.graph.checkpoint_events import CheckpointEventsRequest, checkpoint_events
from jarl.operations.graph.diff import DiffRequest, diff
from jarl.operations.graph.metrics_series import MetricsSeriesRequest, metrics_series
from jarl.operations.graph.subtree import SubtreeRequest, subtree
from jarl.operations.graph.summary import SummaryRequest, summary
from jarl.training.config import RLRunConfig


class TestSummary:
    def test_summary_includes_config_highlights(self, prepared_root: ExperimentGraph[RLRunConfig]) -> None:
        response = summary(prepared_root, SummaryRequest())

        compact = response.to_compact_dict()

        assert compact["experiment"]["node_count"] == 1
        assert compact["nodes"][0]["config_highlights"]["env"] == RLRunConfig().environment.env_id


class TestSubtree:
    def test_subtree_returns_root_node(self, prepared_root: ExperimentGraph[RLRunConfig]) -> None:
        response = subtree(prepared_root, SubtreeRequest(depth=1))

        compact = response.to_compact_dict()

        assert compact["root_id"] == prepared_root.current_node.id
        assert len(compact["nodes"]) == 1


class TestDiff:
    def test_diff_reports_override(self, prepared_root: ExperimentGraph[RLRunConfig]) -> None:
        parent_id = prepared_root.current_node.id
        child = prepared_root.fork(
            "exp",
            from_node=prepared_root.current_node,
            config=RLRunConfig().apply_overrides({"algorithm.learning_rate": 1e-5}),
            prepare=True,
        )
        prepared_root.save()

        response = diff(prepared_root, DiffRequest(node_a=parent_id, node_b=child.id))
        compact = response.to_compact_dict()

        assert "algorithm" in compact["diff"]["changed"]

    def test_compact_diff_hides_reward_and_overlay(self) -> None:
        from jarl.envs.navix.reward_config import RewardWeightsConfig
        from jarl.experiments.summaries import config_highlights
        from jarl.operations.graph.diff import DiffResponse

        parent = RLRunConfig()
        child = parent.apply_overrides(
            {
                "algorithm.learning_rate": 1e-5,
                "environment.reward": RewardWeightsConfig(goal_reached=4.0).model_dump(),
                "environment.scenario_reward_id": "navix.floor_cell",
                "environment.scenario_reward_version": 1,
            }
        )
        response = DiffResponse(
            node_a="parent",
            node_b="child",
            diff=parent.diff(child),
            config_a_highlights=config_highlights(parent),
            config_b_highlights=config_highlights(child),
        )
        compact = response.to_compact_dict()
        dumped = str(compact)

        assert "algorithm" in compact["diff"]["changed"]
        assert "environment" not in compact["diff"]["changed"]
        assert "navix.floor_cell" not in dumped
        assert "scenario_reward_id" not in dumped
        assert "goal_reached" not in dumped
        assert "reward" not in compact["config_a_highlights"]
        assert "reward" not in compact["config_b_highlights"]


class TestMetricsSeries:
    def test_metrics_series_downsamples(self, prepared_root: ExperimentGraph[RLRunConfig]) -> None:
        workspace = prepared_root.current_node
        for step in range(20):
            workspace.log_scalar(step, "rollout/episode_return", float(step))
        prepared_root.save()

        response = metrics_series(
            prepared_root,
            MetricsSeriesRequest(
                node_id=workspace.id,
                metric_keys=["rollout/episode_return"],
                max_points=5,
            ),
        )

        series = response.series["rollout/episode_return"]

        assert len(series) == 5
        assert series[0].step == 0
        assert series[-1].step == 19

    def test_metrics_series_empty_without_file(self, prepared_root: ExperimentGraph[RLRunConfig]) -> None:
        response = metrics_series(
            prepared_root,
            MetricsSeriesRequest(node_id=prepared_root.current_node.id),
        )

        assert response.series["eval/episode_return"] == []

    def test_metrics_series_rejects_unknown_explicit_keys(self, prepared_root: ExperimentGraph[RLRunConfig]) -> None:
        with pytest.raises(ValueError, match="Unknown metric keys"):
            metrics_series(
                prepared_root,
                MetricsSeriesRequest(
                    node_id=prepared_root.current_node.id,
                    metric_keys=["rollout/episode_return"],
                ),
            )

    def test_metrics_series_rejects_invalid_key_with_catalog_hint(
        self,
        prepared_root: ExperimentGraph[RLRunConfig],
    ) -> None:
        workspace = prepared_root.current_node
        workspace.log_scalar(0, "v_value/explained_variance", 0.1)
        prepared_root.save()

        with pytest.raises(ValueError, match=r"loss/explained_variance.*v_value/explained_variance"):
            metrics_series(
                prepared_root,
                MetricsSeriesRequest(
                    node_id=workspace.id,
                    metric_keys=["loss/explained_variance"],
                ),
            )


class TestCheckpointEvents:
    def test_checkpoint_events_lists_saved_checkpoint(self, prepared_root: ExperimentGraph[RLRunConfig]) -> None:
        workspace = prepared_root.current_node
        workspace.init_checkpoint_manager(max_to_keep=None, save_interval_steps=1)
        workspace.save_checkpoint(3, {"value": 3}, metrics={"eval/episode_return": 1.0}, force=True)
        prepared_root.save()

        response = checkpoint_events(
            prepared_root,
            CheckpointEventsRequest(node_id=workspace.id),
        )

        assert response.checkpoints[0].checkpoint_step == 3
        assert response.checkpoints[0].metrics["eval/episode_return"] == 1.0
