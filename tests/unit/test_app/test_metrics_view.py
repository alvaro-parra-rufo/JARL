"""App-specific metrics presentation tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from jarl.app.lib.metrics_view import (
    build_multi_node_frame,
    build_multi_node_latest_table,
    comparison_metric_options,
    default_metric_choices,
    default_metric_prefixes,
    downsample_frame,
    metrics_file_cache_key,
    preset_metric_names,
    snapshot_to_wide_frame,
    tail_frame,
)
from jarl.experiments.io.metrics import JsonlMetricWriter
from jarl.experiments.metrics import MetricsSnapshot


def build_metrics_snapshot_from_steps(by_step: dict[int, dict[str, float]]) -> MetricsSnapshot:
    latest: dict[str, tuple[int, float]] = {}
    names: set[str] = set()
    record_count = 0
    for step, metrics in by_step.items():
        for name, value in metrics.items():
            record_count += 1
            names.add(name)
            current = latest.get(name)
            if current is None or step >= current[0]:
                latest[name] = (step, value)
    return MetricsSnapshot(
        latest={name: value for name, (_step, value) in latest.items()},
        names=tuple(sorted(names)),
        record_count=record_count,
        by_step=by_step,
    )


class TestSnapshotToWideFrame:
    """Pivot helpers for chart rendering."""

    def test_filters_columns_without_full_pivot(self) -> None:
        snapshot = build_metrics_snapshot_from_steps(
            {
                1: {"rollout/episode_return": 0.1, "loss/policy_gradient_loss": 0.5},
                2: {"rollout/episode_return": 0.2, "eval/episode_return": 0.9},
            }
        )

        frame = snapshot_to_wide_frame(snapshot, ["rollout/episode_return"])

        assert list(frame.columns) == ["rollout/episode_return"]
        assert frame.loc[2, "rollout/episode_return"] == pytest.approx(0.2)


class TestMetricPresets:
    """Preset name selection."""

    def test_preset_metric_names_filters_available(self) -> None:
        available = {"rollout/episode_return", "time/sps"}

        names = preset_metric_names("Train", available)

        assert names == ["rollout/episode_return", "time/sps"]

    def test_default_metric_choices_from_latest_mapping(self) -> None:
        latest = {
            "rollout/episode_return": 1.0,
            "eval/episode_return": 0.5,
            "custom/extra": 3.0,
        }

        selected = default_metric_choices(latest)

        assert selected == ["rollout/episode_return", "eval/episode_return"]
        assert len(selected) <= 5

    def test_default_metric_prefixes_prefers_core_groups(self) -> None:
        prefixes = ["wandb", "rollout", "eval", "loss", "custom"]

        assert default_metric_prefixes(prefixes) == ["rollout", "eval", "loss"]


class TestFrameLimits:
    """Downsampling and tail helpers."""

    def test_downsample_preserves_last_row(self) -> None:
        frame = pd.DataFrame({"a": range(10)}, index=range(10))

        sampled = downsample_frame(frame, max_rows=4)

        assert sampled.index[-1] == 9

    def test_tail_frame_limits_rows(self) -> None:
        frame = pd.DataFrame({"a": range(10)}, index=range(10))

        tailed = tail_frame(frame, max_rows=3)

        assert len(tailed) == 3
        assert tailed.index[0] == 7


class TestMultiNodeComparison:
    """Multi-node chart and table helpers."""

    def test_build_multi_node_frame_uses_node_columns(self) -> None:
        snap_a = build_metrics_snapshot_from_steps(
            {
                1: {"rollout/episode_return": 0.1},
                2: {"rollout/episode_return": 0.2},
            }
        )
        snap_b = build_metrics_snapshot_from_steps(
            {
                1: {"rollout/episode_return": 0.05},
                3: {"rollout/episode_return": 0.3},
            }
        )

        frame = build_multi_node_frame({"node_a": snap_a, "node_b": snap_b}, ["rollout/episode_return"])

        assert list(frame.columns) == ["node_a", "node_b"]
        assert frame.loc[2, "node_a"] == pytest.approx(0.2)
        assert frame.loc[3, "node_b"] == pytest.approx(0.3)

    def test_build_multi_node_latest_table(self) -> None:
        snap_a = build_metrics_snapshot_from_steps({2: {"eval/episode_return": 0.8}})
        snap_b = build_metrics_snapshot_from_steps({2: {"eval/episode_return": 0.6}})

        table = build_multi_node_latest_table(
            {"a": snap_a, "b": snap_b},
            ["eval/episode_return"],
        )

        assert table.set_index("node").loc["a", "eval/episode_return"] == pytest.approx(0.8)

    def test_comparison_metric_options_prefers_returns(self) -> None:
        snap = build_metrics_snapshot_from_steps(
            {1: {"custom/x": 1.0, "rollout/episode_return": 0.1}},
        )

        options = comparison_metric_options({"n": snap})

        assert options[0] == "rollout/episode_return"


class TestMetricsFileCacheKey:
    """Cache key tracks file revisions."""

    def test_changes_when_file_grows(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        path.write_text(json.dumps({"step": 1, "name": "loss", "value": 0.1}) + "\n", encoding="utf-8")
        first = metrics_file_cache_key(path)

        with JsonlMetricWriter(path) as writer:
            writer.log_scalar(2, "loss", 0.2)

        second = metrics_file_cache_key(path)

        assert first != second
