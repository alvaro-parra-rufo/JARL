"""App-specific inference presentation tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from jarl.app.lib.inference_view import inference_timing_metrics, video_metrics_for_prefix
from jarl.experiments.node import NodeMetadata, NodeStatus, NodeWorkspace


class TestInferenceTiming:
    """Timing aggregation for result panel."""

    def test_inference_timing_metrics_sums_encode_per_episode(self) -> None:
        frame = pd.DataFrame(
            [
                {"rollout_seconds": 1.5, "transfer_seconds": 0.2, "encode_seconds": 0.4},
                {"rollout_seconds": 1.5, "transfer_seconds": 0.2, "encode_seconds": 0.6},
            ]
        )

        timing = inference_timing_metrics(frame)

        assert timing["rollout_s"] == 1.5
        assert timing["transfer_s"] == 0.2
        assert timing["encode_s"] == 1.0
        assert timing["total_s"] == pytest.approx(2.7)


class TestInferenceArtifactsPresentation:
    """Pandas wrappers for video metrics lookup."""

    def test_video_metrics_for_prefix_returns_dataframe(self, tmp_path: Path) -> None:
        workspace = _workspace_with_video_metrics(tmp_path)
        metrics_path = workspace.video_metrics_jsonl_path
        rows = [
            {
                "status": "completed",
                "video_file": str(workspace.videos_dir / "infer_ckpt_1_episode_01.mp4"),
                "episode_return": 0.5,
                "role": "inference",
            },
        ]
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

        filtered = video_metrics_for_prefix(workspace, "infer_ckpt_1")

        assert len(filtered) == 1
        assert filtered.iloc[0]["episode_return"] == 0.5


def _workspace_with_video_metrics(tmp_path: Path) -> NodeWorkspace:
    node_dir = tmp_path / "nodes" / "node_a"
    metadata = NodeMetadata(id="node_a", branch="main", label="baseline", status=NodeStatus.COMPLETED)
    return NodeWorkspace(node_dir=node_dir, node_metadata=metadata)
