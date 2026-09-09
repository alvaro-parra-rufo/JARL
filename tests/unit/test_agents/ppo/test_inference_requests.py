"""Tests for ``jarl.agents.ppo.inference.requests``."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from jarl.agents.ppo.inference.requests import (
    InferenceLaunchRequest,
    build_inference_command,
    default_inference_name_prefix,
    inference_env_options,
    inference_episodes_default,
    inference_horizon_default,
    jsonl_optional_int,
    latest_inference_job,
    list_inference_jobs,
    name_prefix_from_video_file,
    node_max_episode_steps,
    sanitize_name_prefix,
    video_metrics_for_prefix,
    videos_for_prefix,
)
from jarl.experiments.node import NodeMetadata, NodeStatus, NodeWorkspace


class TestInferenceLaunchRequest:
    """Command builder and defaults."""

    def test_build_inference_command_includes_optional_overrides(self, tmp_path: Path) -> None:
        request = InferenceLaunchRequest(
            node_id="node_a",
            checkpoint_step=128,
            env_id="Navix-Empty-5x5-v0",
            seed=7,
            name_prefix="infer_ckpt_128",
            max_episode_steps=250,
            final_video_episodes=3,
        )

        command = build_inference_command(
            request,
            experiment_dir=tmp_path / "exp",
            inference_module="jarl.agents.ppo.inference",
            python_executable="/usr/bin/python",
        )

        assert "--node-id" in command
        assert "--seed" in command
        assert command[command.index("--seed") + 1] == "7"
        assert command[command.index("--name-prefix") + 1] == "infer_ckpt_128"
        assert command[command.index("--max-episode-steps") + 1] == "250"
        assert command[command.index("--final-video-episodes") + 1] == "3"

    def test_build_inference_command_omits_horizon_when_unset(self, tmp_path: Path) -> None:
        request = InferenceLaunchRequest(
            node_id="node_a",
            checkpoint_step=128,
            env_id="Navix-Empty-5x5-v0",
            seed=7,
            name_prefix="infer_ckpt_128",
        )

        command = build_inference_command(
            request,
            experiment_dir=tmp_path / "exp",
            inference_module="jarl.agents.ppo.inference",
            python_executable="/usr/bin/python",
        )

        assert "--max-episode-steps" not in command
        assert command[command.index("--final-video-episodes") + 1] == "5"

    def test_inference_horizon_default_uses_node_or_fallback(self) -> None:
        from jarl.training.config import EnvironmentConfig, RLRunConfig

        assert inference_horizon_default(RLRunConfig()) == 100
        assert inference_horizon_default(RLRunConfig(environment=EnvironmentConfig(max_episode_steps=64))) == 64

    def test_inference_episodes_default(self) -> None:
        assert inference_episodes_default() == 5

    def test_default_inference_name_prefix(self) -> None:
        assert default_inference_name_prefix(256) == "infer_ckpt_256"

    def test_sanitize_name_prefix(self) -> None:
        assert sanitize_name_prefix(" infer ckpt/1 ") == "infer_ckpt_1"

    def test_node_max_episode_steps_reads_config(self) -> None:
        from jarl.training.config import EnvironmentConfig, RLRunConfig

        assert node_max_episode_steps(RLRunConfig()) == 0
        assert node_max_episode_steps(RLRunConfig(environment=EnvironmentConfig(max_episode_steps=64))) == 64

    def test_inference_env_options_include_node_default(self) -> None:
        from jarl.envs.navix.catalog import list_registered_maps

        registered = [contract.env_id for contract in list_registered_maps()]
        options = inference_env_options(node_env_id="Navix-Empty-5x5-v0")

        assert options == tuple(registered)
        assert "Navix-DoorKey-8x8-v0" in options

    def test_inference_env_options_unknown_node_env_is_first(self) -> None:
        from jarl.envs.navix.catalog import list_registered_maps

        registered = [contract.env_id for contract in list_registered_maps()]
        options = inference_env_options(node_env_id="Navix-Custom-Unknown-v0")

        assert options[0] == "Navix-Custom-Unknown-v0"
        assert options[1:] == tuple(registered)

    def test_name_prefix_from_video_file(self) -> None:
        assert name_prefix_from_video_file("videos/infer_ckpt_1_episode_01.mp4") == "infer_ckpt_1"
        assert name_prefix_from_video_file("other.mp4") is None


class TestLatestInferenceJob:
    """Disk-backed job discovery."""

    def test_latest_inference_job_reads_last_inference_row(self, tmp_path: Path) -> None:
        workspace = _workspace_with_video_metrics(tmp_path)
        metrics_path = workspace.video_metrics_jsonl_path
        rows = [
            {
                "status": "completed",
                "video_file": str(workspace.videos_dir / "infer_ckpt_1_episode_01.mp4"),
                "role": "final",
            },
            {
                "status": "completed",
                "video_file": str(workspace.videos_dir / "infer_ckpt_9_episode_01.mp4"),
                "role": "inference",
                "checkpoint_step": 9,
            },
        ]
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

        job = latest_inference_job(workspace)

        assert job is not None
        assert job["name_prefix"] == "infer_ckpt_9"
        assert job["checkpoint_step"] == 9

    def test_latest_inference_job_ignores_nan_checkpoint_step(self, tmp_path: Path) -> None:
        workspace = _workspace_with_video_metrics(tmp_path)
        metrics_path = workspace.video_metrics_jsonl_path
        rows = [
            {
                "status": "completed",
                "video_file": str(workspace.videos_dir / "infer_ckpt_3_episode_01.mp4"),
                "role": "inference",
                "checkpoint_step": None,
            },
        ]
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

        job = latest_inference_job(workspace)

        assert job is not None
        assert job["name_prefix"] == "infer_ckpt_3"
        assert "checkpoint_step" not in job

    def test_list_inference_jobs_newest_first_and_skips_train_videos(self, tmp_path: Path) -> None:
        workspace = _workspace_with_video_metrics(tmp_path)
        workspace.videos_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = workspace.video_metrics_jsonl_path
        rows = [
            {
                "status": "completed",
                "video_file": str(workspace.videos_dir / "infer_ckpt_1_episode_01.mp4"),
                "role": "inference",
                "checkpoint_step": 1,
                "env_id": "Navix-Empty-5x5-v0",
            },
            {
                "status": "completed",
                "video_file": str(workspace.videos_dir / "train_episode_01.mp4"),
                "role": "final",
            },
            {
                "status": "completed",
                "video_file": str(workspace.videos_dir / "infer_ckpt_9_episode_01.mp4"),
                "role": "inference",
                "checkpoint_step": 9,
            },
        ]
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        (workspace.videos_dir / "infer_ckpt_orphan_episode_01.mp4").write_bytes(b"mp4")

        jobs = list_inference_jobs(workspace)

        assert [job["name_prefix"] for job in jobs] == [
            "infer_ckpt_9",
            "infer_ckpt_1",
            "infer_ckpt_orphan",
        ]
        assert jobs[1]["checkpoint_step"] == 1
        assert jobs[1]["env_id"] == "Navix-Empty-5x5-v0"


class TestJsonlOptionalInt:
    """Safe int coercion from video metrics rows."""

    def test_jsonl_optional_int_rejects_nan(self) -> None:
        assert jsonl_optional_int(float("nan")) is None
        assert jsonl_optional_int(128) == 128

    def test_jsonl_optional_int_rejects_pandas_na(self) -> None:
        assert jsonl_optional_int(pd.NA) is None


class TestInferenceArtifacts:
    """Video metrics and MP4 lookup by prefix."""

    def test_video_metrics_for_prefix_filters_rows(self, tmp_path: Path) -> None:
        workspace = _workspace_with_video_metrics(tmp_path)
        metrics_path = workspace.video_metrics_jsonl_path
        rows = [
            {
                "status": "completed",
                "video_file": str(workspace.videos_dir / "infer_ckpt_1_episode_01.mp4"),
                "episode_return": 0.5,
                "role": "inference",
            },
            {
                "status": "completed",
                "video_file": str(workspace.videos_dir / "train_episode_01.mp4"),
                "episode_return": 0.1,
                "role": "final",
            },
        ]
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        workspace.videos_dir.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        (workspace.videos_dir / "infer_ckpt_1_episode_01.mp4").write_bytes(b"mp4")
        (workspace.videos_dir / "train_episode_01.mp4").write_bytes(b"mp4")

        filtered = video_metrics_for_prefix(workspace, "infer_ckpt_1")

        assert len(filtered) == 1
        assert filtered[0]["episode_return"] == 0.5

    def test_videos_for_prefix(self, tmp_path: Path) -> None:
        workspace = _workspace_with_video_metrics(tmp_path)
        workspace.videos_dir.mkdir(parents=True, exist_ok=True)
        first = workspace.videos_dir / "infer_ckpt_2_episode_01.mp4"
        second = workspace.videos_dir / "infer_ckpt_2_episode_02.mp4"
        first.write_bytes(b"a")
        second.write_bytes(b"b")
        (workspace.videos_dir / "other_episode_01.mp4").write_bytes(b"c")

        paths = videos_for_prefix(workspace, "infer_ckpt_2")

        assert paths == [first, second]


def _workspace_with_video_metrics(tmp_path: Path) -> NodeWorkspace:
    node_dir = tmp_path / "nodes" / "node_a"
    metadata = NodeMetadata(id="node_a", branch="main", label="baseline", status=NodeStatus.COMPLETED)
    return NodeWorkspace(node_dir=node_dir, node_metadata=metadata)
