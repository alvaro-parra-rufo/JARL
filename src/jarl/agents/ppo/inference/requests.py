"""Inference launch contracts and disk discovery helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from math import isnan
from pathlib import Path
from typing import Any

from jarl.envs.navix.catalog import list_registered_maps
from jarl.experiments.node import NodeWorkspace
from jarl.training.config import RLRunConfig

DEFAULT_INFERENCE_ROLE = "inference"
INFERENCE_LOG_NAME = "inference_runner_lab.log"
DEFAULT_INFERENCE_HORIZON = 100
DEFAULT_INFERENCE_EPISODES = 5
_VIDEO_PREFIX_PATTERN = re.compile(r"^(?P<prefix>.+)_episode_\d+\.mp4$")

__all__ = [
    "DEFAULT_INFERENCE_EPISODES",
    "DEFAULT_INFERENCE_HORIZON",
    "DEFAULT_INFERENCE_ROLE",
    "INFERENCE_LOG_NAME",
    "InferenceLaunchRequest",
    "build_inference_command",
    "default_inference_name_prefix",
    "inference_env_options",
    "inference_episodes_default",
    "inference_horizon_default",
    "inference_run_config_summary",
    "jsonl_optional_int",
    "latest_inference_job",
    "list_inference_jobs",
    "load_video_metrics_records",
    "name_prefix_from_video_file",
    "node_max_episode_steps",
    "sanitize_name_prefix",
    "video_metrics_for_prefix",
    "videos_for_prefix",
]


@dataclass(frozen=True, slots=True)
class InferenceLaunchRequest:
    """Payload for launching a greedy checkpoint video job."""

    node_id: str
    checkpoint_step: int
    env_id: str
    seed: int
    name_prefix: str
    max_episode_steps: int | None = None
    final_video_episodes: int = DEFAULT_INFERENCE_EPISODES
    role: str = DEFAULT_INFERENCE_ROLE


def sanitize_name_prefix(value: str) -> str:
    """Return a filesystem-safe prefix for generated MP4 files."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "infer"


def default_inference_name_prefix(checkpoint_step: int) -> str:
    """Return the default MP4 prefix for an inference job."""
    return f"infer_ckpt_{checkpoint_step}"


def build_inference_command(
    request: InferenceLaunchRequest,
    *,
    experiment_dir: Path,
    inference_module: str,
    python_executable: str,
) -> list[str]:
    """Build the subprocess argv for ``python -m jarl.agents.ppo.inference``."""
    command = [
        python_executable,
        "-m",
        inference_module,
        "--experiment-dir",
        str(experiment_dir),
        "--node-id",
        request.node_id,
        "--checkpoint-step",
        str(request.checkpoint_step),
        "--env-id",
        request.env_id,
        "--role",
        request.role,
        "--name-prefix",
        request.name_prefix,
        "--seed",
        str(request.seed),
    ]
    if request.max_episode_steps is not None:
        command.extend(["--max-episode-steps", str(request.max_episode_steps)])
    command.extend(["--final-video-episodes", str(request.final_video_episodes)])
    return command


def node_max_episode_steps(config: RLRunConfig) -> int:
    """Return ``environment.max_episode_steps`` from the node config (0 when unset)."""
    value = config.environment.max_episode_steps
    return int(value) if value is not None else 0


def inference_horizon_default(config: RLRunConfig) -> int:
    """Return the horizon prefilled in the inference form."""
    node_horizon = node_max_episode_steps(config)
    return node_horizon if node_horizon > 0 else DEFAULT_INFERENCE_HORIZON


def inference_episodes_default() -> int:
    """Return the episode count prefilled in the inference form."""
    return DEFAULT_INFERENCE_EPISODES


def inference_env_options(*, node_env_id: str) -> tuple[str, ...]:
    """Return registered Navix env ids, always including the node default."""
    registered = [contract.env_id for contract in list_registered_maps()]
    if not registered:
        return (node_env_id,)
    if node_env_id in registered:
        return tuple(registered)
    return (node_env_id, *registered)


def load_video_metrics_records(workspace: NodeWorkspace) -> list[dict[str, Any]]:
    """Load ``video_metrics.jsonl`` rows when available."""
    path = workspace.video_metrics_jsonl_path
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def video_metrics_for_prefix(workspace: NodeWorkspace, name_prefix: str) -> list[dict[str, Any]]:
    """Return ``video_metrics.jsonl`` rows whose ``video_file`` matches ``name_prefix``."""
    return [row for row in load_video_metrics_records(workspace) if name_prefix in str(row.get("video_file", ""))]


def videos_for_prefix(workspace: NodeWorkspace, name_prefix: str) -> list[Path]:
    """Return MP4 files generated for one inference ``name_prefix``."""
    if not workspace.videos_dir.is_dir():
        return []
    return sorted(workspace.videos_dir.glob(f"{name_prefix}_episode_*.mp4"))


def name_prefix_from_video_file(video_file: str) -> str | None:
    """Extract the MP4 name prefix from a ``video_metrics.jsonl`` ``video_file`` path."""
    match = _VIDEO_PREFIX_PATTERN.match(Path(video_file).name)
    return match.group("prefix") if match is not None else None


def inference_run_config_summary(
    config: RLRunConfig,
    *,
    episode_horizon: int | None = None,
    episode_count: int | None = None,
) -> dict[str, str | int | float | bool]:
    """Return operator-facing fields used when rendering checkpoint videos."""
    horizon = episode_horizon if episode_horizon is not None else inference_horizon_default(config)
    episodes = episode_count if episode_count is not None else int(config.video.final_video_episodes)
    return {
        "algorithm": config.algorithm.name,
        "env_id": config.environment.env_id,
        "seed": config.environment.seed,
        "episode_horizon": horizon,
        "final_video_episodes": episodes,
        "video_fps": config.video.video_fps,
        "video_view_mode": config.video.video_view_mode,
        "video_scale": config.video.video_scale,
        "record_video": config.video.record_video,
    }


def jsonl_optional_int(value: object) -> int | None:
    """Convert a JSONL metric field to ``int``, ignoring null and NaN."""
    if value is None:
        return None
    try:
        import pandas as pd

        if pd.isna(value):
            return None
    except (ImportError, TypeError, ValueError):
        pass
    if isinstance(value, float) and isnan(value):
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def latest_inference_job(workspace: NodeWorkspace) -> dict[str, Any] | None:
    """Return the most recent inference job metadata found on disk for ``workspace``."""
    jobs = list_inference_jobs(workspace)
    if jobs:
        return jobs[0]
    records = load_video_metrics_records(workspace)
    for row in reversed(records):
        job = _job_from_metrics_row(workspace, row)
        if job is not None:
            return job
    return None


def list_inference_jobs(workspace: NodeWorkspace) -> list[dict[str, Any]]:
    """Return saved greedy-inference jobs for ``workspace``, newest first.

    Groups ``video_metrics.jsonl`` rows with role ``inference`` by MP4 name prefix
    and includes ``infer_*`` videos that have no metrics row yet.
    """
    records = load_video_metrics_records(workspace)
    inference_rows = [row for row in records if row.get("role") == DEFAULT_INFERENCE_ROLE]
    jobs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in reversed(inference_rows):
        job = _job_from_metrics_row(workspace, row)
        if job is None:
            continue
        name_prefix = str(job["name_prefix"])
        if name_prefix in seen:
            continue
        seen.add(name_prefix)
        jobs.append(job)
    if workspace.videos_dir.is_dir():
        for path in sorted(workspace.videos_dir.glob("infer_*_episode_*.mp4"), reverse=True):
            name_prefix = name_prefix_from_video_file(path.name)
            if name_prefix is None or name_prefix in seen:
                continue
            seen.add(name_prefix)
            jobs.append({"node_id": workspace.id, "name_prefix": name_prefix})
    return jobs


def _job_from_metrics_row(workspace: NodeWorkspace, row: dict[str, Any]) -> dict[str, Any] | None:
    """Build job metadata from one ``video_metrics.jsonl`` row, or ``None`` if unusable."""
    video_file = row.get("video_file")
    if not isinstance(video_file, str) or not video_file:
        return None
    name_prefix = name_prefix_from_video_file(video_file)
    if name_prefix is None:
        return None
    job: dict[str, Any] = {
        "node_id": workspace.id,
        "name_prefix": name_prefix,
    }
    checkpoint_step = jsonl_optional_int(row.get("checkpoint_step"))
    if checkpoint_step is not None:
        job["checkpoint_step"] = checkpoint_step
    env_id = row.get("env_id")
    if isinstance(env_id, str) and env_id:
        job["env_id"] = env_id
    return job
