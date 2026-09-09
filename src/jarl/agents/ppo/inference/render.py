"""Render greedy Navix videos from a saved checkpoint without training."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from jarl.agents.ppo.video import AsyncNavixVideoRecorder, NavixVideoJob
from jarl.envs.navix.telemetry import NavixCaptureProfile
from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.node import NodeWorkspace
from jarl.inference.run import (
    CheckpointInferenceRequest,
    CheckpointInferenceResult,
    run_checkpoint_inference,
)
from jarl.training.config import RLRunConfig

LOGGER = logging.getLogger(__name__)

__all__ = ["render_checkpoint_greedy_video"]


def render_checkpoint_greedy_video(
    *,
    experiment_dir: Path,
    node_id: str,
    checkpoint_step: int,
    env_id: str,
    role: str,
    name_prefix: str,
    seed: int | None = None,
    max_episode_steps: int | None = None,
    final_video_episodes: int | None = None,
    load_from_node_id: str | None = None,
    use_parent_checkpoint: bool = False,
    parent_id: str | None = None,
    child_id: str | None = None,
) -> list[Path]:
    """Restore a checkpoint and encode greedy rollout videos on ``node_id`` workspace.

    Args:
        experiment_dir: Experiment directory root.
        node_id: Node workspace that receives the MP4 and ``video_metrics.jsonl`` rows.
        checkpoint_step: Orbax checkpoint step to restore.
        env_id: Navix environment used for the greedy rollout.
        role: Showcase role label stored in video metrics.
        name_prefix: Filename prefix for generated MP4 files.
        seed: Optional rollout seed override (defaults to the node config seed).
        max_episode_steps: Optional episode horizon override for greedy rollouts.
        final_video_episodes: Optional episode count override for greedy rollouts.
        load_from_node_id: Node whose ``checkpoint/`` tree is read (defaults to ``node_id``).
        use_parent_checkpoint: When ``True``, restore fork parent weights on a prepared child.
        parent_id: Optional parent node id for metric metadata.
        child_id: Optional child node id for metric metadata.

    Returns:
        Paths to encoded MP4 files.
    """
    graph: ExperimentGraph[RLRunConfig] = ExperimentGraph.from_directory(
        experiment_dir,
        config_cls=RLRunConfig,
    )
    workspace = graph.get_node(node_id)
    overrides: dict[str, object] = {"environment.env_id": env_id}
    if seed is not None:
        overrides["environment.seed"] = int(seed)
    if max_episode_steps is not None:
        overrides["environment.max_episode_steps"] = int(max_episode_steps)
    if final_video_episodes is not None:
        overrides["video.final_video_episodes"] = int(final_video_episodes)
    config = graph.resolve_config(workspace).apply_overrides(overrides)

    from jarl.env_setup import log_jax_training_devices, setup_jax

    setup_jax(config.jax)
    log_jax_training_devices(LOGGER)

    source_workspace, source_step = _resolve_checkpoint_source(
        graph,
        workspace,
        checkpoint_step=checkpoint_step,
        load_from_node_id=load_from_node_id,
        use_parent_checkpoint=use_parent_checkpoint,
    )
    episodes = max(1, int(config.video.final_video_episodes))
    base_seed = int(config.environment.seed)
    result = run_checkpoint_inference(
        source_workspace,
        config,
        CheckpointInferenceRequest(
            checkpoint_step=source_step,
            seeds=tuple(base_seed + episode_index for episode_index in range(episodes)),
            capture_profile=NavixCaptureProfile.video(config.video.video_view_mode),
            summarize=False,
        ),
    )
    return _encode_checkpoint_video(
        workspace=workspace,
        config=config,
        result=result,
        checkpoint_step=source_step,
        env_id=env_id,
        role=role,
        name_prefix=name_prefix,
        parent_id=parent_id,
        child_id=child_id,
    )


def _resolve_checkpoint_source(
    graph: ExperimentGraph[RLRunConfig],
    workspace: NodeWorkspace,
    *,
    checkpoint_step: int,
    load_from_node_id: str | None,
    use_parent_checkpoint: bool,
) -> tuple[NodeWorkspace, int]:
    """Resolve the node and step that own checkpoint bytes."""
    if use_parent_checkpoint:
        workspace_parent_id = workspace.node_metadata.parent_id
        if workspace_parent_id is None or workspace.parent_checkpoint_step is None:
            msg = f"Node {workspace.id!r} has no parent checkpoint to restore for child_start video."
            raise ValueError(msg)
        return (
            graph.get_node(workspace_parent_id),
            int(workspace.parent_checkpoint_step),
        )
    source_id = load_from_node_id or workspace.id
    return graph.get_node(source_id), checkpoint_step


def _encode_checkpoint_video(
    *,
    workspace: NodeWorkspace,
    config: RLRunConfig,
    result: CheckpointInferenceResult,
    checkpoint_step: int,
    env_id: str,
    role: str,
    name_prefix: str,
    parent_id: str | None,
    child_id: str | None,
) -> list[Path]:
    """Encode the RGB arrays captured by the common inference service."""
    frames = result.trace.rgb_frames
    if frames is None:
        msg = "Checkpoint inference did not capture RGB frames."
        raise RuntimeError(msg)
    recorder = AsyncNavixVideoRecorder(
        workspace.videos_dir,
        workspace.video_metrics_jsonl_path,
    )
    metadata = _video_metadata(
        role=role,
        checkpoint_step=checkpoint_step,
        env_id=env_id,
        parent_id=parent_id,
        child_id=child_id,
        global_step=result.global_step,
    )
    job = NavixVideoJob(
        frames=np.asarray(frames),
        episode_returns=np.asarray(result.trace.episode_returns),
        episode_lengths=np.asarray(result.trace.episode_lengths),
        global_step=result.global_step,
        name_prefix=name_prefix,
        fps=int(config.video.video_fps),
        scale=int(config.video.video_scale),
        rollout_seconds=result.rollout_seconds,
        transfer_seconds=result.transfer_seconds,
        is_final=role == "final",
        env_id=env_id,
        view_mode=config.video.video_view_mode,
        metadata=metadata,
    )
    recorder.submit_final(job)
    recorder.shutdown()
    return _videos_for_prefix(workspace, name_prefix)


def _video_metadata(
    *,
    role: str,
    checkpoint_step: int,
    env_id: str,
    parent_id: str | None,
    child_id: str | None,
    global_step: int,
) -> dict[str, Any]:
    """Build metadata persisted beside encoded checkpoint videos."""
    payload: dict[str, Any] = {
        "role": role,
        "checkpoint_step": int(checkpoint_step),
        "env_id": env_id,
        "global_step": int(global_step),
    }
    if parent_id is not None:
        payload["parent_id"] = parent_id
    if child_id is not None:
        payload["child_id"] = child_id
    return payload


def _videos_for_prefix(
    workspace: NodeWorkspace,
    name_prefix: str,
) -> list[Path]:
    """Return encoded videos matching one filename prefix."""
    if not workspace.videos_dir.is_dir():
        return []
    return sorted(workspace.videos_dir.glob(f"{name_prefix}_episode_*.mp4"))
