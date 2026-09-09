"""Execute and persist one analyzed rollout from an exact checkpoint."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np

from jarl.env_setup import setup_jax
from jarl.envs.navix.catalog import assert_transfer_compatible
from jarl.envs.navix.custom.rgb_overlay import apply_empty_variant_floor_tint
from jarl.envs.navix.telemetry import (
    NavixCaptureProfile,
    NavixRolloutSummary,
    NavixSummaryLimits,
    VideoViewMode,
)
from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.io.checkpoints import (
    CHECKPOINT_BEST_LENGTH_METRIC,
    CHECKPOINT_BEST_RETURN_METRIC,
    CheckpointRef,
    CheckpointStatus,
)
from jarl.experiments.io.rollouts import (
    WriteRolloutArtifactResult,
    compute_rollout_id,
    load_cached_rollout_artifact,
    write_rollout_artifact,
)
from jarl.experiments.node import NodeWorkspace
from jarl.inference.run import (
    CheckpointInferenceRequest,
    CheckpointInferenceResult,
    resolve_inference_horizon,
    run_checkpoint_inference,
)
from jarl.io.video import encode_rgb_video
from jarl.operations.contracts.compact import dataclass_to_compact_dict
from jarl.operations.contracts.constants import (
    CHECKPOINT_ROLLOUT_MAX_STEPS,
    CHECKPOINT_ROLLOUT_RECORD_VIDEO_DEFAULT,
    CHECKPOINT_ROLLOUT_VIDEO_VIEW_MODE_DEFAULT,
)
from jarl.training.config import RLRunConfig

_SUMMARY_LIMITS = NavixSummaryLimits(
    max_action_segments=64,
    max_events=64,
    max_visibility_intervals=32,
    max_entity_transitions=32,
    max_path_positions=128,
)
"""Payload limits for deterministic rollout summaries."""

__all__ = [
    "CheckpointRolloutRequest",
    "CheckpointRolloutResponse",
    "checkpoint_rollout",
]


@dataclass(frozen=True, slots=True)
class CheckpointRolloutRequest:
    """Inputs for one deterministic checkpoint rollout."""

    checkpoint_step: int
    seed: int
    node_id: str | None = None
    env_id: str | None = None
    max_steps: int | None = None
    record_video: bool = CHECKPOINT_ROLLOUT_RECORD_VIDEO_DEFAULT
    video_view_mode: VideoViewMode = CHECKPOINT_ROLLOUT_VIDEO_VIEW_MODE_DEFAULT


@dataclass(frozen=True, slots=True)
class CheckpointRolloutResponse:
    """Persisted rollout identity and compact deterministic analysis."""

    node_id: str
    rollout_id: str
    checkpoint_step: int
    seed: int
    env_id: str
    algorithm_name: str
    cache_hit: bool
    summary: NavixRolloutSummary
    checkpoint_eval: dict[str, float] | None = None
    rollout_seconds: float | None = None
    transfer_seconds: float | None = None

    _compact_include: ClassVar[frozenset[str] | None] = None

    def to_compact_dict(self) -> dict[str, object]:
        """Return the bounded summary without trace or RGB arrays."""
        payload = dataclass_to_compact_dict(
            self.summary,
            include=self._compact_include,
        )
        payload["cache_hit"] = self.cache_hit
        if self.checkpoint_eval:
            payload["checkpoint_eval"] = dict(self.checkpoint_eval)
        if self.rollout_seconds is not None:
            payload["rollout_seconds"] = self.rollout_seconds
        if self.transfer_seconds is not None:
            payload["transfer_seconds"] = self.transfer_seconds
        return payload


def checkpoint_rollout(
    graph: ExperimentGraph[RLRunConfig],
    request: CheckpointRolloutRequest,
) -> CheckpointRolloutResponse:
    """Execute or reuse one analyzed rollout from an exact node checkpoint."""
    _validate_request(request)
    workspace = graph.get_node(request.node_id) if request.node_id else graph.current_node
    _require_saved_checkpoint(workspace, request.checkpoint_step)
    checkpoint_eval = _checkpoint_eval_snapshot(workspace, request.checkpoint_step)
    base_config = graph.resolve_config(workspace)
    target_config = _target_config(base_config, request)
    max_steps = request.max_steps if request.max_steps is not None else resolve_inference_horizon(target_config)
    _validate_max_steps(max_steps)
    profile = NavixCaptureProfile.analysis(
        record_video=request.record_video,
        view_mode=request.video_view_mode,
    )
    source_checkpoint = CheckpointRef(
        node_id=workspace.id,
        checkpoint_step=request.checkpoint_step,
    )
    effective_config: dict[str, object] = target_config.model_dump(mode="json")
    rollout_id = compute_rollout_id(
        source_checkpoint=source_checkpoint,
        target_node_id=workspace.id,
        algorithm_name=target_config.algorithm.name,
        env_id=target_config.environment.env_id,
        seed=request.seed,
        max_steps=max_steps,
        capture_profile=profile,
        effective_config=effective_config,
    )
    cached = load_cached_rollout_artifact(
        workspace,
        rollout_id,
        require_video=request.record_video,
    )
    if cached is not None:
        return _response_from_write(cached, checkpoint_eval=checkpoint_eval)

    setup_jax(target_config.jax)
    inference = run_checkpoint_inference(
        workspace,
        target_config,
        CheckpointInferenceRequest(
            checkpoint_step=request.checkpoint_step,
            seeds=(request.seed,),
            capture_profile=profile,
            max_steps=max_steps,
            summarize=True,
            summary_limits=_SUMMARY_LIMITS,
        ),
    )
    if len(inference.summaries) != 1:
        msg = f"Checkpoint rollout inference must return exactly one episode summary, got {len(inference.summaries)}."
        raise RuntimeError(msg)

    if request.record_video:
        with tempfile.TemporaryDirectory(prefix="jarl_checkpoint_rollout_") as temp_dir:
            video_path = _encode_rollout_video(
                inference,
                Path(temp_dir) / "rollout.mp4",
                target_config,
                view_mode=request.video_view_mode,
            )
            written = write_rollout_artifact(
                workspace,
                rollout_id=rollout_id,
                source_checkpoint=source_checkpoint,
                algorithm_name=target_config.algorithm.name,
                env_id=target_config.environment.env_id,
                seed=request.seed,
                max_steps=max_steps,
                capture_profile=profile,
                effective_config=effective_config,
                trace=inference.trace,
                summary=inference.summaries[0],
                video_path=video_path,
            )
    else:
        written = write_rollout_artifact(
            workspace,
            rollout_id=rollout_id,
            source_checkpoint=source_checkpoint,
            algorithm_name=target_config.algorithm.name,
            env_id=target_config.environment.env_id,
            seed=request.seed,
            max_steps=max_steps,
            capture_profile=profile,
            effective_config=effective_config,
            trace=inference.trace,
            summary=inference.summaries[0],
        )
    return _response_from_write(
        written,
        checkpoint_eval=checkpoint_eval,
        rollout_seconds=inference.rollout_seconds,
        transfer_seconds=inference.transfer_seconds,
    )


def _validate_request(request: CheckpointRolloutRequest) -> None:
    if request.checkpoint_step < 0:
        msg = f"checkpoint_step must be >= 0, got {request.checkpoint_step}."
        raise ValueError(msg)
    if request.max_steps is not None:
        _validate_max_steps(request.max_steps)


def _validate_max_steps(max_steps: int) -> None:
    if max_steps <= 0:
        msg = f"max_steps must be > 0, got {max_steps}."
        raise ValueError(msg)
    if max_steps > CHECKPOINT_ROLLOUT_MAX_STEPS:
        msg = f"max_steps must be <= {CHECKPOINT_ROLLOUT_MAX_STEPS}, got {max_steps}."
        raise ValueError(msg)


def _require_saved_checkpoint(
    workspace: NodeWorkspace,
    checkpoint_step: int,
) -> None:
    record = next(
        (item for item in workspace.list_checkpoints() if item.checkpoint_step == checkpoint_step),
        None,
    )
    if record is None:
        msg = f"No checkpoint record registered for step {checkpoint_step} on node {workspace.id!r}."
        raise ValueError(msg)
    if record.status != CheckpointStatus.SAVED:
        msg = (
            f"Checkpoint step {checkpoint_step} on node {workspace.id!r} is not saved (status={record.status.value!r})."
        )
        raise ValueError(msg)


def _target_config(
    base_config: RLRunConfig,
    request: CheckpointRolloutRequest,
) -> RLRunConfig:
    target_env_id = request.env_id or base_config.environment.env_id
    if target_env_id != base_config.environment.env_id:
        assert_transfer_compatible(
            base_config.environment.env_id,
            target_env_id,
        )
    overrides: dict[str, object] = {
        "environment.env_id": target_env_id,
        "environment.seed": request.seed,
    }
    if request.max_steps is not None:
        overrides["environment.max_episode_steps"] = request.max_steps
    return base_config.apply_overrides(overrides)


def _encode_rollout_video(
    inference: CheckpointInferenceResult,
    destination: Path,
    config: RLRunConfig,
    *,
    view_mode: VideoViewMode,
) -> Path:
    frames = inference.trace.rgb_frames
    if frames is None:
        msg = "Video rollout inference did not capture RGB frames."
        raise RuntimeError(msg)
    array = np.asarray(frames)
    if array.shape[0] != 1:
        msg = f"Checkpoint rollout video requires one episode, got batch size {array.shape[0]}."
        raise RuntimeError(msg)
    episode_length = int(np.asarray(inference.trace.episode_lengths)[0])
    frame_count = max(1, min(int(array.shape[1]), episode_length + 1))
    episode_frames = np.asarray(array[0, :frame_count])
    player_positions = inference.trace.player_positions
    if player_positions is not None:
        player_positions = np.asarray(player_positions)[0, :frame_count]
    episode_frames = apply_empty_variant_floor_tint(
        episode_frames,
        env_id=config.environment.env_id,
        view_mode=view_mode,
        player_positions=player_positions,
    )
    return encode_rgb_video(
        episode_frames,
        destination,
        fps=int(config.video.video_fps),
        scale=int(config.video.video_scale),
    )


def _checkpoint_eval_snapshot(
    workspace: NodeWorkspace,
    checkpoint_step: int,
) -> dict[str, float] | None:
    """Return eval metrics captured when the checkpoint was saved."""
    record = next(
        (item for item in workspace.list_checkpoints() if item.checkpoint_step == checkpoint_step),
        None,
    )
    if record is None:
        return None
    snapshot: dict[str, float] = {}
    raw_return = record.metrics.get(CHECKPOINT_BEST_RETURN_METRIC)
    if raw_return is not None:
        snapshot["episode_return"] = float(raw_return)
    raw_length = record.metrics.get(CHECKPOINT_BEST_LENGTH_METRIC)
    if raw_length is not None:
        snapshot["episode_length"] = float(raw_length)
    return snapshot or None


def _response_from_write(
    written: WriteRolloutArtifactResult,
    *,
    checkpoint_eval: dict[str, float] | None = None,
    rollout_seconds: float | None = None,
    transfer_seconds: float | None = None,
) -> CheckpointRolloutResponse:
    manifest = written.manifest
    return CheckpointRolloutResponse(
        node_id=manifest.target_node_id,
        rollout_id=manifest.rollout_id,
        checkpoint_step=manifest.source_checkpoint.checkpoint_step,
        seed=manifest.seed,
        env_id=manifest.env_id,
        algorithm_name=manifest.algorithm_name,
        cache_hit=written.cache_hit,
        summary=manifest.summary,
        checkpoint_eval=checkpoint_eval,
        rollout_seconds=rollout_seconds,
        transfer_seconds=transfer_seconds,
    )
