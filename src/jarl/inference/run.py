"""Common checkpoint inference orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from jarl.envs.navix.scenario_rewards import resolve_scenario_reward_spec
from jarl.envs.navix.telemetry import (
    NavixCaptureProfile,
    NavixRolloutIdentity,
    NavixRolloutSummary,
    NavixSummaryLimits,
    NavixTelemetryRollout,
    NavixTraceArrays,
    summarize_navix_trace,
)
from jarl.experiments.node import NodeWorkspace
from jarl.inference.registry import resolve_inference_backend_from_name
from jarl.training.config import RLRunConfig
from jarl.training.env_factory import navix_full_jit_env_factory

__all__ = [
    "CheckpointInferenceRequest",
    "CheckpointInferenceResult",
    "resolve_inference_horizon",
    "run_checkpoint_inference",
]


@dataclass(frozen=True, slots=True)
class CheckpointInferenceRequest:
    """Inputs for one batched checkpoint inference execution."""

    checkpoint_step: int
    seeds: tuple[int, ...]
    capture_profile: NavixCaptureProfile
    max_steps: int | None = None
    summarize: bool = True
    summary_limits: NavixSummaryLimits = field(default_factory=NavixSummaryLimits)

    def __post_init__(self) -> None:
        """Validate checkpoint, seed, and horizon inputs."""
        if self.checkpoint_step < 0:
            msg = f"checkpoint_step must be >= 0, got {self.checkpoint_step}."
            raise ValueError(msg)
        if not self.seeds:
            msg = "At least one inference seed is required."
            raise ValueError(msg)
        if self.max_steps is not None and self.max_steps <= 0:
            msg = f"max_steps must be > 0, got {self.max_steps}."
            raise ValueError(msg)
        if self.summarize and not self.capture_profile.capture_symbolic:
            msg = "Summarization requires NavixCaptureProfile.analysis()."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class CheckpointInferenceResult:
    """Host-resident trace, summaries, and checkpoint metadata."""

    trace: NavixTraceArrays
    summaries: tuple[NavixRolloutSummary, ...]
    action_names: tuple[str, ...]
    source_node_id: str
    checkpoint_step: int
    checkpoint_kind: str
    checkpoint_version: int
    algorithm_name: str
    global_step: int
    optimizer_updates: int
    seeds: tuple[int, ...]
    max_steps: int
    rollout_seconds: float
    transfer_seconds: float


def resolve_inference_horizon(target_config: RLRunConfig) -> int:
    """Return the effective Navix horizon for an inference config."""
    return int(navix_full_jit_env_factory(target_config).horizon)


def run_checkpoint_inference(
    source_workspace: NodeWorkspace,
    target_config: RLRunConfig,
    request: CheckpointInferenceRequest,
) -> CheckpointInferenceResult:
    """Restore a checkpoint and execute greedy Navix rollouts.

    The source workspace owns checkpoint bytes. The target config owns policy
    architecture, observation preprocessing, environment, and episode horizon.
    """
    policy_env = navix_full_jit_env_factory(target_config)
    backend = resolve_inference_backend_from_name(target_config.algorithm.name)
    loaded = backend(
        source_workspace,
        target_config,
        request.checkpoint_step,
        policy_env,
    )
    max_steps = request.max_steps or int(policy_env.horizon)
    environment = target_config.environment
    scenario_spec = resolve_scenario_reward_spec(
        environment.env_id,
        environment.scenario_reward_id,
        environment.scenario_reward_version,
    )
    rollout = NavixTelemetryRollout(
        environment.env_id,
        loaded.runtime,
        max_steps,
        request.capture_profile,
        max_episode_steps=int(policy_env.horizon),
        reward=environment.reward,
        scenario_spec=scenario_spec,
    )
    keys = jnp.stack([jax.random.PRNGKey(seed) for seed in request.seeds])

    rollout_start = time.perf_counter()
    device_trace = rollout.rollout(loaded.params, keys)
    jax.block_until_ready(device_trace.actions)
    rollout_seconds = time.perf_counter() - rollout_start

    transfer_start = time.perf_counter()
    trace = device_trace.to_host()
    transfer_seconds = time.perf_counter() - transfer_start

    if request.summarize:
        summaries = tuple(
            summarize_navix_trace(
                trace,
                identity=NavixRolloutIdentity(
                    env_id=target_config.environment.env_id,
                    seed=seed,
                    node_id=source_workspace.id,
                    checkpoint_step=request.checkpoint_step,
                    global_step=loaded.global_step,
                    algorithm_name=target_config.algorithm.name,
                ),
                action_names=rollout.action_names,
                episode_index=episode_index,
                limits=request.summary_limits,
            )
            for episode_index, seed in enumerate(request.seeds)
        )
    else:
        summaries = ()

    return CheckpointInferenceResult(
        trace=trace,
        summaries=summaries,
        action_names=rollout.action_names,
        source_node_id=loaded.source_node_id,
        checkpoint_step=loaded.checkpoint_step,
        checkpoint_kind=loaded.checkpoint_kind,
        checkpoint_version=loaded.checkpoint_version,
        algorithm_name=loaded.algorithm_name,
        global_step=loaded.global_step,
        optimizer_updates=loaded.optimizer_updates,
        seeds=request.seeds,
        max_steps=max_steps,
        rollout_seconds=rollout_seconds,
        transfer_seconds=transfer_seconds,
    )
