"""Occupancy and success oracles for EmptyVariant agentic validation.

This module is a test/`validate()` helper. It is not an LLM tool surface.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from jarl.env_setup import setup_jax
from jarl.envs.navix.custom.empty_variant import CELL_ENTRY_POSITION
from jarl.envs.navix.reward_config import RewardWeightsConfig
from jarl.envs.navix.rewards import occupancy_entries, occupancy_mask
from jarl.envs.navix.scenario_rewards import resolve_scenario_reward_spec
from jarl.envs.navix.telemetry import NavixCaptureProfile, NavixTelemetryRollout, NavixTraceArrays
from jarl.experiments.io.checkpoints import (
    CHECKPOINT_ALIAS_BEST,
    CHECKPOINT_ALIAS_FINAL,
    CHECKPOINT_ALIAS_LATEST,
    CheckpointRecord,
    CheckpointStatus,
)
from jarl.experiments.node import NodeWorkspace
from jarl.inference.policy import InferencePolicyRuntime, PolicyState
from jarl.inference.run import CheckpointInferenceRequest, run_checkpoint_inference
from jarl.training.config import RLRunConfig
from jarl.training.env_factory import navix_full_jit_env_factory

__all__ = [
    "EVAL_SEEDS",
    "EvalMetrics",
    "checkpoint_eval_metrics",
    "hooked_baseline_metrics",
    "latest_saved_checkpoint",
    "original_reward_payload",
    "overlay_entry_scale",
    "reward_payload",
]

EVAL_SEEDS = (0, 1, 2, 3, 4)
"""Fixed eval seeds shared by pre/post policy rollouts."""

_ORIGINAL_REWARD = RewardWeightsConfig(goal_reached=1.0)
"""Materialized EmptyVariant mix before `graph_set_reward`."""

_ROT_CW = 1
_FORWARD = 2
_FARM_PERIOD = (_FORWARD, _ROT_CW, _ROT_CW, _FORWARD, _ROT_CW, _ROT_CW)
_FARM_CYCLES = 15


@dataclass(frozen=True, slots=True)
class EvalMetrics:
    """Mean success and re-entry counts over a seed set or scripted baseline."""

    success_rate: float
    occupancy_mean: float


def original_reward_payload() -> dict[str, float]:
    """Return the materialized configurable mix of the experiment case."""
    return {name: float(value) for name, value in _ORIGINAL_REWARD.model_dump().items()}


def reward_payload(config: RLRunConfig) -> dict[str, float] | None:
    """Return named configurable weights, or ``None`` for native Navix reward."""
    reward = config.environment.reward
    if reward is None:
        return None
    return {name: float(value) for name, value in reward.model_dump().items()}


def latest_saved_checkpoint(workspace: NodeWorkspace) -> CheckpointRecord | None:
    """Return the preferred saved checkpoint on a node, if any."""
    for alias in (CHECKPOINT_ALIAS_BEST, CHECKPOINT_ALIAS_LATEST, CHECKPOINT_ALIAS_FINAL):
        record = workspace.resolve_checkpoint_alias(alias)
        if record is not None and record.status is CheckpointStatus.SAVED:
            return record
    saved = [record for record in workspace.list_checkpoints() if record.status is CheckpointStatus.SAVED]
    if not saved:
        return None
    return max(saved, key=lambda record: record.checkpoint_step)


def overlay_entry_scale(config: RLRunConfig) -> float:
    """Return the overlay pulse on cell entry with configurable weights zeroed.

    Zeroing the mix isolates ``r_scenario`` so a redesigned mix cannot mask a
    missing overlay.
    """
    setup_jax(config.jax)
    zero_mix = RewardWeightsConfig.model_validate(dict.fromkeys(RewardWeightsConfig.model_fields, 0.0))
    probe = config.model_copy(
        update={"environment": config.environment.model_copy(update={"reward": zero_mix})},
    )
    env = navix_full_jit_env_factory(probe)
    keys = jax.random.split(jax.random.PRNGKey(0), 1)
    state = env.reset(keys, eval_mode=True)
    state = env.step(state, jnp.asarray([_FORWARD], dtype=jnp.int32))
    return float(state.reward[0])


def hooked_baseline_metrics(config: RLRunConfig) -> EvalMetrics:
    """Return scripted re-entry occupancy when no hooked checkpoint exists."""
    setup_jax(config.jax)
    actions = _FARM_PERIOD * _FARM_CYCLES
    environment = config.environment
    spec = resolve_scenario_reward_spec(
        environment.env_id,
        environment.scenario_reward_id,
        environment.scenario_reward_version,
    )
    rollout = NavixTelemetryRollout(
        environment.env_id,
        _sequence_runtime(),
        len(actions),
        NavixCaptureProfile.analysis(),
        reward=environment.reward,
        scenario_spec=spec,
    )
    keys = jax.random.split(jax.random.PRNGKey(0), 1)
    trace = rollout.rollout(jnp.asarray(actions, dtype=jnp.int32), keys).to_host()
    return EvalMetrics(success_rate=0.0, occupancy_mean=float(_episode_occupancy(trace, 0)))


def checkpoint_eval_metrics(
    workspace: NodeWorkspace,
    config: RLRunConfig,
    *,
    seeds: tuple[int, ...] = EVAL_SEEDS,
) -> EvalMetrics:
    """Evaluate greedy success and occupancy of one checkpoint over ``seeds``."""
    record = latest_saved_checkpoint(workspace)
    if record is None:
        msg = f"Node {workspace.id} has no saved checkpoint to evaluate."
        raise ValueError(msg)
    setup_jax(config.jax)
    result = run_checkpoint_inference(
        workspace,
        config,
        CheckpointInferenceRequest(
            checkpoint_step=record.checkpoint_step,
            seeds=seeds,
            capture_profile=NavixCaptureProfile.analysis(),
        ),
    )
    successes = [summary.outcome.success is True for summary in result.summaries]
    occupancies = [_episode_occupancy(result.trace, index) for index in range(len(result.summaries))]
    return EvalMetrics(
        success_rate=float(np.mean(successes)),
        occupancy_mean=float(np.mean(occupancies)),
    )


def _episode_occupancy(trace: NavixTraceArrays, episode_index: int) -> int:
    """Count rising-edge visits to the overlay cell in one episode."""
    if trace.player_positions is None:
        return 0
    length = int(np.asarray(trace.episode_lengths)[episode_index])
    positions = np.asarray(trace.player_positions)[episode_index, : length + 1]
    return occupancy_entries(occupancy_mask(positions, CELL_ENTRY_POSITION))


def _sequence_runtime() -> InferencePolicyRuntime:
    """Return a runtime that plays an action sequence stored in ``params``."""

    def preprocess_observation(observation: jax.Array) -> jax.Array:
        return jnp.asarray(observation, dtype=jnp.float32).reshape((observation.shape[0], -1)) / 255.0

    def initial_state(batch_size: int) -> jax.Array:
        return jnp.zeros((batch_size,), dtype=jnp.int32)

    def greedy_step(
        params: object,
        _observation: jax.Array,
        state: PolicyState,
    ) -> tuple[jax.Array, PolicyState]:
        action_sequence = jnp.asarray(params, dtype=jnp.int32)
        indices = jnp.minimum(jnp.asarray(state, dtype=jnp.int32), action_sequence.shape[0] - 1)
        return action_sequence[indices], jnp.asarray(state, dtype=jnp.int32) + 1

    return InferencePolicyRuntime(
        preprocess_observation=preprocess_observation,
        initial_state=initial_state,
        greedy_step=greedy_step,
    )
