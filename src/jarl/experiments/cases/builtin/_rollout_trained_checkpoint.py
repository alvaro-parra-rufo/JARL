"""Shared training config and validation for rollout analysis experiment cases."""

from __future__ import annotations

from jarl.agents.ppo.trainer import ppo_full_jax_trainer
from jarl.experiments.cases.builtin._navix_rollout_checkpoint import (
    ROLLOUT_TRAINED_CHECKPOINT_EVAL_FREQUENCY,
    ROLLOUT_TRAINED_ENV_ID,
    ROLLOUT_TRAINED_MAX_STEPS,
)
from jarl.experiments.cases.case import ExperimentCaseContext
from jarl.experiments.io.checkpoints import (
    CHECKPOINT_BEST_LENGTH_METRIC,
    CHECKPOINT_BEST_RETURN_METRIC,
    CheckpointStatus,
)
from jarl.experiments.node import NodeWorkspace
from jarl.training.config import AlgorithmConfig, EnvironmentConfig, RLRunConfig, VideoConfig
from jarl.training.runner import run_training
from jarl.training.schedule import compute_training_schedule

__all__ = [
    "ROLLOUT_TRAINED_CHECKPOINT_EVAL_FREQUENCY",
    "ROLLOUT_TRAINED_ENV_ID",
    "ROLLOUT_TRAINED_ENV_SEED",
    "ROLLOUT_TRAINED_MAX_STEPS",
    "ROLLOUT_TRAINED_TRAINING_TIMESTEPS",
    "assert_varied_saved_checkpoint_eval",
    "materialize_rollout_trained_case",
    "rollout_trained_run_config",
    "saved_checkpoint_count",
]

ROLLOUT_TRAINED_ENV_SEED = 4
"""Environment seed that reliably produces a learning curve on Empty-5x5."""

ROLLOUT_TRAINED_TRAINING_TIMESTEPS = 16_384
"""Short PPO budget for rollout analysis cases (enough for two-plus evals)."""

ROLLOUT_TRAINED_NR_ENVS = 8
"""Parallel env count used by trained rollout scenarios."""

ROLLOUT_TRAINED_NR_STEPS = 16
"""Rollout length per PPO update in trained rollout scenarios."""

ROLLOUT_TRAINED_MINIBATCH_SIZE = 64
"""Minibatch size used by trained rollout scenarios."""

ROLLOUT_TRAINED_NR_EPOCHS = 2
"""Optimizer epochs per rollout update in trained rollout scenarios."""

_MIN_DISTINCT_CHECKPOINT_EVAL_VALUES = 2
"""Minimum distinct eval metric values required across saved checkpoints."""


def rollout_trained_run_config(
    *,
    evaluation_and_save_frequency: int = ROLLOUT_TRAINED_CHECKPOINT_EVAL_FREQUENCY,
) -> RLRunConfig:
    """Return a PPO config that saves periodic checkpoints with evolving eval metrics."""
    return RLRunConfig(
        environment=EnvironmentConfig(
            env_id=ROLLOUT_TRAINED_ENV_ID,
            nr_envs=ROLLOUT_TRAINED_NR_ENVS,
            seed=ROLLOUT_TRAINED_ENV_SEED,
            max_episode_steps=ROLLOUT_TRAINED_MAX_STEPS,
        ),
        algorithm=AlgorithmConfig(
            name="ppo.full_jax.navix",
            total_timesteps=ROLLOUT_TRAINED_TRAINING_TIMESTEPS,
            nr_steps=ROLLOUT_TRAINED_NR_STEPS,
            minibatch_size=ROLLOUT_TRAINED_MINIBATCH_SIZE,
            nr_epochs=ROLLOUT_TRAINED_NR_EPOCHS,
            evaluation_and_save_frequency=evaluation_and_save_frequency,
        ),
        video=VideoConfig(record_video=False, record_final_video=False),
    )


def saved_checkpoint_count(*, evaluation_and_save_frequency: int) -> int:
    """Return the number of periodic checkpoints saved for one trained rollout config."""
    config = rollout_trained_run_config(evaluation_and_save_frequency=evaluation_and_save_frequency)
    schedule = compute_training_schedule(config.environment, config.algorithm)
    return schedule.actual_total_timesteps // schedule.effective_evaluation_and_save_frequency


def rollout_trained_checkpoint_step(*, evaluation_and_save_frequency: int) -> int:
    """Return the final checkpoint step for one trained rollout config."""
    config = rollout_trained_run_config(evaluation_and_save_frequency=evaluation_and_save_frequency)
    return compute_training_schedule(config.environment, config.algorithm).actual_total_timesteps


def materialize_rollout_trained_case(
    context: ExperimentCaseContext[RLRunConfig],
    *,
    evaluation_and_save_frequency: int,
    label: str,
) -> NodeWorkspace:
    """Train a root, persist periodic checkpoints, and validate eval diversity."""
    result = run_training(
        trainer=ppo_full_jax_trainer,
        experiment_dir=context.experiment_dir,
        config=rollout_trained_run_config(evaluation_and_save_frequency=evaluation_and_save_frequency),
        create_root=True,
        branch="main",
        label=label,
    )
    assert_varied_saved_checkpoint_eval(result.workspace)
    context.register_alias("root", result.workspace)
    return result.workspace


def assert_varied_saved_checkpoint_eval(workspace: NodeWorkspace) -> None:
    """Require saved checkpoints to expose at least two eval return and length values."""
    saved = [record for record in workspace.list_checkpoints() if record.status is CheckpointStatus.SAVED]
    eval_returns = {
        float(record.metrics[CHECKPOINT_BEST_RETURN_METRIC])
        for record in saved
        if CHECKPOINT_BEST_RETURN_METRIC in record.metrics
    }
    eval_lengths = {
        float(record.metrics[CHECKPOINT_BEST_LENGTH_METRIC])
        for record in saved
        if CHECKPOINT_BEST_LENGTH_METRIC in record.metrics
    }
    if len(eval_returns) < _MIN_DISTINCT_CHECKPOINT_EVAL_VALUES:
        msg = (
            "Expected at least two distinct eval/episode_return values across saved checkpoints, "
            f"got {sorted(eval_returns)}."
        )
        raise RuntimeError(msg)
    if len(eval_lengths) < _MIN_DISTINCT_CHECKPOINT_EVAL_VALUES:
        msg = (
            "Expected at least two distinct eval/episode_length values across saved checkpoints, "
            f"got {sorted(eval_lengths)}."
        )
        raise RuntimeError(msg)
