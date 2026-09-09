"""Derived full-JAX PPO training schedule counters."""

from __future__ import annotations

from dataclasses import dataclass

from jarl.training.config import AlgorithmConfig, EnvironmentConfig, RLRunConfig

__all__ = [
    "TrainingSchedule",
    "apply_training_schedule",
    "compute_training_schedule",
    "format_training_schedule_message",
]


@dataclass(frozen=True, slots=True)
class TrainingSchedule:
    """Immutable derived counters for a full-JAX PPO training loop."""

    requested_total_timesteps: int
    actual_total_timesteps: int
    actual_rollout_updates: int
    actual_optimizer_updates: int
    batch_size: int
    nr_minibatches: int
    effective_evaluation_and_save_frequency: int


def compute_training_schedule(
    environment: EnvironmentConfig,
    algorithm: AlgorithmConfig,
) -> TrainingSchedule:
    """Compute derived PPO loop counters from environment and algorithm inputs.

    Derives the effective training schedule without mutating the input configs.

    Args:
        environment: Resolved environment settings containing ``nr_envs``.
        algorithm: Resolved algorithm settings containing rollout and PPO fields.

    Returns:
        Derived schedule counters for the training loop.

    Raises:
        ValueError: If rollout or evaluation frequencies are inconsistent with
            ``batch_size`` or ``minibatch_size``.
    """
    batch_size = int(environment.nr_envs * algorithm.nr_steps)
    if batch_size <= 0:
        msg = "batch_size must be positive; check nr_envs and nr_steps."
        raise ValueError(msg)

    requested_total_timesteps = int(algorithm.total_timesteps)
    evaluation_frequency = int(algorithm.evaluation_and_save_frequency)
    if evaluation_frequency == -1:
        evaluation_frequency = batch_size * (requested_total_timesteps // batch_size)
    elif evaluation_frequency % batch_size != 0:
        msg = f"evaluation_and_save_frequency ({evaluation_frequency}) must be divisible by batch_size ({batch_size})."
        raise ValueError(msg)

    minibatch_size = int(algorithm.minibatch_size)
    if batch_size % minibatch_size != 0:
        msg = f"minibatch_size ({minibatch_size}) must divide batch_size ({batch_size})."
        raise ValueError(msg)

    updates_per_eval = evaluation_frequency // batch_size
    multi_iterations = requested_total_timesteps // evaluation_frequency if evaluation_frequency > 0 else 0
    rollout_updates = multi_iterations * updates_per_eval
    nr_minibatches = batch_size // minibatch_size
    optimizer_updates = rollout_updates * int(algorithm.nr_epochs) * nr_minibatches

    return TrainingSchedule(
        requested_total_timesteps=requested_total_timesteps,
        actual_total_timesteps=rollout_updates * batch_size,
        actual_rollout_updates=rollout_updates,
        actual_optimizer_updates=optimizer_updates,
        batch_size=batch_size,
        nr_minibatches=nr_minibatches,
        effective_evaluation_and_save_frequency=evaluation_frequency,
    )


def apply_training_schedule(config: RLRunConfig) -> RLRunConfig:
    """Fill derived algorithm schedule fields on a resolved RL config.

    Args:
        config: Resolved run config with user-facing algorithm inputs.

    Returns:
        New config whose ``algorithm`` block includes derived schedule fields.
    """
    schedule = compute_training_schedule(config.environment, config.algorithm)
    return config.apply_overrides(
        {
            "algorithm.requested_total_timesteps": schedule.requested_total_timesteps,
            "algorithm.actual_total_timesteps": schedule.actual_total_timesteps,
            "algorithm.actual_rollout_updates": schedule.actual_rollout_updates,
            "algorithm.actual_optimizer_updates": schedule.actual_optimizer_updates,
            "algorithm.batch_size": schedule.batch_size,
            "algorithm.nr_minibatches": schedule.nr_minibatches,
            "algorithm.effective_evaluation_and_save_frequency": (schedule.effective_evaluation_and_save_frequency),
        }
    )


def format_training_schedule_message(schedule: TrainingSchedule) -> str:
    """Format a single-line schedule summary for logging.

    Args:
        schedule: Derived training schedule counters.

    Returns:
        Human-readable summary suitable for ``train.log`` INFO lines.
    """
    return (
        "training schedule: "
        f"requested_total_timesteps={schedule.requested_total_timesteps}, "
        f"actual_total_timesteps={schedule.actual_total_timesteps}, "
        f"batch_size={schedule.batch_size}, "
        f"nr_minibatches={schedule.nr_minibatches}, "
        f"actual_rollout_updates={schedule.actual_rollout_updates}, "
        f"actual_optimizer_updates={schedule.actual_optimizer_updates}, "
        f"effective_evaluation_and_save_frequency={schedule.effective_evaluation_and_save_frequency}"
    )
