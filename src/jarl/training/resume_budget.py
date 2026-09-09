"""Remaining training budget helpers for intra-node resume."""

from __future__ import annotations

from jarl.training.config import RLRunConfig
from jarl.training.schedule import TrainingSchedule, apply_training_schedule, compute_training_schedule

__all__ = [
    "apply_remaining_timestep_budget",
    "remaining_timesteps",
]


def remaining_timesteps(config: RLRunConfig, completed_global_step: int) -> int:
    """Return environment steps still owed after ``completed_global_step``."""
    requested = int(config.algorithm.total_timesteps)
    return max(0, requested - int(completed_global_step))


def apply_remaining_timestep_budget(
    config: RLRunConfig,
    completed_global_step: int,
) -> tuple[RLRunConfig, TrainingSchedule]:
    """Shrink ``total_timesteps`` to the remaining budget and recompute schedule."""
    remaining = remaining_timesteps(config, completed_global_step)
    adjusted = config.apply_overrides({"algorithm.total_timesteps": remaining})
    schedule = compute_training_schedule(adjusted.environment, adjusted.algorithm)
    return apply_training_schedule(adjusted), schedule
