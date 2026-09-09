"""Static configuration for compiled PPO training loops."""

from __future__ import annotations

from dataclasses import dataclass

from jarl.training.config import RLRunConfig
from jarl.training.schedule import TrainingSchedule

__all__ = ["PPOFeedforwardLoopConfig", "PPOGrULoopConfig"]


@dataclass(frozen=True, slots=True)
class PPOFeedforwardLoopConfig:
    """Immutable integers and flags consumed inside ``jax.jit`` training code."""

    nr_envs: int
    nr_steps: int
    nr_epochs: int
    minibatch_size: int
    batch_size: int
    nr_minibatches: int
    nr_updates: int
    nr_updates_per_eval: int
    nr_multi_eval_iterations: int
    max_grad_norm: float
    learning_rate: float
    anneal_learning_rate: bool
    evaluation_active: bool
    save_model: bool
    save_checkpoint: bool
    horizon: int

    @classmethod
    def from_run(
        cls,
        config: RLRunConfig,
        schedule: TrainingSchedule,
        *,
        horizon: int,
    ) -> PPOFeedforwardLoopConfig:
        """Build loop constants from resolved config and schedule."""
        algorithm = config.algorithm
        evaluation_frequency = int(
            algorithm.effective_evaluation_and_save_frequency or schedule.effective_evaluation_and_save_frequency
        )
        batch_size = int(algorithm.batch_size or schedule.batch_size)
        updates_per_eval = evaluation_frequency // batch_size
        multi_iterations = (
            int(config.algorithm.total_timesteps) // evaluation_frequency if evaluation_frequency > 0 else 0
        )
        rollout_updates = int(algorithm.actual_rollout_updates or schedule.actual_rollout_updates)
        return cls(
            nr_envs=config.environment.nr_envs,
            nr_steps=algorithm.nr_steps,
            nr_epochs=algorithm.nr_epochs,
            minibatch_size=algorithm.minibatch_size,
            batch_size=batch_size,
            nr_minibatches=int(algorithm.nr_minibatches or schedule.nr_minibatches),
            nr_updates=rollout_updates,
            nr_updates_per_eval=updates_per_eval,
            nr_multi_eval_iterations=multi_iterations,
            max_grad_norm=algorithm.max_grad_norm,
            learning_rate=algorithm.learning_rate,
            anneal_learning_rate=algorithm.anneal_learning_rate,
            evaluation_active=algorithm.evaluation_active,
            save_model=config.runner.save_model,
            save_checkpoint=True,
            horizon=horizon,
        )


@dataclass(frozen=True, slots=True)
class PPOGrULoopConfig:
    """Immutable integers and flags consumed inside recurrent ``jax.jit`` training code."""

    nr_envs: int
    nr_steps: int
    nr_epochs: int
    minibatch_size: int
    batch_size: int
    nr_minibatches: int
    nr_minibatch_envs: int
    nr_updates: int
    nr_updates_per_eval: int
    nr_multi_eval_iterations: int
    max_grad_norm: float
    learning_rate: float
    anneal_learning_rate: bool
    evaluation_active: bool
    save_model: bool
    save_checkpoint: bool
    horizon: int

    @classmethod
    def from_run(
        cls,
        config: RLRunConfig,
        schedule: TrainingSchedule,
        *,
        horizon: int,
    ) -> PPOGrULoopConfig:
        """Build recurrent loop constants from resolved config and schedule."""
        algorithm = config.algorithm
        if algorithm.minibatch_size % algorithm.nr_steps != 0:
            msg = (
                f"minibatch_size ({algorithm.minibatch_size}) must be divisible by "
                f"nr_steps ({algorithm.nr_steps}) for PPO-GRU."
            )
            raise ValueError(msg)
        evaluation_frequency = int(
            algorithm.effective_evaluation_and_save_frequency or schedule.effective_evaluation_and_save_frequency
        )
        batch_size = int(algorithm.batch_size or schedule.batch_size)
        updates_per_eval = evaluation_frequency // batch_size
        multi_iterations = (
            int(config.algorithm.total_timesteps) // evaluation_frequency if evaluation_frequency > 0 else 0
        )
        rollout_updates = int(algorithm.actual_rollout_updates or schedule.actual_rollout_updates)
        nr_minibatch_envs = algorithm.minibatch_size // algorithm.nr_steps
        return cls(
            nr_envs=config.environment.nr_envs,
            nr_steps=algorithm.nr_steps,
            nr_epochs=algorithm.nr_epochs,
            minibatch_size=algorithm.minibatch_size,
            batch_size=batch_size,
            nr_minibatches=int(algorithm.nr_minibatches or schedule.nr_minibatches),
            nr_minibatch_envs=nr_minibatch_envs,
            nr_updates=rollout_updates,
            nr_updates_per_eval=updates_per_eval,
            nr_multi_eval_iterations=multi_iterations,
            max_grad_norm=algorithm.max_grad_norm,
            learning_rate=algorithm.learning_rate,
            anneal_learning_rate=algorithm.anneal_learning_rate,
            evaluation_active=algorithm.evaluation_active,
            save_model=config.runner.save_model,
            save_checkpoint=True,
            horizon=horizon,
        )
