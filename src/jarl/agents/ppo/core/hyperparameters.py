"""PPO hyperparameters shared by feedforward and recurrent trainers."""

from __future__ import annotations

from dataclasses import dataclass

from jarl.training.config import AlgorithmConfig

__all__ = ["PPOHyperparameters"]


@dataclass(frozen=True, slots=True)
class PPOHyperparameters:
    """Scalar PPO coefficients used by GAE, loss, and optimization helpers."""

    gamma: float
    gae_lambda: float
    clip_range: float
    entropy_coef: float
    critic_coef: float

    @classmethod
    def from_algorithm_config(cls, algorithm: AlgorithmConfig) -> PPOHyperparameters:
        """Build hyperparameters from a resolved ``AlgorithmConfig``."""
        return cls(
            gamma=algorithm.gamma,
            gae_lambda=algorithm.gae_lambda,
            clip_range=algorithm.clip_range,
            entropy_coef=algorithm.entropy_coef,
            critic_coef=algorithm.critic_coef,
        )
