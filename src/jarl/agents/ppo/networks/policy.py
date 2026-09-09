"""Flax policy networks for PPO full-JAX."""

from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

__all__ = [
    "ContinuousPolicy",
    "DiscretePolicy",
]


class ContinuousPolicy(nn.Module):
    """Diagonal Gaussian policy head for continuous action spaces."""

    action_shape: Sequence[int]
    std_dev: float
    observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return action mean and log-standard-deviation for batched observations."""
        features = observations[..., self.observation_indices]
        mean = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(features)
        mean = nn.LayerNorm()(mean)
        mean = nn.elu(mean)
        mean = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(mean)
        mean = nn.elu(mean)
        mean = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(mean)
        mean = nn.elu(mean)
        mean = nn.Dense(int(np.prod(self.action_shape)), kernel_init=orthogonal(0.01), bias_init=constant(0.0))(mean)
        logstd = self.param(
            "policy_logstd",
            constant(jnp.log(self.std_dev)),
            (1, int(np.prod(self.action_shape))),
        )
        return mean, logstd


class DiscretePolicy(nn.Module):
    """Categorical policy head for discrete action spaces."""

    n_actions: int
    observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        """Return unnormalized action logits for batched observations."""
        features = observations[..., self.observation_indices]
        logits = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(features)
        logits = nn.LayerNorm()(logits)
        logits = nn.elu(logits)
        logits = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(logits)
        logits = nn.elu(logits)
        logits = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(logits)
        logits = nn.elu(logits)
        return nn.Dense(self.n_actions, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(logits)
