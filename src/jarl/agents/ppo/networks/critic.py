"""Flax critic network for PPO full-JAX."""

from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

__all__ = ["Critic"]


class Critic(nn.Module):
    """State-value critic for PPO full-JAX."""

    observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        """Return scalar value estimates with shape ``(..., 1)``."""
        features = observations[..., self.observation_indices]
        value = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(features)
        value = nn.LayerNorm()(value)
        value = nn.elu(value)
        value = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(value)
        value = nn.elu(value)
        value = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(value)
        value = nn.elu(value)
        return nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(value)
