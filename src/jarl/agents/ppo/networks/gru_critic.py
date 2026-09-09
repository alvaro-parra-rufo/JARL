"""Flax GRU critic network for PPO-GRU full-JAX."""

from __future__ import annotations

import flax.linen as nn
import jax
from flax.linen.initializers import constant, orthogonal

from jarl.agents.ppo.networks.gru_torso import GrUTorso, previous_done_mask

__all__ = ["GrUCritic"]


class GrUCritic(GrUTorso):
    """Recurrent state-value critic that shares the GRU history contract of PPO-GRU policies."""

    def setup(self) -> None:
        """Register a scalar value head on top of the shared GRU torso."""
        super().setup()
        self.value_head = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

    def apply_one_step(self, obs: jax.Array, carry: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Advance the GRU carry and return a value estimate of shape ``(..., 1)``."""
        torso, next_carry = self.encode_step(obs, carry)
        return self.value_head(torso), next_carry

    def forward_sequence(
        self,
        obs_seq: jax.Array,
        done_seq: jax.Array,
        init_carry: jax.Array,
    ) -> jax.Array:
        """Unroll the critic over a time-major sequence with episode-boundary resets."""
        done_prev = previous_done_mask(done_seq)

        def step(carry: jax.Array, inputs: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
            obs_t, done_prev_t = inputs
            carry = carry * (1.0 - done_prev_t)
            value_t, next_carry = self.apply_one_step(obs_t, carry)
            return next_carry, value_t

        _, value_seq = jax.lax.scan(step, init_carry, (obs_seq, done_prev), unroll=True)
        return value_seq
