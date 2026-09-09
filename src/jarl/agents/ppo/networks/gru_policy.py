"""Flax GRU policy networks for PPO-GRU full-JAX."""

from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from jarl.agents.ppo.networks.gru_torso import GrUTorso, previous_done_mask

__all__ = [
    "ContinuousGrUPolicy",
    "DiscreteGrUPolicy",
]


class ContinuousGrUPolicy(GrUTorso):
    """Diagonal Gaussian GRU policy for continuous action spaces."""

    action_shape: Sequence[int]
    std_dev: float

    def setup(self) -> None:
        """Register continuous action head parameters on top of the shared GRU torso."""
        super().setup()
        self.mean_head = nn.Dense(
            int(np.prod(self.action_shape)),
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )
        self.logstd = self.param(
            "policy_logstd",
            constant(jnp.log(self.std_dev)),
            (1, int(np.prod(self.action_shape))),
        )

    def apply_one_step(
        self,
        obs: jax.Array,
        carry: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Advance the GRU carry and return action mean and log-standard-deviation."""
        torso, next_carry = self.encode_step(obs, carry)
        return self.mean_head(torso), self.logstd, next_carry

    def forward_sequence(
        self,
        obs_seq: jax.Array,
        done_seq: jax.Array,
        init_carry: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Unroll the policy over a time-major sequence with episode-boundary resets."""
        done_prev = previous_done_mask(done_seq)

        def step(
            carry: jax.Array, inputs: tuple[jax.Array, jax.Array]
        ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
            obs_t, done_prev_t = inputs
            carry = carry * (1.0 - done_prev_t)
            mean_t, logstd_t, next_carry = self.apply_one_step(obs_t, carry)
            return next_carry, (mean_t, logstd_t)

        _, (mean_seq, logstd_seq) = jax.lax.scan(step, init_carry, (obs_seq, done_prev), unroll=True)
        return mean_seq, logstd_seq


class DiscreteGrUPolicy(GrUTorso):
    """Categorical GRU policy for discrete action spaces."""

    n_actions: int

    def setup(self) -> None:
        """Register discrete logits head parameters on top of the shared GRU torso."""
        super().setup()
        self.logits_head = nn.Dense(self.n_actions, kernel_init=orthogonal(0.01), bias_init=constant(0.0))

    def apply_one_step(self, obs: jax.Array, carry: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Advance the GRU carry and return action logits."""
        torso, next_carry = self.encode_step(obs, carry)
        return self.logits_head(torso), next_carry

    def forward_sequence(
        self,
        obs_seq: jax.Array,
        done_seq: jax.Array,
        init_carry: jax.Array,
    ) -> jax.Array:
        """Unroll the policy over a time-major sequence with episode-boundary resets."""
        done_prev = previous_done_mask(done_seq)

        def step(carry: jax.Array, inputs: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
            obs_t, done_prev_t = inputs
            carry = carry * (1.0 - done_prev_t)
            logits_t, next_carry = self.apply_one_step(obs_t, carry)
            return next_carry, logits_t

        _, logits_seq = jax.lax.scan(step, init_carry, (obs_seq, done_prev), unroll=True)
        return logits_seq
