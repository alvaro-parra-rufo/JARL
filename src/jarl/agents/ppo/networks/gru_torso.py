"""Shared GRU observation encoding and MLP torso for PPO-GRU networks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

__all__ = ["GrUTorso", "previous_done_mask"]


def previous_done_mask(done_seq: jax.Array) -> jax.Array:
    """Return the per-timestep carry reset mask for a time-major done sequence.

    Timestep ``t`` resets the GRU carry when timestep ``t - 1`` ended. The first
    step keeps the provided initial carry.
    """
    return jnp.concatenate(
        [jnp.zeros((1,), dtype=jnp.float32), done_seq.astype(jnp.float32)[:-1]],
        axis=0,
    )


class GrUTorso(nn.Module):
    """GRU encoder plus fused MLP torso shared by PPO-GRU policy and critic."""

    obs_encoding_dim: int
    gru_hidden_dim: int
    gru_obs_combine_method: Literal["concat", "film"]
    share_gru_obs_encoder: bool
    observation_indices: Sequence[int]

    def setup(self) -> None:
        """Register GRU encoder, optional FiLM mixer, and MLP torso layers."""
        self.gru_obs_encoder_dense = nn.Dense(
            self.obs_encoding_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.gru_obs_encoder_ln = nn.LayerNorm()
        if not self.share_gru_obs_encoder:
            self.obs_encoder_dense = nn.Dense(
                self.obs_encoding_dim,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )
            self.obs_encoder_ln = nn.LayerNorm()
        self.gru = nn.GRUCell(features=self.gru_hidden_dim)
        self.gru_ln = nn.LayerNorm()
        if self.gru_obs_combine_method == "film":
            self.gru_film_gamma = nn.Dense(
                self.obs_encoding_dim,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )
            self.gru_film_beta = nn.Dense(
                self.obs_encoding_dim,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )
        self.torso_dense1 = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.torso_ln1 = nn.LayerNorm()
        self.torso_dense2 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.torso_dense3 = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))

    def initialize_carry(self, nr_envs: int) -> jax.Array:
        """Return a zero GRU carry for ``nr_envs`` parallel environments."""
        return jnp.zeros((nr_envs, self.gru_hidden_dim), dtype=jnp.float32)

    def obs_encode(self, obs: jax.Array) -> jax.Array:
        """Encode observations for the torso when encoders are not shared."""
        features = obs[..., self.observation_indices]
        encoded = self.obs_encoder_dense(features)
        encoded = self.obs_encoder_ln(encoded)
        return nn.elu(encoded)

    def gru_obs_encode(self, obs: jax.Array) -> jax.Array:
        """Encode observations fed into the GRU cell."""
        features = obs[..., self.observation_indices]
        encoded = self.gru_obs_encoder_dense(features)
        encoded = self.gru_obs_encoder_ln(encoded)
        return nn.elu(encoded)

    def decode_torso(self, obs_latent: jax.Array, gru_latent: jax.Array) -> jax.Array:
        """Fuse observation and GRU latents and run the shared MLP torso."""
        gru_latent = self.gru_ln(gru_latent)
        gru_latent = nn.elu(gru_latent)
        if self.gru_obs_combine_method == "concat":
            torso_in = jnp.concatenate([obs_latent, gru_latent], axis=-1)
        else:
            gamma = self.gru_film_gamma(gru_latent)
            beta = self.gru_film_beta(gru_latent)
            torso_in = obs_latent * gamma + beta
        hidden = self.torso_dense1(torso_in)
        hidden = self.torso_ln1(hidden)
        hidden = nn.elu(hidden)
        hidden = self.torso_dense2(hidden)
        hidden = nn.elu(hidden)
        hidden = self.torso_dense3(hidden)
        return nn.elu(hidden)

    def encode_step(self, obs: jax.Array, carry: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Advance the GRU and return fused torso features with the next carry."""
        gru_obs_latent = self.gru_obs_encode(obs)
        carry, hidden = self.gru(carry, gru_obs_latent)
        obs_latent = gru_obs_latent if self.share_gru_obs_encoder else self.obs_encode(obs)
        torso = self.decode_torso(obs_latent, hidden)
        return torso, carry
