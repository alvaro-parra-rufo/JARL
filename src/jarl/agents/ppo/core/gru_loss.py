"""PPO loss for recurrent GRU policies and critics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import jax
import jax.numpy as jnp

from jarl.agents.ppo.core.hyperparameters import PPOHyperparameters
from jarl.agents.ppo.core.value_loss import clipped_value_loss

__all__ = [
    "make_ppo_gru_loss_fn",
]


class _GrUSequenceApply(Protocol):
    def __call__(
        self,
        params: object,
        obs_seq: jax.Array,
        done_seq: jax.Array,
        init_carry: jax.Array,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        """Apply a GRU network over a time-major sequence."""


def make_ppo_gru_loss_fn(
    *,
    policy_forward_sequence: _GrUSequenceApply,
    critic_forward_sequence: _GrUSequenceApply,
    is_discrete: bool,
    hyperparameters: PPOHyperparameters,
) -> Callable[..., tuple[jax.Array, dict[str, jax.Array]]]:
    """Build a per-environment sequence PPO loss callable for ``jax.vmap``.

    Policy and critic both unroll their GRU carries over the rollout window so
    value estimates condition on the same episode history as the actor.
    """

    def loss_fn(
        policy_params: object,
        critic_params: object,
        state_seq: jax.Array,
        action_seq: jax.Array,
        old_log_prob_seq: jax.Array,
        return_seq: jax.Array,
        advantage_seq: jax.Array,
        old_value_seq: jax.Array,
        done_seq: jax.Array,
        init_policy_carry: jax.Array,
        init_critic_carry: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        if is_discrete:
            logits = policy_forward_sequence(policy_params, state_seq, done_seq, init_policy_carry)
            log_probs = jax.nn.log_softmax(logits)
            probs = jax.nn.softmax(logits)
            new_log_prob = jnp.take_along_axis(
                log_probs,
                action_seq.astype(jnp.int32)[..., None],
                axis=-1,
            ).squeeze(-1)
            entropy_loss = -jnp.sum(probs * log_probs, axis=-1)
        else:
            action_mean, action_logstd = policy_forward_sequence(
                policy_params,
                state_seq,
                done_seq,
                init_policy_carry,
            )
            action_std = jnp.exp(action_logstd)
            new_log_prob = (
                -0.5 * ((action_seq - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd
            ).sum(-1)
            entropy = action_logstd + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)
            entropy_loss = entropy.sum(-1)

        value_prediction = critic_forward_sequence(
            critic_params,
            state_seq,
            done_seq,
            init_critic_carry,
        ).squeeze(-1)
        old_log_prob_seq = jax.lax.stop_gradient(old_log_prob_seq)
        advantage_seq = jax.lax.stop_gradient(advantage_seq)
        log_ratio = new_log_prob - old_log_prob_seq
        ratio = jnp.exp(log_ratio)
        approx_kl = (ratio - 1.0) - log_ratio
        clip_fraction = jnp.float32(jnp.abs(ratio - 1.0) > hyperparameters.clip_range)

        pg_loss_unclipped = -advantage_seq * ratio
        pg_loss_clipped = -advantage_seq * jnp.clip(
            ratio,
            1.0 - hyperparameters.clip_range,
            1.0 + hyperparameters.clip_range,
        )
        policy_gradient_loss = jnp.maximum(pg_loss_unclipped, pg_loss_clipped)
        critic_loss = clipped_value_loss(
            value_prediction=value_prediction,
            old_value=old_value_seq,
            return_target=return_seq,
            clip_range=hyperparameters.clip_range,
        )
        loss = (
            policy_gradient_loss
            - hyperparameters.entropy_coef * entropy_loss
            + hyperparameters.critic_coef * critic_loss
        )
        metrics = {
            "loss/policy_gradient_loss": policy_gradient_loss,
            "loss/critic_loss": critic_loss,
            "loss/entropy_loss": entropy_loss,
            "policy_ratio/approx_kl": approx_kl,
            "policy_ratio/clip_fraction": clip_fraction,
            "policy_ratio/ratio": jnp.max(ratio),
        }
        return loss, metrics

    return loss_fn
