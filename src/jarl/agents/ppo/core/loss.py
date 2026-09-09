"""PPO clipped surrogate loss and per-sample metrics."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp

from jarl.agents.ppo.core.hyperparameters import PPOHyperparameters
from jarl.agents.ppo.core.value_loss import clipped_value_loss

__all__ = [
    "PPOLossMetrics",
    "make_ppo_loss_fn",
    "mean_loss_metrics",
    "ppo_loss_and_metrics",
]


class PPOLossMetrics(NamedTuple):
    """Per-sample PPO optimization metrics."""

    policy_gradient_loss: jax.Array
    critic_loss: jax.Array
    entropy_loss: jax.Array
    approx_kl: jax.Array
    clip_fraction: jax.Array
    policy_ratio: jax.Array


class _PolicyApply(Protocol):
    def __call__(self, params: object, observation: jax.Array) -> jax.Array | tuple[jax.Array, jax.Array]:
        """Apply the policy network to a single observation."""


class _CriticApply(Protocol):
    def __call__(self, params: object, observation: jax.Array) -> jax.Array:
        """Apply the critic network to a single observation."""


def _discrete_policy_terms(
    logits: jax.Array,
    action: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    log_probs = jax.nn.log_softmax(logits)
    probs = jax.nn.softmax(logits)
    new_log_prob = log_probs[action.astype(jnp.int32)]
    entropy_loss = -jnp.sum(probs * log_probs)
    return new_log_prob, entropy_loss


def _continuous_policy_terms(
    action_mean: jax.Array,
    action_logstd: jax.Array,
    action: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    action_std = jnp.exp(action_logstd)
    new_log_prob = (
        -0.5 * ((action - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd
    ).sum()
    entropy = action_logstd + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)
    entropy_loss = entropy.sum()
    return new_log_prob, entropy_loss


def ppo_loss_and_metrics(
    *,
    new_log_prob: jax.Array,
    old_log_prob: jax.Array,
    return_target: jax.Array,
    advantage: jax.Array,
    old_value: jax.Array,
    value_prediction: jax.Array,
    entropy_loss: jax.Array,
    hyperparameters: PPOHyperparameters,
) -> tuple[jax.Array, PPOLossMetrics]:
    """Combine policy, value, and entropy terms into the scalar PPO loss."""
    old_log_prob = jax.lax.stop_gradient(old_log_prob)
    advantage = jax.lax.stop_gradient(advantage)
    log_ratio = new_log_prob - old_log_prob
    ratio = jnp.exp(log_ratio)
    approx_kl = (ratio - 1.0) - log_ratio
    clip_fraction = jnp.float32(jnp.abs(ratio - 1.0) > hyperparameters.clip_range)

    pg_loss_unclipped = -advantage * ratio
    pg_loss_clipped = -advantage * jnp.clip(
        ratio,
        1.0 - hyperparameters.clip_range,
        1.0 + hyperparameters.clip_range,
    )
    policy_gradient_loss = jnp.maximum(pg_loss_unclipped, pg_loss_clipped)
    critic_loss = clipped_value_loss(
        value_prediction=value_prediction,
        old_value=old_value,
        return_target=return_target,
        clip_range=hyperparameters.clip_range,
    )

    loss = (
        policy_gradient_loss - hyperparameters.entropy_coef * entropy_loss + hyperparameters.critic_coef * critic_loss
    )
    metrics = PPOLossMetrics(
        policy_gradient_loss=policy_gradient_loss,
        critic_loss=critic_loss,
        entropy_loss=entropy_loss,
        approx_kl=approx_kl,
        clip_fraction=clip_fraction,
        policy_ratio=ratio,
    )
    return loss, metrics


def make_ppo_loss_fn(
    *,
    policy_apply: _PolicyApply,
    critic_apply: _CriticApply,
    is_discrete: bool,
    hyperparameters: PPOHyperparameters,
) -> Callable[..., tuple[jax.Array, dict[str, jax.Array]]]:
    """Build a per-sample PPO loss callable for ``jax.vmap``.

    Build the per-sample PPO loss used by the full-JAX trainer.
    """

    def loss_fn(
        policy_params: object,
        critic_params: object,
        state: jax.Array,
        action: jax.Array,
        old_log_prob: jax.Array,
        return_target: jax.Array,
        advantage: jax.Array,
        old_value: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        if is_discrete:
            logits = policy_apply(policy_params, state)
            new_log_prob, entropy_loss = _discrete_policy_terms(logits, action)
        else:
            action_mean, action_logstd = policy_apply(policy_params, state)
            new_log_prob, entropy_loss = _continuous_policy_terms(action_mean, action_logstd, action)

        value_prediction = critic_apply(critic_params, state).squeeze(-1)
        loss, metrics = ppo_loss_and_metrics(
            new_log_prob=new_log_prob,
            old_log_prob=old_log_prob,
            return_target=return_target,
            advantage=advantage,
            old_value=old_value,
            value_prediction=value_prediction,
            entropy_loss=entropy_loss,
            hyperparameters=hyperparameters,
        )
        metric_dict = {
            "loss/policy_gradient_loss": metrics.policy_gradient_loss,
            "loss/critic_loss": metrics.critic_loss,
            "loss/entropy_loss": metrics.entropy_loss,
            "policy_ratio/approx_kl": metrics.approx_kl,
            "policy_ratio/clip_fraction": metrics.clip_fraction,
            "policy_ratio/ratio": metrics.policy_ratio,
        }
        return loss, metric_dict

    return loss_fn


def mean_loss_metrics(
    metrics: dict[str, jax.Array],
) -> dict[str, jax.Array]:
    """Average a vmap-stacked metrics dictionary."""
    return {key: jnp.mean(value) for key, value in metrics.items()}
