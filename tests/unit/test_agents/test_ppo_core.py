"""Tests for shared PPO full-JAX core helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jarl.agents.ppo.core import (
    PPOHyperparameters,
    compute_explained_variance,
    compute_gae_advantages,
    compute_policy_std_dev,
    flatten_rollout_tensor,
    make_minibatch_indices,
    make_ppo_loss_fn,
    mean_loss_metrics,
    normalize_advantages,
    ppo_loss_and_metrics,
)
from jarl.agents.ppo.networks import ContinuousPolicy, Critic, DiscretePolicy
from jarl.training.config import AlgorithmConfig


@pytest.fixture()
def hyperparameters() -> PPOHyperparameters:
    """Default PPO hyperparameters for core tests."""
    return PPOHyperparameters.from_algorithm_config(AlgorithmConfig())


class TestGAE:
    """Tests for generalized advantage estimation."""

    def test_compute_gae_advantages_matches_reference_implementation(self) -> None:
        gamma = 0.99
        gae_lambda = 0.95
        rewards = jnp.array([[1.0, 0.5], [1.0, 0.0], [0.5, 1.0]])
        values = jnp.array([[0.2, 0.3], [0.4, 0.1], [0.6, 0.2]])
        next_values = jnp.array([[0.4, 0.1], [0.6, 0.2], [0.0, 0.0]])
        terminations = jnp.array([[False, False], [False, True], [True, False]])

        advantages, returns = compute_gae_advantages(
            rewards,
            values,
            next_values,
            terminations,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        termination_mask = terminations.astype(jnp.float32)
        delta = rewards + gamma * next_values * (1.0 - termination_mask) - values
        init_advantage = delta[-1]
        expected_advantages = [init_advantage]
        for step_index in range(delta.shape[0] - 2, -1, -1):
            previous = expected_advantages[-1]
            expected_advantages.append(
                delta[step_index] + gamma * gae_lambda * (1.0 - termination_mask[step_index]) * previous
            )
        expected_advantages = jnp.stack(list(reversed(expected_advantages)), axis=0)
        expected_returns = expected_advantages + values

        assert advantages.shape == rewards.shape
        assert jnp.allclose(advantages, expected_advantages, atol=1e-6)
        assert jnp.allclose(returns, expected_returns, atol=1e-6)

    def test_single_step_rollout(self) -> None:
        rewards = jnp.array([[1.0]])
        values = jnp.array([[0.5]])
        next_values = jnp.array([[0.0]])
        terminations = jnp.array([[True]])

        advantages, returns = compute_gae_advantages(
            rewards,
            values,
            next_values,
            terminations,
            gamma=0.99,
            gae_lambda=0.95,
        )

        assert advantages.shape == (1, 1)
        assert float(advantages[0, 0]) == pytest.approx(0.5)
        assert float(returns[0, 0]) == pytest.approx(1.0)


class TestPPOLoss:
    """Tests for PPO loss construction."""

    def test_ppo_loss_and_metrics_returns_finite_scalar(self, hyperparameters: PPOHyperparameters) -> None:
        loss, metrics = ppo_loss_and_metrics(
            new_log_prob=jnp.array(-0.2),
            old_log_prob=jnp.array(-0.3),
            return_target=jnp.array(1.0),
            advantage=jnp.array(0.5),
            old_value=jnp.array(0.8),
            value_prediction=jnp.array(0.8),
            entropy_loss=jnp.array(0.1),
            hyperparameters=hyperparameters,
        )

        assert jnp.isfinite(loss)
        assert jnp.isfinite(metrics.policy_gradient_loss)
        assert jnp.isfinite(metrics.critic_loss)

    def test_make_ppo_loss_fn_discrete_matches_metric_keys(
        self,
        hyperparameters: PPOHyperparameters,
    ) -> None:
        policy = DiscretePolicy(n_actions=3, observation_indices=jnp.arange(4))
        critic = Critic(observation_indices=jnp.arange(4))
        observation = jnp.zeros((4,))
        policy_key, critic_key = jax.random.split(jax.random.PRNGKey(0))
        policy_params = policy.init(policy_key, observation[None])
        critic_params = critic.init(critic_key, observation[None])

        loss_fn = make_ppo_loss_fn(
            policy_apply=policy.apply,
            critic_apply=critic.apply,
            is_discrete=True,
            hyperparameters=hyperparameters,
        )

        loss, metrics = loss_fn(
            policy_params,
            critic_params,
            observation,
            jnp.array(1),
            jnp.array(-0.5),
            jnp.array(0.2),
            jnp.array(0.1),
            jnp.array(0.15),
        )

        assert jnp.isfinite(loss)
        assert set(metrics) == {
            "loss/policy_gradient_loss",
            "loss/critic_loss",
            "loss/entropy_loss",
            "policy_ratio/approx_kl",
            "policy_ratio/clip_fraction",
            "policy_ratio/ratio",
        }

    def test_make_ppo_loss_fn_continuous(self, hyperparameters: PPOHyperparameters) -> None:
        policy = ContinuousPolicy(action_shape=(2,), std_dev=1.0, observation_indices=jnp.arange(4))
        critic = Critic(observation_indices=jnp.arange(4))
        observation = jnp.zeros((4,))
        policy_key, critic_key = jax.random.split(jax.random.PRNGKey(1))
        policy_params = policy.init(policy_key, observation[None])
        critic_params = critic.init(critic_key, observation[None])

        loss_fn = make_ppo_loss_fn(
            policy_apply=policy.apply,
            critic_apply=critic.apply,
            is_discrete=False,
            hyperparameters=hyperparameters,
        )

        loss, metrics = loss_fn(
            policy_params,
            critic_params,
            observation,
            jnp.zeros((2,)),
            jnp.array(-1.0),
            jnp.array(0.5),
            jnp.array(0.2),
            jnp.array(0.4),
        )

        assert jnp.isfinite(loss)
        assert jnp.isfinite(metrics["loss/entropy_loss"])

    def test_mean_loss_metrics_averages_vmapped_outputs(self) -> None:
        stacked = {
            "loss/policy_gradient_loss": jnp.array([1.0, 3.0]),
            "loss/critic_loss": jnp.array([0.5, 1.5]),
        }

        averaged = mean_loss_metrics(stacked)

        assert float(averaged["loss/policy_gradient_loss"]) == pytest.approx(2.0)
        assert float(averaged["loss/critic_loss"]) == pytest.approx(1.0)


class TestMinibatchHelpers:
    """Tests for minibatch index construction and advantage normalization."""

    def test_make_minibatch_indices_shape_and_coverage(self) -> None:
        batch_size = 8
        nr_epochs = 2
        nr_minibatches = 2
        minibatch_size = 4
        indices = make_minibatch_indices(
            jax.random.PRNGKey(2),
            batch_size=batch_size,
            nr_epochs=nr_epochs,
            nr_minibatches=nr_minibatches,
            minibatch_size=minibatch_size,
        )

        assert indices.shape == (nr_epochs * nr_minibatches, minibatch_size)
        assert jnp.all(indices >= 0)
        assert jnp.all(indices < batch_size)

    def test_normalize_advantages_has_zero_mean(self) -> None:
        advantages = jnp.array([1.0, 2.0, 3.0, 4.0])

        normalized = normalize_advantages(advantages)

        assert float(jnp.mean(normalized)) == pytest.approx(0.0, abs=1e-6)
        assert float(jnp.std(normalized)) == pytest.approx(1.0, rel=1e-5)


class TestBatchAndMetricsHelpers:
    """Tests for rollout flattening and post-update metrics."""

    def test_flatten_rollout_tensor(self) -> None:
        tensor = jnp.arange(12).reshape(3, 2, 2)

        flattened = flatten_rollout_tensor(tensor, trailing_shape=(2,))

        assert flattened.shape == (6, 2)
        assert int(flattened[0, 0]) == 0
        assert int(flattened[-1, -1]) == 11

    def test_compute_explained_variance(self) -> None:
        returns = jnp.array([1.0, 2.0, 3.0, 4.0])
        values = jnp.array([1.0, 2.0, 3.0, 4.0])

        explained = compute_explained_variance(returns, values)

        assert float(explained) == pytest.approx(1.0)

    def test_compute_policy_std_dev_discrete_is_zero(self) -> None:
        assert float(compute_policy_std_dev(is_discrete=True)) == 0.0

    def test_compute_policy_std_dev_continuous(self) -> None:
        logstd = jnp.log(jnp.array([0.5, 1.0]))

        std_dev = compute_policy_std_dev(is_discrete=False, policy_logstd=logstd)

        assert float(std_dev) == pytest.approx(0.75)
