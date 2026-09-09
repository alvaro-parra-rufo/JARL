"""Tests for clipped PPO value loss (Fase 2 critic stability)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jarl.agents.ppo.core.hyperparameters import PPOHyperparameters
from jarl.agents.ppo.core.loss import ppo_loss_and_metrics
from jarl.agents.ppo.core.value_loss import clipped_value_loss, pure_value_loss
from jarl.training.config import AlgorithmConfig


@pytest.fixture()
def hyperparameters() -> PPOHyperparameters:
    """Default PPO hyperparameters for value-loss tests."""
    return PPOHyperparameters.from_algorithm_config(AlgorithmConfig())


class TestClippedValueLoss:
    """Clipped value loss matches standard PPO and bounds critic updates."""

    def test_clipped_value_loss_equals_mse_when_prediction_within_clip_range(self) -> None:
        old_value = jnp.array(1.0)
        return_target = jnp.array(2.0)
        value_prediction = jnp.array(1.1)
        clip_range = 0.2

        clipped = clipped_value_loss(
            value_prediction=value_prediction,
            old_value=old_value,
            return_target=return_target,
            clip_range=clip_range,
        )
        mse = pure_value_loss(
            value_prediction=value_prediction,
            return_target=return_target,
        )

        assert float(clipped) == pytest.approx(float(mse), rel=1e-6)

    def test_clipped_value_loss_uses_clipped_branch_when_prediction_exceeds_clip_range(self) -> None:
        old_value = jnp.array(0.0)
        return_target = jnp.array(100.0)
        value_prediction = jnp.array(10.0)
        clip_range = 0.2

        clipped = clipped_value_loss(
            value_prediction=value_prediction,
            old_value=old_value,
            return_target=return_target,
            clip_range=clip_range,
        )
        mse = pure_value_loss(
            value_prediction=value_prediction,
            return_target=return_target,
        )
        value_clipped = old_value + clip_range
        expected = 0.5 * (value_clipped - return_target) ** 2

        assert float(clipped) == pytest.approx(float(expected), rel=1e-6)
        assert float(clipped) > float(mse)

    def test_clipped_value_loss_uses_clipped_branch_when_prediction_below_clip_range(self) -> None:
        old_value = jnp.array(10.0)
        return_target = jnp.array(-100.0)
        value_prediction = jnp.array(0.0)
        clip_range = 0.2

        clipped = clipped_value_loss(
            value_prediction=value_prediction,
            old_value=old_value,
            return_target=return_target,
            clip_range=clip_range,
        )
        mse = pure_value_loss(
            value_prediction=value_prediction,
            return_target=return_target,
        )
        value_clipped = old_value - clip_range
        expected = 0.5 * (value_clipped - return_target) ** 2

        assert float(clipped) == pytest.approx(float(expected), rel=1e-6)
        assert float(clipped) > float(mse)

    def test_clipped_value_loss_gradient_is_zero_when_clipped_branch_dominates(self) -> None:
        old_value = jnp.array(0.0)
        return_target = jnp.array(1_000_000.0)
        clip_range = 0.2

        def loss_fn(value_prediction: jax.Array) -> jax.Array:
            return clipped_value_loss(
                value_prediction=value_prediction,
                old_value=old_value,
                return_target=return_target,
                clip_range=clip_range,
            )

        gradient = jax.grad(loss_fn)(jnp.array(50.0))

        assert float(gradient) == pytest.approx(0.0, abs=1e-6)

    def test_clipped_value_loss_gradient_smaller_than_pure_mse_on_extreme_return(self) -> None:
        old_value = jnp.array(0.0)
        return_target = jnp.array(1_000_000.0)
        clip_range = 0.2
        value_prediction = jnp.array(5.0)

        def clipped_loss_fn(prediction: jax.Array) -> jax.Array:
            return clipped_value_loss(
                value_prediction=prediction,
                old_value=old_value,
                return_target=return_target,
                clip_range=clip_range,
            )

        def mse_loss_fn(prediction: jax.Array) -> jax.Array:
            return pure_value_loss(
                value_prediction=prediction,
                return_target=return_target,
            )

        clipped_grad = jax.grad(clipped_loss_fn)(value_prediction)
        mse_grad = jax.grad(mse_loss_fn)(value_prediction)

        assert abs(float(clipped_grad)) < abs(float(mse_grad))


class TestPPOLossValueIntegration:
    """``ppo_loss_and_metrics`` uses rollout ``old_value`` and clipped critic loss."""

    def test_ppo_loss_and_metrics_requires_old_value(self, hyperparameters: PPOHyperparameters) -> None:
        loss, metrics = ppo_loss_and_metrics(
            new_log_prob=jnp.array(-0.2),
            old_log_prob=jnp.array(-0.3),
            return_target=jnp.array(100.0),
            advantage=jnp.array(0.5),
            old_value=jnp.array(0.0),
            value_prediction=jnp.array(10.0),
            entropy_loss=jnp.array(0.1),
            hyperparameters=hyperparameters,
        )

        assert jnp.isfinite(loss)
        assert jnp.isfinite(metrics.critic_loss)

    def test_ppo_loss_and_metrics_clips_critic_term_with_old_value(
        self,
        hyperparameters: PPOHyperparameters,
    ) -> None:
        _, metrics = ppo_loss_and_metrics(
            new_log_prob=jnp.array(-0.2),
            old_log_prob=jnp.array(-0.3),
            return_target=jnp.array(100.0),
            advantage=jnp.array(0.0),
            old_value=jnp.array(0.0),
            value_prediction=jnp.array(10.0),
            entropy_loss=jnp.array(0.0),
            hyperparameters=hyperparameters,
        )
        expected_critic = clipped_value_loss(
            value_prediction=jnp.array(10.0),
            old_value=jnp.array(0.0),
            return_target=jnp.array(100.0),
            clip_range=hyperparameters.clip_range,
        )

        assert float(metrics.critic_loss) == pytest.approx(float(expected_critic), rel=1e-6)
