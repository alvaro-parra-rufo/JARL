"""Tests for rollout target debug statistics (Fase 3 critic stability)."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from jarl.agents.ppo.core.metrics import aggregate_optimization_metrics, compute_rollout_target_stats


class TestRolloutTargetStats:
    """Rollout-level value/return/advantage summaries for debug logging."""

    def test_compute_rollout_target_stats_matches_tensor_statistics(self) -> None:
        values = jnp.array([[0.0, 2.0], [4.0, 6.0]])
        returns = jnp.array([[1.0, 3.0], [5.0, 7.0]])
        advantages = jnp.array([[0.0, 1.0], [-1.0, 2.0]])

        stats = compute_rollout_target_stats(values, returns, advantages)

        assert float(stats["rollout/value_mean"]) == pytest.approx(3.0, rel=1e-6)
        assert float(stats["rollout/return_mean"]) == pytest.approx(4.0, rel=1e-6)
        assert float(stats["rollout/advantage_std"]) == pytest.approx(float(jnp.std(advantages)), rel=1e-6)


class TestAggregateOptimizationMetrics:
    """Optimization scan aggregation keeps policy ratio max across minibatches."""

    def test_aggregate_optimization_metrics_uses_max_for_policy_ratio(self) -> None:
        metrics = {
            "loss/critic_loss": jnp.array([1.0, 3.0]),
            "policy_ratio/max": jnp.array([1.1, 2.5]),
        }

        aggregated = aggregate_optimization_metrics(metrics)

        assert float(aggregated["loss/critic_loss"]) == pytest.approx(2.0, rel=1e-6)
        assert float(aggregated["policy_ratio/max"]) == pytest.approx(2.5, rel=1e-6)
