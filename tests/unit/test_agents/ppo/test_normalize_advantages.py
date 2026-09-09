"""Regression tests for advantage normalization behavior (Fase 4)."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from jarl.agents.ppo.core.minibatch import normalize_advantages


class TestNormalizeAdvantagesRegression:
    """Document current normalization behavior without changing production defaults."""

    def test_normalize_advantages_has_zero_mean_and_unit_std(self) -> None:
        advantages = jnp.array([1.0, 2.0, 3.0, 4.0])

        normalized = normalize_advantages(advantages)

        assert float(jnp.mean(normalized)) == pytest.approx(0.0, abs=1e-6)
        assert float(jnp.std(normalized)) == pytest.approx(1.0, rel=1e-5)

    def test_normalize_advantages_near_zero_variance_uses_epsilon_denominator(self) -> None:
        """When input std is below ``1e-8``, normalization is dominated by the epsilon floor."""
        advantages = jnp.array([0.0, 0.0, 0.0, 1e-12])

        normalized = normalize_advantages(advantages)
        input_centered = advantages - jnp.mean(advantages)
        input_scale = float(jnp.max(jnp.abs(input_centered)))
        output_scale = float(jnp.max(jnp.abs(normalized)))

        assert float(jnp.mean(normalized)) == pytest.approx(0.0, abs=1e-6)
        assert output_scale > max(input_scale * 1e4, 1e-5)

    def test_normalize_advantages_with_outlier_preserves_zero_mean(self) -> None:
        advantages = jnp.array([0.0, 0.0, 0.0, 100.0])

        normalized = normalize_advantages(advantages)

        assert float(jnp.mean(normalized)) == pytest.approx(0.0, abs=1e-6)
        assert float(jnp.std(normalized)) == pytest.approx(1.0, rel=1e-5)
        assert float(normalized[-1]) > float(normalized[0])
