"""Shared PPO override expectations for graph extend and fork agentic cases."""

from __future__ import annotations

__all__ = ["EXPECTED_PPO_OVERRIDES"]

EXPECTED_PPO_OVERRIDES: dict[str, float] = {
    "algorithm.learning_rate": 0.001,
    "algorithm.gamma": 0.97,
    "algorithm.entropy_coef": 0.08,
    "algorithm.gae_lambda": 0.9,
    "algorithm.clip_range": 0.15,
}
"""Dotted overrides requested by the PPO override scenarios."""
