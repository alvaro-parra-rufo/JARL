"""Regression tests for checkpoint video kind validation."""

from __future__ import annotations

from jarl.agents.ppo.checkpoint_state import CHECKPOINT_KIND_PPO, PPOCheckpoint, checkpoint_kind_for_algorithm


def test_checkpoint_kind_matches_feedforward_constant() -> None:
    assert checkpoint_kind_for_algorithm("ppo.full_jax.navix") == CHECKPOINT_KIND_PPO


def test_ppo_checkpoint_class_kind_is_not_string_constant() -> None:
    """Guard against comparing ``PPOCheckpoint.kind`` (field descriptor) to kind strings."""
    assert PPOCheckpoint.kind is not CHECKPOINT_KIND_PPO
    assert PPOCheckpoint.kind != CHECKPOINT_KIND_PPO
