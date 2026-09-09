"""Tests for the lazy built-in experiment case catalog."""

from __future__ import annotations

from jarl.experiments.cases import get_experiment_case_registry


class TestExperimentCaseCatalog:
    def test_returns_cached_registry(self) -> None:
        first = get_experiment_case_registry()
        second = get_experiment_case_registry()

        assert first is second
        assert first.ids() == frozenset(
            {
                "demo_tree",
                "empty_experiment",
                "navix_ppo_gru_rollout_checkpoint",
                "navix_ppo_rollout_checkpoint",
                "navix_ppo_rollout_trained_checkpoint",
                "navix_ppo_rollout_long_trained_checkpoint",
                "navix_empty_variant",
                "navix_prepared_root",
                "navix_root_best_vs_latest",
                "navix_root_failed_resumable",
                "navix_root_improving_eval",
            }
        )
