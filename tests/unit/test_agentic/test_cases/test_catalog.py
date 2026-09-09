"""Tests for the lazy built-in agentic case catalog."""

from __future__ import annotations

from jarl.agentic.cases import get_agentic_case_registry


class TestAgenticCaseCatalog:
    def test_returns_cached_builtin_registry(self) -> None:
        first = get_agentic_case_registry()
        second = get_agentic_case_registry()

        assert first is second
        assert first.ids() == frozenset(
            {
                "graph_checkpoint_rollout_analysis",
                "graph_checkpoint_rollout_analysis_long",
                "graph_create_root_baseline",
                "graph_extend_recover_override",
                "graph_extend_same_branch",
                "graph_extend_ppo_overrides",
                "graph_fork_from_latest_not_best",
                "graph_fork_parallel_branch",
                "graph_fork_ppo_overrides",
                "graph_read_without_mutation",
                "train_recovery_status_resume",
                "subagent_metrics_analysis_improving",
                "env_navix_maps_read",
                "env_navix_maps_search_key",
                "env_navix_maps_query_lava",
                "env_navix_maps_difficulty_easy",
                "env_navix_maps_transfer_ready",
                "env_navix_maps_filters_combo",
                "navix_empty_variant_spec",
            }
        )

    def test_builtin_cases_leave_objective_empty(self) -> None:
        for case in get_agentic_case_registry().values():
            assert case.spec.objective == "", case.spec.id
