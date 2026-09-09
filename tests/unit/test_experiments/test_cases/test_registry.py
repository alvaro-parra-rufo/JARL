"""Tests for experiment case discovery and filtering."""

from __future__ import annotations

import pytest

from jarl.experiments.cases import CaseRegistry, ExperimentCase
from jarl.experiments.cases.builtin.empty_experiment import EmptyExperimentCase

_BUILTIN_PACKAGE = "jarl.experiments.cases.builtin"


class TestCaseRegistryDiscovery:
    def test_discovers_explicit_builtin_cases(self) -> None:
        registry = CaseRegistry.from_package(
            _BUILTIN_PACKAGE,
            case_type=ExperimentCase,
        )

        assert registry.ids() == frozenset(
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

    def test_rejects_duplicate_case_ids(self) -> None:
        case = EmptyExperimentCase()

        with pytest.raises(ValueError, match="Duplicate case id"):
            CaseRegistry([case, case])

    def test_rejects_leaf_without_case_export(self) -> None:
        with pytest.raises(ValueError, match="missing CASE"):
            CaseRegistry.from_package(
                "tests.helpers.experiment_cases_broken",
                case_type=ExperimentCase,
            )

    def test_rejects_non_package_module(self) -> None:
        with pytest.raises(ValueError, match="no __path__"):
            CaseRegistry.from_package(
                "jarl.experiments.cases.spec",
                case_type=ExperimentCase,
            )

    def test_get_missing_case_raises(self) -> None:
        registry = CaseRegistry[ExperimentCase]()

        with pytest.raises(KeyError):
            registry.get("missing")


class TestCaseRegistryFiltering:
    @pytest.fixture()
    def registry(self) -> CaseRegistry[ExperimentCase]:
        return CaseRegistry.from_package(
            _BUILTIN_PACKAGE,
            case_type=ExperimentCase,
        )

    def test_filters_by_tag(self, registry: CaseRegistry[ExperimentCase]) -> None:
        filtered = registry.filter_by(include_tags={"tree"})

        assert filtered.ids() == frozenset({"demo_tree"})

    def test_filters_by_id_pattern(self, registry: CaseRegistry[ExperimentCase]) -> None:
        filtered = registry.filter_by(include_ids={"*root"})

        assert filtered.ids() == frozenset({"navix_prepared_root"})

    def test_combines_id_and_tag_filters(self, registry: CaseRegistry[ExperimentCase]) -> None:
        filtered = registry.filter_by(
            include_ids={"navix_*"},
            include_tags={"prepared"},
        )

        assert filtered.ids() == frozenset(
            {
                "navix_empty_variant",
                "navix_ppo_gru_rollout_checkpoint",
                "navix_ppo_rollout_checkpoint",
                "navix_prepared_root",
                "navix_root_best_vs_latest",
                "navix_root_improving_eval",
            }
        )
