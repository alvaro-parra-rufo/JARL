"""Tests for experiment case application services."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarl.experiments.cases import (
    allocate_case_destination,
    describe_experiment_case,
    list_experiment_cases,
    materialize_experiment_case,
)


class TestExperimentCaseCatalogServices:
    def test_list_exposes_stable_builtin_metadata(self) -> None:
        cases = list_experiment_cases()

        assert {case.id for case in cases} == {
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
        assert all(case.config_type == "RLRunConfig" for case in cases)

    def test_describe_returns_requested_case(self) -> None:
        case = describe_experiment_case("empty_experiment")

        assert case.id == "empty_experiment"
        assert case.title
        assert case.to_dict()["id"] == "empty_experiment"


class TestExperimentCaseMaterializationServices:
    def test_allocate_reserves_top_level_empty_directory(self, tmp_path: Path) -> None:
        destination = allocate_case_destination(
            "empty_experiment",
            cases_root=tmp_path,
            run_name="case-run",
        )

        assert destination == tmp_path / "case-run"
        assert destination.is_dir()
        assert not any(destination.iterdir())

    @pytest.mark.parametrize("run_name", ["nested/name", ".."])
    def test_allocate_rejects_non_component_run_name(
        self,
        tmp_path: Path,
        run_name: str,
    ) -> None:
        with pytest.raises(ValueError, match="single path component"):
            allocate_case_destination(
                "empty_experiment",
                cases_root=tmp_path,
                run_name=run_name,
            )

    def test_materialize_returns_aliases_and_experiment_path(self, tmp_path: Path) -> None:
        destination = tmp_path / "prepared"

        result = materialize_experiment_case(
            "navix_prepared_root",
            destination,
        )

        assert result.experiment_dir == destination.resolve()
        assert set(result.aliases) == {"root"}
        assert (destination / "experiment.json").is_file()
