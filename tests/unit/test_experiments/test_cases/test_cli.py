"""Tests for the experiment case CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarl.experiments.cases.cli import (
    CASE_EXIT_INVALID,
    CASE_EXIT_SUCCESS,
    main,
)


class TestExperimentCasesCli:
    def test_list_emits_json_catalog(self, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code = main(["list", "--json"])

        payload = json.loads(capsys.readouterr().out)
        assert exit_code == CASE_EXIT_SUCCESS
        assert {case["id"] for case in payload["cases"]} == {
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

    def test_describe_unknown_case_returns_invalid_code(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        exit_code = main(["describe", "missing"])

        captured = capsys.readouterr()
        assert exit_code == CASE_EXIT_INVALID
        assert "missing" in captured.err

    def test_materialize_uses_explicit_destination(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        destination = tmp_path / "empty"

        exit_code = main(
            [
                "materialize",
                "empty_experiment",
                "--destination",
                str(destination),
                "--json",
            ]
        )

        payload = json.loads(capsys.readouterr().out)
        assert exit_code == CASE_EXIT_SUCCESS
        assert payload["result"]["experiment_dir"] == str(destination.resolve())
        assert (destination / "experiment.json").is_file()
