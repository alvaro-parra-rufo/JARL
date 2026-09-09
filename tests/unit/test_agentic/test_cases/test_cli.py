"""Tests for the agentic case CLI."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from jarl.agentic.cases import AgenticCaseResult, get_agentic_case_registry
from jarl.agentic.cases.cli import (
    AGENTIC_CASE_EXIT_ERROR,
    AGENTIC_CASE_EXIT_FAILED,
    AGENTIC_CASE_EXIT_INVALID,
    AGENTIC_CASE_EXIT_PASSED,
    main,
)
from jarl.agentic.cases.services import AgenticCaseStatus


class TestAgenticCasesCli:
    def test_list_emits_json_catalog(self, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code = main(["list", "--json"])

        payload = json.loads(capsys.readouterr().out)
        assert exit_code == AGENTIC_CASE_EXIT_PASSED
        assert len(payload["cases"]) == len(get_agentic_case_registry())

    def test_describe_unknown_case_returns_invalid_code(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        exit_code = main(["describe", "missing"])

        captured = capsys.readouterr()
        assert exit_code == AGENTIC_CASE_EXIT_INVALID
        assert "missing" in captured.err

    @pytest.mark.parametrize(
        ("status", "expected_exit"),
        [
            pytest.param("passed", AGENTIC_CASE_EXIT_PASSED, id="passed"),
            pytest.param("failed", AGENTIC_CASE_EXIT_FAILED, id="failed"),
            pytest.param("error", AGENTIC_CASE_EXIT_ERROR, id="error"),
        ],
    )
    def test_run_maps_result_status_to_exit_code(
        self,
        status: AgenticCaseStatus,
        expected_exit: int,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        mocker: MockerFixture,
    ) -> None:
        destination = tmp_path / "run"
        result = AgenticCaseResult(
            case_id="graph_extend_same_branch",
            experiment_dir=destination,
            status=status,
            failures=("Expected one child.",) if status == "failed" else (),
            error="RuntimeError: unavailable" if status == "error" else None,
            started_at="2026-08-07T10:00:00+00:00",
            finished_at="2026-08-07T10:00:01+00:00",
        )
        mocker.patch(
            "jarl.agentic.cases.cli.execute_agentic_case",
            return_value=result,
        )

        exit_code = main(
            [
                "run",
                result.case_id,
                "--destination",
                str(destination),
                "--json",
            ]
        )

        payload = json.loads(capsys.readouterr().out)
        assert exit_code == expected_exit
        assert payload["result"]["status"] == status

    def test_run_forwards_llm_config(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        destination = tmp_path / "run"
        catalog = tmp_path / "custom.yaml"
        catalog.write_text(
            "\n".join(
                [
                    "profiles:",
                    "  cheap:",
                    "    provider: ollama",
                    "    model: qwen",
                    "fallback: cheap",
                ]
            ),
            encoding="utf-8",
        )
        monkeypatch.delenv("JARL_LLM_CONFIG", raising=False)
        execute = mocker.patch(
            "jarl.agentic.cases.cli.execute_agentic_case",
            return_value=AgenticCaseResult(
                case_id="graph_extend_same_branch",
                experiment_dir=destination,
                status="passed",
                failures=(),
                error=None,
                started_at="2026-08-07T10:00:00+00:00",
                finished_at="2026-08-07T10:00:01+00:00",
            ),
        )

        exit_code = main(
            [
                "run",
                "graph_extend_same_branch",
                "--destination",
                str(destination),
                "--llm-config",
                str(catalog),
                "--json",
            ]
        )

        assert exit_code == AGENTIC_CASE_EXIT_PASSED
        assert execute.call_args.kwargs.get("llm_settings") is None
        assert Path(os.environ["JARL_LLM_CONFIG"]) == catalog.resolve()
        os.environ.pop("JARL_LLM_CONFIG", None)
