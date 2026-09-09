"""Tests for agentic case subprocess command builders."""

from __future__ import annotations

from pathlib import Path

from jarl.agentic.cases import (
    AgenticCaseLaunchRequest,
    agentic_case_log_path,
    build_agentic_case_command,
)


class TestAgenticCaseLaunch:
    def test_build_command_uses_module_entrypoint_and_json_result(
        self,
        tmp_path: Path,
    ) -> None:
        destination = tmp_path / "case-run"
        request = AgenticCaseLaunchRequest(
            case_id="graph_extend_same_branch",
            destination=destination,
            package="custom.cases",
        )

        command = build_agentic_case_command(
            request,
            python_executable="/venv/bin/python",
        )

        assert command == [
            "/venv/bin/python",
            "-m",
            "jarl.agentic.cases",
            "run",
            "graph_extend_same_branch",
            "--destination",
            str(destination),
            "--json",
            "--package",
            "custom.cases",
        ]

    def test_build_command_includes_prompt_overrides(self, tmp_path: Path) -> None:
        destination = tmp_path / "case-run"
        request = AgenticCaseLaunchRequest(
            case_id="env_navix_maps_read",
            destination=destination,
            prompts=("Busca mapas de lava.",),
        )

        command = build_agentic_case_command(
            request,
            python_executable="/venv/bin/python",
        )

        assert command[-2:] == ["--prompt", "Busca mapas de lava."]

    def test_build_command_includes_llm_config(self, tmp_path: Path) -> None:
        destination = tmp_path / "case-run"
        catalog = tmp_path / "custom.yaml"
        request = AgenticCaseLaunchRequest(
            case_id="env_navix_maps_read",
            destination=destination,
            llm_config=str(catalog),
        )

        command = build_agentic_case_command(
            request,
            python_executable="/venv/bin/python",
        )

        assert command[-2:] == ["--llm-config", str(catalog)]

    def test_log_path_is_sibling_of_empty_destination(self, tmp_path: Path) -> None:
        destination = tmp_path / "case-run"

        log_path = agentic_case_log_path(destination)

        assert log_path == tmp_path / ".case-run.agentic_case.log"
