"""Tests for the stdio MCP server entrypoint."""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from jarl.agentic.errors import ExperimentNotFoundError
from jarl.agentic.workflow import AgenticWorkflow
from jarl.logging import console
from jarl.mcp.server import (
    JARL_MCP_EXPERIMENT_DIR_ENV,
    SERVER_INSTRUCTIONS,
    configure_stderr_logging,
    create_server,
    main,
    workflow_from_env,
)


class TestCreateServer:
    def test_create_server_uses_jarl_name_and_instructions(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)

        server = create_server(workflow=workflow)

        assert server.name == "jarl"
        assert server.instructions == SERVER_INSTRUCTIONS
        assert JARL_MCP_EXPERIMENT_DIR_ENV in SERVER_INSTRUCTIONS

    def test_instructions_are_host_agnostic(self) -> None:
        lowered = SERVER_INSTRUCTIONS.lower()

        assert "cursor" not in lowered
        assert "codex" not in lowered

    def test_create_server_does_not_write_to_stdout(
        self,
        prepared_experiment: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)

        create_server(workflow=workflow)

        captured = capsys.readouterr()

        assert captured.out == ""

    def test_configure_stderr_logging_keeps_jarl_console_off_stdout(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        configure_stderr_logging()
        console.info("mcp-stderr-probe")

        captured = capsys.readouterr()

        assert captured.out == ""
        assert "mcp-stderr-probe" in captured.err


class TestWorkflowFromEnv:
    def test_missing_env_raises(self, mocker: MockerFixture) -> None:
        mocker.patch.dict("os.environ", {}, clear=True)

        with pytest.raises(ExperimentNotFoundError, match=JARL_MCP_EXPERIMENT_DIR_ENV):
            workflow_from_env()

    def test_loads_experiment(self, prepared_experiment: Path, mocker: MockerFixture) -> None:
        mocker.patch.dict("os.environ", {JARL_MCP_EXPERIMENT_DIR_ENV: str(prepared_experiment)})

        workflow = workflow_from_env()

        assert workflow.exp_dir == prepared_experiment.resolve()


class TestMain:
    def test_main_runs_stdio(self, mocker: MockerFixture) -> None:
        server = mocker.Mock()
        mocker.patch("jarl.mcp.server.create_server", return_value=server)

        exit_code = main()

        server.run.assert_called_once_with(transport="stdio")
        assert exit_code == 0

    def test_main_missing_experiment_returns_2(self, mocker: MockerFixture) -> None:
        mocker.patch.dict("os.environ", {}, clear=True)

        exit_code = main()

        assert exit_code == 2

    def test_main_module_reexports_server_main(self) -> None:
        from jarl.mcp.__main__ import main as module_main

        assert module_main is main
