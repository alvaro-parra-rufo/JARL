"""Tests for the Runner Lab agentic case subprocess bridge."""

from __future__ import annotations

import subprocess
from pathlib import Path

from pytest_mock import MockerFixture

from jarl.agentic.cases import AgenticCaseLaunchRequest
from jarl.app.lib.agentic_case_bridge import launch_agentic_case


def test_launch_agentic_case_uses_core_builder_and_stores_run(
    tmp_path: Path,
    mocker: MockerFixture,
) -> None:
    destination = tmp_path / "case-run"
    request = AgenticCaseLaunchRequest(
        case_id="graph_extend_same_branch",
        destination=destination,
    )
    process = mocker.Mock(spec=subprocess.Popen)
    popen = mocker.patch(
        "jarl.app.lib.agentic_case_bridge.subprocess.Popen",
        return_value=process,
    )
    set_active = mocker.patch(
        "jarl.app.lib.agentic_case_bridge.set_active_agentic_case_run",
    )
    prepare = mocker.patch(
        "jarl.app.lib.agentic_case_bridge.prepare_live_feeds_for_launch",
    )
    mocker.patch(
        "jarl.app.lib.agentic_case_bridge.env_with_wandb_secrets",
        return_value={"JARL_LLM_PROVIDER": "ollama"},
    )

    run = launch_agentic_case(request)

    assert run.process is process
    assert run.case_id == request.case_id
    assert run.experiment_dir == destination
    assert run.log_path == tmp_path / ".case-run.agentic_case.log"
    assert popen.call_args.args[0][3:6] == [
        "run",
        "graph_extend_same_branch",
        "--destination",
    ]
    set_active.assert_called_once_with(run)
    prepare.assert_called_once_with(
        log_paths=[run.log_path],
        terminal_buffer_keys=["jarl_terminal_agentic_case"],
    )
