"""Streamlit subprocess bridge for agentic case evaluation."""

from __future__ import annotations

import subprocess
import sys

from jarl.agentic.cases import (
    AgenticCaseLaunchRequest,
    agentic_case_log_path,
    build_agentic_case_command,
)
from jarl.agentic.progress import progress_path
from jarl.app.lib.live_feed import reset_log_feed
from jarl.app.lib.live_ui import prepare_live_feeds_for_launch
from jarl.app.lib.session import (
    ActiveAgenticCaseRun,
    env_with_wandb_secrets,
    set_active_agentic_case_run,
)

__all__ = ["AGENTIC_PROGRESS_BUFFER_KEY", "launch_agentic_case"]

AGENTIC_PROGRESS_BUFFER_KEY = "jarl_terminal_agentic_progress"


def launch_agentic_case(
    request: AgenticCaseLaunchRequest,
) -> ActiveAgenticCaseRun:
    """Start one agentic case in a background subprocess."""
    log_path = agentic_case_log_path(request.destination)
    command = build_agentic_case_command(
        request,
        python_executable=sys.executable,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        env=env_with_wandb_secrets(),
    )
    run = ActiveAgenticCaseRun(
        process=process,
        log_path=log_path,
        case_id=request.case_id,
        experiment_dir=request.destination,
    )
    set_active_agentic_case_run(run)
    prepare_live_feeds_for_launch(
        log_paths=[log_path],
        terminal_buffer_keys=["jarl_terminal_agentic_case"],
    )
    reset_log_feed(progress_path(request.destination), AGENTIC_PROGRESS_BUFFER_KEY)
    return run
