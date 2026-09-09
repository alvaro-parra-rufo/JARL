"""Subprocess command builders for agentic case execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

AGENTIC_CASES_MODULE = "jarl.agentic.cases"
"""Default module invoked by `python -m` for agentic cases."""

__all__ = [
    "AGENTIC_CASES_MODULE",
    "AgenticCaseLaunchRequest",
    "agentic_case_log_path",
    "build_agentic_case_command",
]


@dataclass(frozen=True, slots=True)
class AgenticCaseLaunchRequest:
    """Inputs required to launch one isolated agentic case."""

    case_id: str
    destination: Path
    package: str | None = None
    prompts: tuple[str, ...] | None = None
    llm_config: str | None = None
    launch_options: dict[str, object] | None = None


def agentic_case_log_path(destination: str | Path) -> Path:
    """Return a sibling log path that keeps the case destination empty."""
    experiment_dir = Path(destination)
    return experiment_dir.parent / f".{experiment_dir.name}.agentic_case.log"


def build_agentic_case_command(
    request: AgenticCaseLaunchRequest,
    *,
    python_executable: str,
    run_module: str = AGENTIC_CASES_MODULE,
) -> list[str]:
    """Build argv for executing an agentic case in a subprocess."""
    command = [
        python_executable,
        "-m",
        run_module,
        "run",
        request.case_id,
        "--destination",
        str(request.destination),
        "--json",
    ]
    if request.package is not None:
        command.extend(["--package", request.package])
    if request.llm_config is not None:
        command.extend(["--llm-config", request.llm_config])
    if request.prompts is not None:
        for prompt in request.prompts:
            command.extend(["--prompt", prompt])
    if request.launch_options:
        for key, value in sorted(request.launch_options.items()):
            command.extend(["--launch-option", f"{key}={value}"])
    return command
