"""Training subprocess command builders and config snapshots."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from jarl.training.config import RLRunConfig

TRAINING_RUN_MODULE = "jarl.training.run"
"""Default module invoked by ``python -m`` for training subprocesses."""

TRAIN_DETACH_ENV = "JARL_TRAIN_DETACH"
"""When set to a truthy value, MCP ``train_run`` / ``train_resume`` spawn a subprocess."""

TRAIN_WORKER_PID_NAME = ".train_worker.pid"
"""Experiment-level pid file for a live detached training worker."""

__all__ = [
    "TRAINING_RUN_MODULE",
    "TRAIN_DETACH_ENV",
    "TRAIN_WORKER_PID_NAME",
    "SpawnedTraining",
    "build_create_and_train_command",
    "build_resume_command",
    "build_train_current_command",
    "clear_train_worker_pid",
    "live_training_pid",
    "spawn_training",
    "train_detach_enabled",
    "write_run_config",
]

_LIVE_WORKERS: dict[Path, subprocess.Popen[str]] = {}
"""In-process Popen handles keyed by resolved experiment directory."""


@dataclass(frozen=True, slots=True)
class SpawnedTraining:
    """Handle for a detached training subprocess."""

    pid: int
    log_path: Path
    config_path: Path


def train_detach_enabled() -> bool:
    """Return whether training operations should spawn a subprocess."""
    raw = os.environ.get(TRAIN_DETACH_ENV, "").strip().lower()
    return raw in {"1", "true", "yes"}


def live_training_pid(experiment_dir: Path) -> int | None:
    """Return the pid of a still-running detached worker, if any.

    Reaps a finished child kept by this process (including zombies) even when
    the pid file is already gone, then removes a matching pid file.
    """
    key = experiment_dir.resolve()
    process = _LIVE_WORKERS.get(key)
    if process is not None:
        if process.poll() is None:
            return process.pid
        process.wait()
        _LIVE_WORKERS.pop(key, None)
        clear_train_worker_pid(key, pid=process.pid)
        return None
    recorded = _read_pidfile(key)
    if recorded is None:
        return None
    try:
        os.kill(recorded, 0)
    except OSError:
        clear_train_worker_pid(key, pid=recorded)
        return None
    return recorded


def clear_train_worker_pid(experiment_dir: Path, *, pid: int | None = None) -> None:
    """Remove ``.train_worker.pid`` when it still refers to ``pid`` (or this process)."""
    path = experiment_dir.resolve() / TRAIN_WORKER_PID_NAME
    if not path.is_file():
        return
    try:
        recorded = int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return
    expected = os.getpid() if pid is None else pid
    if recorded != expected:
        return
    path.unlink(missing_ok=True)


def _read_pidfile(experiment_dir: Path) -> int | None:
    """Return the pid recorded for ``experiment_dir``, if the file is valid."""
    path = experiment_dir.resolve() / TRAIN_WORKER_PID_NAME
    if not path.is_file():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def write_run_config(config: RLRunConfig, path: Path) -> Path:
    """Persist a run config snapshot for subprocess consumption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.model_dump(), indent=2), encoding="utf-8")
    return path


def build_create_and_train_command(
    *,
    experiment_dir: Path,
    config_path: Path,
    python_executable: str,
    run_module: str = TRAINING_RUN_MODULE,
    label: str = "",
) -> list[str]:
    """Build argv for creating a root node and training it."""
    command = [
        python_executable,
        "-m",
        run_module,
        "--experiment-dir",
        str(experiment_dir),
        "--config-json",
        str(config_path),
        "--create-root",
    ]
    if label:
        command.extend(["--label", label])
    return command


def build_train_current_command(
    *,
    experiment_dir: Path,
    config_path: Path,
    python_executable: str,
    run_module: str = TRAINING_RUN_MODULE,
    node_id: str | None = None,
) -> list[str]:
    """Build argv for training the current or selected node."""
    command = [
        python_executable,
        "-m",
        run_module,
        "--experiment-dir",
        str(experiment_dir),
        "--config-json",
        str(config_path),
    ]
    if node_id is not None:
        command.extend(["--node-id", node_id])
    return command


def build_resume_command(
    *,
    experiment_dir: Path,
    config_path: Path,
    node_id: str,
    python_executable: str,
    run_module: str = TRAINING_RUN_MODULE,
) -> list[str]:
    """Build argv for resuming a failed or interrupted node."""
    return [
        python_executable,
        "-m",
        run_module,
        "--experiment-dir",
        str(experiment_dir),
        "--config-json",
        str(config_path),
        "--resume",
        "--node-id",
        node_id,
    ]


def spawn_training(
    *,
    command: list[str],
    experiment_dir: Path,
    config_path: Path,
    log_path: Path | None = None,
) -> SpawnedTraining:
    """Start a training CLI subprocess that outlives the MCP host process.

    Uses a new session so a cancelled MCP tool call does not SIGTERM the worker.
    """
    live = live_training_pid(experiment_dir)
    if live is not None:
        msg = f"Training worker pid {live} is still running. Wait for it to finish."
        raise RuntimeError(msg)
    key = experiment_dir.resolve()
    resolved_log = log_path if log_path is not None else key / "runner_lab.log"
    resolved_log.parent.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy()
    child_env.pop(TRAIN_DETACH_ENV, None)
    with resolved_log.open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(  # noqa: S603
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            text=True,
            env=child_env,
        )
    pid_path = key / TRAIN_WORKER_PID_NAME
    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    _LIVE_WORKERS[key] = process
    return SpawnedTraining(pid=process.pid, log_path=resolved_log, config_path=config_path)
