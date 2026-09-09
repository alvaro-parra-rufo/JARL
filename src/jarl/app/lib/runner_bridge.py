"""Subprocess bridge between the UI and ``jarl.training.runner``."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from jarl.agents.ppo.inference.requests import (
    INFERENCE_LOG_NAME,
    InferenceLaunchRequest,
    build_inference_command,
)
from jarl.app.lib.live_ui import prepare_live_feeds_for_launch
from jarl.app.lib.session import (
    INFERENCE_MODULE,
    ActiveInferenceRun,
    ActiveRun,
    env_with_wandb_secrets,
    set_active_inference_run,
    set_active_run,
)
from jarl.training.config import RLRunConfig
from jarl.training.launch import (
    TRAINING_RUN_MODULE,
    build_create_and_train_command,
    build_resume_command,
    build_train_current_command,
    write_run_config,
)

__all__ = [
    "launch_create_and_train",
    "launch_inference_video",
    "launch_resume",
    "launch_train_current",
]


def _launch(command: list[str], *, log_path: Path) -> ActiveRun:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        env=env_with_wandb_secrets(),
    )
    run = ActiveRun(process=process, log_path=log_path)
    set_active_run(run)
    prepare_live_feeds_for_launch(
        log_paths=[log_path],
        terminal_buffer_keys=["jarl_terminal_runner_lab"],
    )
    return run


def launch_create_and_train(
    *,
    experiment_dir: Path,
    config: RLRunConfig,
    label: str = "",
) -> ActiveRun:
    """Start a subprocess that creates a root node and trains it."""
    config_path = experiment_dir.parent / f".{experiment_dir.name}_pending_config.json"
    write_run_config(config, config_path)
    log_path = experiment_dir / "runner_lab.log"
    command = build_create_and_train_command(
        experiment_dir=experiment_dir,
        config_path=config_path,
        python_executable=sys.executable,
        run_module=TRAINING_RUN_MODULE,
        label=label,
    )
    return _launch(command, log_path=log_path)


def launch_train_current(
    *,
    experiment_dir: Path,
    config: RLRunConfig,
    node_id: str | None = None,
) -> ActiveRun:
    """Start a subprocess that trains the current or selected node."""
    config_path = experiment_dir / ".runner_lab_pending_config.json"
    write_run_config(config, config_path)
    log_path = experiment_dir / "runner_lab.log"
    command = build_train_current_command(
        experiment_dir=experiment_dir,
        config_path=config_path,
        python_executable=sys.executable,
        run_module=TRAINING_RUN_MODULE,
        node_id=node_id,
    )
    return _launch(command, log_path=log_path)


def launch_resume(
    *,
    experiment_dir: Path,
    node_id: str,
    config: RLRunConfig,
) -> ActiveRun:
    """Start a subprocess that resumes a failed or interrupted node."""
    config_path = experiment_dir / ".runner_lab_resume_config.json"
    write_run_config(config, config_path)
    log_path = experiment_dir / "runner_lab.log"
    command = build_resume_command(
        experiment_dir=experiment_dir,
        config_path=config_path,
        node_id=node_id,
        python_executable=sys.executable,
        run_module=TRAINING_RUN_MODULE,
    )
    return _launch(command, log_path=log_path)


def launch_inference_video(
    *,
    experiment_dir: Path,
    request: InferenceLaunchRequest,
) -> ActiveInferenceRun:
    """Start a subprocess that renders greedy checkpoint videos."""
    log_path = experiment_dir / INFERENCE_LOG_NAME
    command = build_inference_command(
        request,
        experiment_dir=experiment_dir,
        inference_module=INFERENCE_MODULE,
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
    run = ActiveInferenceRun(
        process=process,
        log_path=log_path,
        node_id=request.node_id,
        name_prefix=request.name_prefix,
        checkpoint_step=request.checkpoint_step,
    )
    set_active_inference_run(run)
    prepare_live_feeds_for_launch(
        log_paths=[log_path],
        terminal_buffer_keys=["jarl_terminal_inference"],
    )
    return run
