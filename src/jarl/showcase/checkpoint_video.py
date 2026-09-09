"""Subprocess wrapper for checkpoint greedy video rendering."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

INFERENCE_MODULE = "jarl.agents.ppo.inference"
"""Default module invoked by ``python -m`` for showcase inference subprocesses."""

__all__ = ["INFERENCE_MODULE", "render_checkpoint_video_subprocess"]


def render_checkpoint_video_subprocess(
    *,
    experiment_dir: Path,
    node_id: str,
    checkpoint_step: int,
    env_id: str,
    role: str,
    name_prefix: str,
    log_path: Path,
    seed: int | None = None,
    load_from_node_id: str | None = None,
    use_parent_checkpoint: bool = False,
    parent_id: str | None = None,
    child_id: str | None = None,
) -> int:
    """Invoke ``python -m jarl.agents.ppo.inference`` and return the exit code."""
    command = [
        sys.executable,
        "-m",
        INFERENCE_MODULE,
        "--experiment-dir",
        str(experiment_dir),
        "--node-id",
        node_id,
        "--checkpoint-step",
        str(checkpoint_step),
        "--env-id",
        env_id,
        "--role",
        role,
        "--name-prefix",
        name_prefix,
    ]
    if seed is not None:
        command.extend(["--seed", str(seed)])
    if load_from_node_id is not None:
        command.extend(["--load-from-node-id", load_from_node_id])
    if use_parent_checkpoint:
        command.append("--use-parent-checkpoint")
    if parent_id is not None:
        command.extend(["--parent-id", parent_id])
    if child_id is not None:
        command.extend(["--child-id", child_id])

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        completed = subprocess.run(  # noqa: S603
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=_subprocess_env(),
        )
    return int(completed.returncode)


def _subprocess_env() -> dict[str, str]:
    """Return a subprocess environment with JAX platform defaults."""
    env = os.environ.copy()
    env.setdefault("JAX_PLATFORMS", "cuda,cpu")
    return env
