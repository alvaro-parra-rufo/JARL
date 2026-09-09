"""Run real JAX training out of process so tests can use GPU.

Pytest workers pin ``JAX_PLATFORMS=cpu`` before JAX imports. A child process
can select CUDA instead. GPU runs take a process lock so ``pytest -n`` does
not share a small GPU across workers.
"""

from __future__ import annotations

import contextlib
import fcntl
import functools
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

from jarl.experiments.cases.case import ExperimentCase, ExperimentCaseContext
from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.node import NodeWorkspace
from jarl.training.config import RLRunConfig
from jarl.training.launch import build_create_and_train_command, write_run_config

_GPU_LOCK_PATH = Path(tempfile.gettempdir()) / "jarl-test-gpu.lock"
"""Shared exclusive lock for CUDA probes and GPU training subprocesses."""

__all__ = [
    "cuda_available",
    "jax_training_env",
    "materialize_case_in_training_subprocess",
    "run_command_with_jax_training_env",
    "run_create_and_train_in_subprocess",
]


@functools.cache
def cuda_available() -> bool:
    """Return whether a child process can initialize the JAX CUDA backend."""
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cuda,cpu"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    probe = "import jax; raise SystemExit(0 if jax.default_backend() == 'gpu' else 1)"
    with _exclusive_lock(_GPU_LOCK_PATH):
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            check=False,
            capture_output=True,
            env=env,
            text=True,
        )
    return completed.returncode == 0


def jax_training_env() -> dict[str, str]:
    """Return subprocess environment for one training run.

    Uses CUDA when the probe succeeds; otherwise keeps the child on CPU.
    GPU children keep CPU listed so host callbacks inside the JIT loop work.
    """
    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    if cuda_available():
        env["JAX_PLATFORMS"] = "cuda,cpu"
        env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"
    else:
        env["JAX_PLATFORMS"] = "cpu"
    return env


def run_command_with_jax_training_env(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run ``command`` with the training env, locking when the child uses GPU.

    Args:
        command: Argv for ``subprocess.run``.

    Returns:
        Completed process with captured stdout and stderr.

    Raises:
        RuntimeError: If the child exits non-zero.
    """
    env = jax_training_env()
    with _gpu_lock(env):
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            env=env,
            text=True,
        )
    if completed.returncode != 0:
        joined = " ".join(command)
        msg = (
            f"Training subprocess failed ({completed.returncode}): {joined}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
        raise RuntimeError(msg)
    return completed


def run_create_and_train_in_subprocess(
    experiment_dir: Path,
    config: RLRunConfig,
    *,
    label: str = "",
) -> NodeWorkspace:
    """Create a root and train it in a GPU-capable child process.

    Args:
        experiment_dir: Empty experiment directory for the run.
        config: Run config snapshot passed via ``--config-json``.
        label: Optional root label forwarded to the training CLI.

    Returns:
        Reloaded workspace of the trained current node.
    """
    experiment_dir.mkdir(parents=True, exist_ok=True)
    config_path = write_run_config(config, experiment_dir / "_test_run_config.json")
    command = build_create_and_train_command(
        experiment_dir=experiment_dir,
        config_path=config_path,
        python_executable=sys.executable,
        label=label,
    )
    run_command_with_jax_training_env(command)
    graph = ExperimentGraph.from_directory(experiment_dir, config_cls=RLRunConfig)
    return graph.current_node


def materialize_case_in_training_subprocess(
    case: ExperimentCase[RLRunConfig],
    destination: str | Path,
) -> ExperimentCaseContext[RLRunConfig]:
    """Materialize an experiment case in a GPU-capable child process.

    Args:
        case: Case whose ``spec.id`` is passed to the cases CLI.
        destination: New or empty experiment directory.

    Returns:
        Context with the materialized graph and a ``root`` alias for the
        current node (the alias used by trained rollout cases).
    """
    destination_path = Path(destination)
    command = [
        sys.executable,
        "-m",
        "jarl.experiments.cases",
        "materialize",
        case.spec.id,
        "--destination",
        str(destination_path),
        "--json",
    ]
    completed = run_command_with_jax_training_env(command)
    context = ExperimentCaseContext(
        experiment_dir=destination_path.resolve(),
        config_cls=case.config_cls,
    )
    graph = context.reload_graph()
    aliases = _aliases_from_materialize_stdout(completed.stdout)
    if not aliases:
        aliases = {"root": graph.current_node.id}
    for alias, node_id in aliases.items():
        context.register_alias(alias, node_id)
    return context


def _aliases_from_materialize_stdout(stdout: str) -> dict[str, str]:
    """Parse ``root`` aliases from cases CLI ``--json`` output."""
    text = stdout.strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.rfind("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}
    result = payload.get("result")
    if not isinstance(result, dict):
        return {}
    aliases = result.get("aliases")
    if not isinstance(aliases, dict):
        return {}
    return {str(alias): str(node_id) for alias, node_id in aliases.items()}


@contextlib.contextmanager
def _gpu_lock(env: Mapping[str, str]) -> Iterator[None]:
    """Serialize CUDA children; CPU runs do not take the lock."""
    platforms = env.get("JAX_PLATFORMS", "")
    if "cuda" not in {part.strip() for part in platforms.split(",")}:
        yield
        return
    with _exclusive_lock(_GPU_LOCK_PATH):
        yield


@contextlib.contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive advisory lock on ``path``."""
    path.touch(exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
