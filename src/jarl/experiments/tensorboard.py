"""TensorBoard integration helpers for experiment DAGs.

These helpers build ``--logdir_spec`` strings from on-disk TensorBoard event
directories. Scalar metrics for analysis and plotting are stored primarily in
``metrics.jsonl`` per node; enable ``TrackingConfig.track_tensorboard`` during
training to populate the event files referenced here.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import overload

from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.node import NodeWorkspace

__all__ = [
    "build_logdir_spec",
    "launch_tensorboard",
    "tensorboard_compare",
    "tensorboard_lineage",
]

_STARTUP_TIMEOUT_S = 2.0


def build_logdir_spec(entries: dict[str, Path]) -> str:
    """Build a TensorBoard `--logdir_spec` string from labeled directories.

    Args:
        entries: Mapping from run label to TensorBoard log directory.

    Returns:
        Comma-separated `label:path` pairs for `--logdir_spec`.
    """
    if not entries:
        msg = "At least one logdir entry is required."
        raise ValueError(msg)
    parts = []
    for label, path in entries.items():
        safe_label = label.replace(":", "_").replace(",", "_")
        parts.append(f"{safe_label}:{Path(path).resolve().as_posix()}")
    return ",".join(parts)


def _tensorboard_entries(workspaces: Sequence[NodeWorkspace]) -> dict[str, Path]:
    return {ws.id: ws.tensorboard_dir for ws in workspaces}


@overload
def tensorboard_lineage(graph: ExperimentGraph, node: str) -> str: ...
@overload
def tensorboard_lineage(graph: ExperimentGraph, node: NodeWorkspace) -> str: ...


def tensorboard_lineage(graph: ExperimentGraph, node: str | NodeWorkspace) -> str:
    """Build a `--logdir_spec` covering the full lineage of a node.

    Each ancestor appears as a separate labeled run so TensorBoard can overlay
    the winner-horse story from root to the target node.

    Args:
        graph: Experiment graph.
        node: Target node ID or workspace.

    Returns:
        TensorBoard `--logdir_spec` string.
    """
    lineage = graph.get_lineage(node)
    entries = {f"{index}_{ws.branch}": ws.tensorboard_dir for index, ws in enumerate(lineage)}
    return build_logdir_spec(entries)


def _resolve_targets(
    graph: ExperimentGraph,
    nodes_or_branches: Sequence[str | NodeWorkspace],
) -> list[NodeWorkspace]:
    """Resolve node IDs, workspaces, or branch names to workspaces."""
    branches = graph.get_branches()
    resolved: list[NodeWorkspace] = []
    for item in nodes_or_branches:
        if isinstance(item, NodeWorkspace):
            resolved.append(item)
            continue
        if item in graph.all_nodes:
            resolved.append(graph.get_node(item))
            continue
        if item in branches:
            resolved.append(branches[item])
            continue
        msg = f"Unknown node or branch: {item}"
        raise KeyError(msg)
    return resolved


def tensorboard_compare(
    graph: ExperimentGraph,
    nodes_or_branches: Sequence[str | NodeWorkspace],
) -> str:
    """Build a `--logdir_spec` overlaying multiple nodes or branch heads.

    Args:
        graph: Experiment graph.
        nodes_or_branches: Node IDs, workspaces, or branch names.

    Returns:
        TensorBoard `--logdir_spec` string.

    Raises:
        KeyError: If any name does not resolve to a node or branch.
    """
    workspaces = _resolve_targets(graph, nodes_or_branches)
    return build_logdir_spec(_tensorboard_entries(workspaces))


def launch_tensorboard(
    logdir_spec: str,
    *,
    port: int = 6006,
    host: str = "localhost",
    extra_args: Sequence[str] | None = None,
) -> subprocess.Popen[str]:
    """Launch TensorBoard as a background subprocess.

    Args:
        logdir_spec: Value for TensorBoard `--logdir_spec`.
        port: Port to bind.
        host: Host to bind.
        extra_args: Additional TensorBoard CLI arguments.

    Returns:
        Running `Popen` handle for the TensorBoard process.

    Raises:
        FileNotFoundError: If the `tensorboard` executable is not on `PATH`.
    """
    executable = shutil.which("tensorboard")
    if executable is None:
        msg = "tensorboard executable not found on PATH."
        raise FileNotFoundError(msg)

    cmd = [
        executable,
        f"--logdir_spec={logdir_spec}",
        f"--port={port}",
        f"--host={host}",
        *(extra_args or ()),
    ]
    proc = subprocess.Popen(  # noqa: S603
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    deadline = time.monotonic() + _STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stderr = proc.stderr.read() if proc.stderr is not None else ""
            msg = f"TensorBoard exited immediately (code={proc.returncode})."
            if stderr.strip():
                msg = f"{msg}\n{stderr.strip()}"
            raise RuntimeError(msg)
        time.sleep(0.05)
    return proc
