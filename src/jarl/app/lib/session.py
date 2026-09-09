"""Streamlit session helpers for the Runner Lab."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.node import NodeWorkspace
from jarl.experiments.paths import resolve_workspace_root
from jarl.training.config import RLRunConfig

TRAINING_RUN_MODULE = "jarl.training.run"
INFERENCE_MODULE = "jarl.agents.ppo.inference"

NODE_PICKER_WIDGET_KEYS: tuple[str, ...] = (
    "entrenar_node",
    "arbol_action_node",
    "sidebar_checkout_node",
    "continuar_source_node",
)
PENDING_NODE_PICKER_SYNC_KEY = "jarl_pending_node_picker_sync"


@dataclass(slots=True)
class ActiveRun:
    """Subprocess handle for a background training job."""

    process: subprocess.Popen[str]
    log_path: Path


@dataclass(slots=True)
class ActiveInferenceRun:
    """Subprocess handle for a background inference video job."""

    process: subprocess.Popen[str]
    log_path: Path
    node_id: str
    name_prefix: str
    checkpoint_step: int


@dataclass(slots=True)
class ActiveAgenticCaseRun:
    """Subprocess handle for an agentic case evaluation."""

    process: subprocess.Popen[str]
    log_path: Path
    case_id: str
    experiment_dir: Path


def ensure_web_path() -> None:
    """No-op kept for page compatibility (package imports replace ``sys.path`` hacks)."""


def init_session_state(defaults: dict[str, Any] | None = None) -> None:
    """Initialize Streamlit session keys when missing."""
    import streamlit as st

    for key, value in (defaults or {}).items():
        if key not in st.session_state:
            st.session_state[key] = value


def workspace_root() -> Path:
    """Return the experiment parent directory (env default or sidebar override)."""
    import streamlit as st

    override = st.session_state.get("experiments_root_override")
    root = resolve_workspace_root(override=override)
    root.mkdir(parents=True, exist_ok=True)
    return root


def set_workspace_root(path: Path) -> None:
    """Persist a sidebar override for the experiment parent directory."""
    import streamlit as st

    st.session_state["experiments_root_override"] = str(path.resolve())


def experiment_dir() -> Path | None:
    """Return the active experiment directory, if any."""
    import streamlit as st

    raw = st.session_state.get("experiment_dir")
    return Path(raw) if raw else None


def set_experiment_dir(path: Path | None) -> None:
    """Set or clear the active experiment directory."""
    import streamlit as st

    st.session_state["experiment_dir"] = str(path.resolve()) if path is not None else None


def list_experiments(root: Path | None = None) -> list[Path]:
    """List experiment directories that contain a manifest under ``root``.

    If ``root`` itself is an experiment (it contains ``experiment.json``), return
    that directory so a pasted experiment path still appears in the picker.
    """
    base = root or workspace_root()
    if not base.is_dir():
        return []
    if (base / "experiment.json").is_file():
        return [base]
    return [child for child in sorted(base.iterdir()) if child.is_dir() and (child / "experiment.json").is_file()]


def load_graph(exp_dir: Path) -> ExperimentGraph[RLRunConfig]:
    """Load an experiment graph from disk."""
    return ExperimentGraph.from_directory(exp_dir, config_cls=RLRunConfig)


def current_workspace(exp_dir: Path) -> NodeWorkspace:
    """Return the current node workspace for an experiment."""
    return load_graph(exp_dir).current_node


def list_node_summaries(
    exp_dir: Path,
    graph: ExperimentGraph[RLRunConfig] | None = None,
) -> list[tuple[str, str]]:
    """Return ``(node_id, status)`` pairs for all nodes in an experiment."""
    resolved = graph if graph is not None else load_graph(exp_dir)
    network = resolved.as_networkx()
    return sorted((node_id, resolved.get_node(node_id).status.value) for node_id in network.nodes)


def active_run() -> ActiveRun | None:
    """Return the active subprocess run, if any."""
    import streamlit as st

    return st.session_state.get("active_run")


def set_active_run(run: ActiveRun | None) -> None:
    """Store or clear the active subprocess run."""
    import streamlit as st

    st.session_state["active_run"] = run


def poll_active_run() -> int | None:
    """Refresh subprocess state and return the exit code when finished."""
    run = active_run()
    if run is None:
        return None
    code = run.process.poll()
    if code is not None:
        set_active_run(None)
        bump_graph_revision()
    return code


def active_inference_run() -> ActiveInferenceRun | None:
    """Return the active inference subprocess, if any."""
    import streamlit as st

    return st.session_state.get("active_inference_run")


def set_active_inference_run(run: ActiveInferenceRun | None) -> None:
    """Store or clear the active inference subprocess."""
    import streamlit as st

    st.session_state["active_inference_run"] = run


def poll_active_inference_run() -> int | None:
    """Refresh inference subprocess state and return the exit code when finished."""
    run = active_inference_run()
    if run is None:
        return None
    code = run.process.poll()
    if code is not None:
        set_active_inference_run(None)
        bump_graph_revision()
    return code


def active_agentic_case_run() -> ActiveAgenticCaseRun | None:
    """Return the active agentic case subprocess, if any."""
    import streamlit as st

    return st.session_state.get("active_agentic_case_run")


def set_active_agentic_case_run(run: ActiveAgenticCaseRun | None) -> None:
    """Store or clear the active agentic case subprocess."""
    import streamlit as st

    st.session_state["active_agentic_case_run"] = run


def poll_active_agentic_case_run() -> int | None:
    """Refresh agentic case subprocess state and return its exit code."""
    run = active_agentic_case_run()
    if run is None:
        return None
    code = run.process.poll()
    if code is not None:
        set_active_agentic_case_run(None)
    return code


def graph_revision() -> int:
    """Return a monotonic counter used to invalidate cached graph renders."""
    import streamlit as st

    return int(st.session_state.get("graph_revision", 0))


def bump_graph_revision() -> None:
    """Increment the graph revision counter after structural graph changes."""
    import streamlit as st

    st.session_state["graph_revision"] = graph_revision() + 1


def _apply_node_picker_sync(node_id: str) -> None:
    """Write a node id into all node-picker widget session keys."""
    import streamlit as st

    for key in NODE_PICKER_WIDGET_KEYS:
        st.session_state[key] = node_id


def request_node_picker_sync(node_id: str) -> None:
    """Schedule node-picker alignment for the next script run."""
    import streamlit as st

    st.session_state[PENDING_NODE_PICKER_SYNC_KEY] = node_id


def apply_pending_node_picker_sync() -> str | None:
    """Apply a deferred node-picker sync before widgets are instantiated.

    Returns:
        The synced node id, or ``None`` when no sync was pending.
    """
    import streamlit as st

    pending = st.session_state.pop(PENDING_NODE_PICKER_SYNC_KEY, None)
    if pending is None:
        return None
    node_id = str(pending)
    _apply_node_picker_sync(node_id)
    return node_id


def sync_node_picker_widgets(node_id: str) -> None:
    """Align Streamlit node pickers on the next run (safe after widgets exist)."""
    request_node_picker_sync(node_id)


def checkout_experiment_node(exp_dir: Path, node_id: str) -> None:
    """Checkout a node on disk and sync Runner Lab widget state."""
    from jarl.app.lib.graph_ops import checkout_node

    checkout_node(exp_dir, node_id)
    bump_graph_revision()
    request_node_picker_sync(node_id)


def set_last_created_node(node_id: str | None) -> None:
    """Remember the most recently created node for quick training."""
    import streamlit as st

    st.session_state["last_created_node"] = node_id


def last_created_node() -> str | None:
    """Return the most recently created node id, if any."""
    import streamlit as st

    value = st.session_state.get("last_created_node")
    return str(value) if value else None


def clear_last_created_node() -> None:
    """Clear the remembered created node."""
    set_last_created_node(None)


def tail_file(path: Path, *, max_lines: int = 80) -> str:
    """Return the last ``max_lines`` from a text file."""
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def env_with_wandb_secrets() -> dict[str, str]:
    """Build subprocess env, forwarding W&B secrets when configured."""
    import streamlit as st
    from streamlit.errors import StreamlitSecretNotFoundError

    env = os.environ.copy()
    try:
        secrets = st.secrets
        for key in ("WANDB_API_KEY", "WANDB_MODE", "WANDB_ENTITY"):
            if key in secrets:
                env[key] = str(secrets[key])
    except StreamlitSecretNotFoundError:
        pass
    return env
