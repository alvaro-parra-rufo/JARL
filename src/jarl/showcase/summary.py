"""Write and load ``showcase_summary.json`` for Runner Lab."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.tensorboard import tensorboard_compare, tensorboard_lineage
from jarl.training.config import RLRunConfig

SHOWCASE_SUMMARY_FILENAME = "showcase_summary.json"
SHOWCASE_RUN_FILENAME = "showcase_run.json"

__all__ = [
    "SHOWCASE_RUN_FILENAME",
    "SHOWCASE_SUMMARY_FILENAME",
    "load_showcase_run",
    "load_showcase_summary",
    "save_showcase_run",
    "write_showcase_summary",
]


def load_showcase_summary(exp_dir: Path) -> dict[str, Any] | None:
    """Load ``showcase_summary.json`` when present."""
    path = exp_dir / SHOWCASE_SUMMARY_FILENAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_showcase_run(exp_dir: Path) -> dict[str, Any] | None:
    """Load incremental ``showcase_run.json`` state."""
    path = exp_dir / SHOWCASE_RUN_FILENAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_showcase_run(exp_dir: Path, payload: dict[str, Any]) -> Path:
    """Persist incremental showcase run state."""
    path = exp_dir / SHOWCASE_RUN_FILENAME
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_showcase_summary(
    *,
    exp_dir: Path,
    plan_meta: dict[str, Any],
    run_state: dict[str, Any],
) -> Path:
    """Build ``showcase_summary.json`` from the experiment graph and run state."""
    graph: ExperimentGraph[RLRunConfig] = ExperimentGraph.from_directory(exp_dir, config_cls=RLRunConfig)
    nodes_payload: list[dict[str, Any]] = []
    for logical_name, node_state in run_state.get("nodes", {}).items():
        node_id = node_state.get("node_id")
        if not node_id:
            continue
        workspace = graph.get_node(node_id)
        nodes_payload.append(
            {
                "logical_name": logical_name,
                "node_id": node_id,
                "status": node_state.get("status"),
                "branch": workspace.branch,
                "label": workspace.node_metadata.label,
                "env_id": graph.resolve_config(workspace).environment.env_id,
                "parent_checkpoint_step": workspace.parent_checkpoint_step,
                "checkpoint_step": node_state.get("checkpoint_step"),
                "videos": node_state.get("videos", {}),
            }
        )

    head_node = graph.current_node.id
    summary = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "experiment_dir": str(exp_dir),
        "plan": plan_meta,
        "nodes": nodes_payload,
        "branch_heads": {branch: head.id for branch, head in graph.branch_heads.items()},
        "tensorboard_lineage": tensorboard_lineage(graph, head_node),
        "tensorboard_compare": tensorboard_compare(graph, [item["node_id"] for item in nodes_payload[:3]]),
        "streamlit_command": "uv run streamlit run src/jarl/app/app.py",
    }
    path = exp_dir / SHOWCASE_SUMMARY_FILENAME
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path
