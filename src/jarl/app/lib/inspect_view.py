"""Inspection helpers for experiment nodes and configs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from jarl.config import ConfigDiff
from jarl.envs.navix.reward_config import RewardWeightsConfig
from jarl.envs.navix.scenario_rewards import ScenarioRewardError, resolve_scenario_reward_spec
from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.node import NodeWorkspace
from jarl.operations.graph.reward import RewardRequest, reward
from jarl.training.config import RLRunConfig


def load_run_metadata(exp_dir: Path) -> dict[str, Any] | None:
    """Load ``run_metadata.json`` when present."""
    path = exp_dir / "run_metadata.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_metadata_table(exp_dir: Path) -> pd.DataFrame:
    """Return operator-facing run metadata rows."""
    metadata = load_run_metadata(exp_dir)
    if metadata is None:
        return pd.DataFrame(columns=["field", "value"])
    rows = [
        {"field": "start_time", "value": metadata.get("start_time", "")},
        {"field": "transfer_group", "value": metadata.get("transfer_group", "")},
        {"field": "git_commit", "value": _nested(metadata, "git", "commit")},
        {"field": "git_branch", "value": _nested(metadata, "git", "branch")},
        {"field": "python", "value": _nested(metadata, "env", "python_version")},
        {"field": "platform", "value": _nested(metadata, "env", "platform")},
    ]
    return pd.DataFrame(rows)


def branch_heads_table(graph: ExperimentGraph[RLRunConfig]) -> pd.DataFrame:
    """List branches with their head node and status."""
    rows = []
    for branch, head in graph.branch_heads.items():
        rows.append(
            {
                "branch": branch,
                "head": head.id,
                "status": head.status.value,
                "label": head.node_metadata.label,
                "step": head.node_metadata.step,
            }
        )
    return pd.DataFrame(rows)


def config_diff_dataframe(diff: ConfigDiff) -> pd.DataFrame:
    """Render a ``ConfigDiff`` as a flat table."""
    rows: list[dict[str, str]] = [
        {"path": path, "change": "added", "value": _format_diff_value(diff.added[path])} for path in sorted(diff.added)
    ]
    rows.extend(
        {
            "path": path,
            "change": "changed",
            "value": f"{_format_diff_value(old)} → {_format_diff_value(new)}",
        }
        for path, (old, new) in sorted(diff.changed.items())
    )
    rows.extend(
        {"path": path, "change": "removed", "value": _format_diff_value(diff.removed[path])}
        for path in sorted(diff.removed)
    )
    return pd.DataFrame(rows)


def execution_attempts_table(workspace: NodeWorkspace) -> pd.DataFrame:
    """List execution attempts for resume debugging."""
    rows = [
        {
            "attempt": record.attempt_id,
            "status": record.status.value,
            "started_at": record.started_at,
            "ended_at": record.ended_at or "",
            "error": record.error_message or "",
        }
        for record in workspace.list_execution_attempts()
    ]
    return pd.DataFrame(rows)


def config_overrides_table(workspace: NodeWorkspace) -> pd.DataFrame:
    """Show sparse overrides saved for a node."""
    overrides = workspace.load_config_overrides()
    if not overrides:
        return pd.DataFrame(columns=["path", "value"])
    return pd.DataFrame(
        [{"path": path, "value": _format_diff_value(value)} for path, value in sorted(overrides.items())]
    )


def reward_mix_table(graph: ExperimentGraph[RLRunConfig], workspace: NodeWorkspace) -> pd.DataFrame:
    """Return the resolved configurable reward mix for ``workspace``."""
    mix = reward(graph, RewardRequest(node_id=workspace.id)).reward
    if mix is None:
        return pd.DataFrame(columns=["channel", "weight", "description"])
    fields = RewardWeightsConfig.model_fields
    rows = [
        {
            "channel": name,
            "weight": value,
            "description": str(fields[name].description or "") if name in fields else "",
        }
        for name, value in mix.items()
    ]
    frame = pd.DataFrame(rows)
    return frame.sort_values(["weight", "channel"], ascending=[False, True], ignore_index=True)


def scenario_overlay_table(config: RLRunConfig) -> pd.DataFrame:
    """Return the scenario overlay configured on ``config``, if any."""
    environment = config.environment
    try:
        spec = resolve_scenario_reward_spec(
            environment.env_id,
            environment.scenario_reward_id,
            environment.scenario_reward_version,
        )
    except ScenarioRewardError as exc:
        return pd.DataFrame(
            [
                {
                    "id": environment.scenario_reward_id or "",
                    "version": environment.scenario_reward_version,
                    "scale": None,
                    "position": "",
                    "error": str(exc),
                }
            ]
        )
    if spec is None:
        return pd.DataFrame(columns=["id", "version", "scale", "position"])
    return pd.DataFrame(
        [
            {
                "id": spec.id,
                "version": spec.version,
                "scale": spec.scale,
                "position": f"{spec.position[0]}, {spec.position[1]}",
            }
        ]
    )


def artifacts_table(workspace: NodeWorkspace) -> pd.DataFrame:
    """List model archives and artifact aliases."""
    rows = [
        {
            "name": record.name,
            "kind": record.kind,
            "path": record.relative_path,
            "step": record.step,
        }
        for record in workspace.list_artifacts()
    ]
    return pd.DataFrame(rows)


def lineage_metrics_dataframe(graph: ExperimentGraph[RLRunConfig], node: str | NodeWorkspace) -> pd.DataFrame:
    """Return metrics collected along the lineage of a node."""
    metrics = graph.get_metrics_along_lineage(node)
    if not metrics:
        return pd.DataFrame()
    return pd.DataFrame(metrics)


def video_metrics_table(workspace: NodeWorkspace) -> pd.DataFrame:
    """Load ``video_metrics.jsonl`` rows when available."""
    path = workspace.video_metrics_jsonl_path
    if not path.is_file():
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def video_metrics_by_role(workspace: NodeWorkspace) -> dict[str, pd.DataFrame]:
    """Group video metric rows by showcase ``role`` when present."""
    table = video_metrics_table(workspace)
    if table.empty or "role" not in table.columns:
        return {}
    grouped: dict[str, pd.DataFrame] = {}
    for role, frame in table.groupby("role", dropna=False):
        key = str(role) if role is not None else "unknown"
        grouped[key] = frame.reset_index(drop=True)
    return grouped


def list_video_files(workspace: NodeWorkspace) -> list[Path]:
    """Return MP4 files under the node videos directory."""
    if not workspace.videos_dir.is_dir():
        return []
    return sorted(workspace.videos_dir.glob("*.mp4"))


def can_extend_from_head(graph: ExperimentGraph[RLRunConfig], node: str | NodeWorkspace) -> bool:
    """Return whether ``node`` is the head of its branch."""
    return graph.is_branch_head(node)


def _format_diff_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def _nested(payload: dict[str, Any], *keys: str) -> str:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return ""
        current = current.get(key, "")
    return str(current) if current is not None else ""
