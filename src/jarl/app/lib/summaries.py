"""Pandas presentation helpers for experiment summaries in Runner Lab."""

from __future__ import annotations

import pandas as pd

from jarl.app.lib.metrics_view import snapshot_to_wide_frame
from jarl.experiments.metrics import build_metrics_snapshot
from jarl.experiments.node import NodeWorkspace
from jarl.experiments.summaries import ExperimentSummary

__all__ = [
    "metrics_overview",
    "node_table",
]


def node_table(summary: ExperimentSummary) -> pd.DataFrame:
    """Return node summaries as a dataframe."""
    rows = [
        {
            "node": node.node_id,
            "status": node.status,
            "branch": node.branch,
            "label": node.label,
            "parent": node.parent_id or "",
            "step": node.step,
            "train_return": node.latest_return,
            "eval_return": node.latest_eval_return,
            "metrics": node.metrics_count,
            "checkpoints": node.checkpoint_count,
            "artifacts": node.artifact_count,
            "updated": node.updated_at,
        }
        for node in summary.nodes
    ]
    return pd.DataFrame(rows)


def metrics_overview(workspace: NodeWorkspace) -> tuple[pd.DataFrame, list[str]]:
    """Return wide metrics dataframe and available metric names."""
    snapshot = build_metrics_snapshot(workspace.metrics_jsonl_path)
    return snapshot_to_wide_frame(snapshot), list(snapshot.names)
