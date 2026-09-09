"""Pandas presentation helpers for inference results in Runner Lab."""

from __future__ import annotations

import pandas as pd

from jarl.agents.ppo.inference.requests import video_metrics_for_prefix as _video_metrics_for_prefix
from jarl.experiments.node import NodeWorkspace

__all__ = [
    "inference_timing_metrics",
    "video_metrics_for_prefix",
]


def video_metrics_for_prefix(workspace: NodeWorkspace, name_prefix: str) -> pd.DataFrame:
    """Return ``video_metrics.jsonl`` rows whose ``video_file`` matches ``name_prefix``."""
    rows = _video_metrics_for_prefix(workspace, name_prefix)
    return pd.DataFrame(rows)


def inference_timing_metrics(metrics_df: pd.DataFrame) -> dict[str, float]:
    """Aggregate rollout/encode/transfer seconds from one inference job."""
    if metrics_df.empty:
        return {}

    timing: dict[str, float] = {}
    if "rollout_seconds" in metrics_df.columns:
        timing["rollout_s"] = float(metrics_df["rollout_seconds"].iloc[0])
    if "transfer_seconds" in metrics_df.columns:
        timing["transfer_s"] = float(metrics_df["transfer_seconds"].iloc[0])
    if "encode_seconds" in metrics_df.columns:
        timing["encode_s"] = float(metrics_df["encode_seconds"].fillna(0).sum())
    total = sum(timing.values())
    if total > 0:
        timing["total_s"] = total
    return timing
