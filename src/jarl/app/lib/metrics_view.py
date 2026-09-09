"""Metric helpers for the Runner Lab UI."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import pandas as pd

from jarl.agents.ppo.metric_schema import PPO_EVAL_METRIC_SCHEMA, PPO_TRAIN_METRIC_SCHEMA
from jarl.experiments.io.metrics import MetricRecord
from jarl.experiments.metrics import MetricsSnapshot

DEFAULT_TRAIN_METRICS = (
    "rollout/episode_return",
    "loss/policy_gradient_loss",
    "steps/nr_env_steps",
)
DEFAULT_EVAL_METRICS = PPO_EVAL_METRIC_SCHEMA.device_names
DEFAULT_MAX_CHART_ROWS = 2_000
DEFAULT_MAX_TABLE_ROWS = 1_000
DEFAULT_METRIC_PRESET = "Train"
DEFAULT_METRIC_PREFIXES: tuple[str, ...] = ("rollout", "eval", "loss", "steps", "time")

METRIC_PRESETS: dict[str, tuple[str, ...]] = {
    "Train": (
        "rollout/episode_return",
        "rollout/episode_length",
        "rollout/value_mean",
        "steps/nr_env_steps",
        "time/sps",
    ),
    "Eval": PPO_EVAL_METRIC_SCHEMA.device_names,
    "Loss": (
        "loss/policy_gradient_loss",
        "loss/critic_loss",
        "loss/entropy_loss",
    ),
    "Debug": (
        "policy_ratio/approx_kl",
        "policy_ratio/clip_fraction",
        "gradients/policy_grad_norm",
        "gradients/critic_grad_norm",
        "v_value/explained_variance",
    ),
}

DEFAULT_COMPARISON_METRICS = (
    "rollout/episode_return",
    "eval/episode_return",
    "loss/policy_gradient_loss",
    "time/sps",
)

__all__ = [
    "DEFAULT_COMPARISON_METRICS",
    "DEFAULT_EVAL_METRICS",
    "DEFAULT_MAX_CHART_ROWS",
    "DEFAULT_MAX_TABLE_ROWS",
    "DEFAULT_METRIC_PREFIXES",
    "DEFAULT_METRIC_PRESET",
    "DEFAULT_TRAIN_METRICS",
    "METRIC_PRESETS",
    "build_multi_node_frame",
    "build_multi_node_latest_table",
    "comparison_metric_options",
    "default_metric_choices",
    "default_metric_prefixes",
    "downsample_frame",
    "metrics_file_cache_key",
    "preset_metric_names",
    "records_to_wide_frame",
    "snapshot_to_wide_frame",
    "tail_frame",
]


def metrics_file_cache_key(metrics_path: Path) -> tuple[str, int, int]:
    """Return a cache key from path identity and file metadata."""
    resolved = metrics_path.resolve()
    if not resolved.is_file():
        return (str(resolved), 0, 0)
    stat = resolved.stat()
    return (str(resolved), stat.st_mtime_ns, stat.st_size)


def snapshot_to_wide_frame(
    snapshot: MetricsSnapshot,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Pivot a snapshot into a wide dataframe indexed by step."""
    if not snapshot.by_step:
        return pd.DataFrame()

    if columns is None:
        frame = pd.DataFrame.from_dict(snapshot.by_step, orient="index").sort_index()
        frame.index.name = "step"
        return frame

    column_set = set(columns)
    if not column_set:
        return pd.DataFrame()

    filtered = {
        step: {name: value for name, value in metrics.items() if name in column_set}
        for step, metrics in snapshot.by_step.items()
    }
    frame = pd.DataFrame.from_dict(filtered, orient="index").sort_index()
    frame.index.name = "step"
    return frame


def preset_metric_names(preset: str, available: set[str] | frozenset[str]) -> list[str]:
    """Return preset metric names that exist in ``available``."""
    return [name for name in METRIC_PRESETS.get(preset, ()) if name in available]


def comparison_metric_options(snapshots: Mapping[str, MetricsSnapshot]) -> list[str]:
    """Return sorted metric names available across ``snapshots``, preferred first."""
    available: set[str] = set()
    for snapshot in snapshots.values():
        available.update(snapshot.names)
    preferred = [*DEFAULT_COMPARISON_METRICS, *PPO_TRAIN_METRIC_SCHEMA.all_names, *PPO_EVAL_METRIC_SCHEMA.all_names]
    ordered = [name for name in preferred if name in available]
    ordered.extend(sorted(name for name in available if name not in ordered))
    return ordered


def build_multi_node_frame(
    snapshots: Mapping[str, MetricsSnapshot],
    metrics: Sequence[str],
) -> pd.DataFrame:
    """Build a wide chart frame with one column per node (or node·metric)."""
    if not snapshots or not metrics:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    metric_list = list(metrics)
    for node_id, snapshot in snapshots.items():
        wide = snapshot_to_wide_frame(snapshot, metric_list)
        if wide.empty:
            continue
        if len(metric_list) == 1:
            metric = metric_list[0]
            if metric in wide.columns:
                frames.append(wide[[metric]].rename(columns={metric: node_id}))
        else:
            frames.append(wide.rename(columns={name: f"{node_id} · {name}" for name in wide.columns}))

    if not frames:
        return pd.DataFrame()

    frame = frames[0]
    for extra in frames[1:]:
        frame = frame.join(extra, how="outer")
    frame.index.name = "step"
    return frame.sort_index()


def build_multi_node_latest_table(
    snapshots: Mapping[str, MetricsSnapshot],
    metrics: Sequence[str],
) -> pd.DataFrame:
    """Return latest scalar values per node for the selected metrics."""
    rows: list[dict[str, float | str | None]] = []
    for node_id, snapshot in snapshots.items():
        row: dict[str, float | str | None] = {"node": node_id}
        for metric in metrics:
            row[metric] = snapshot.latest.get(metric)
        rows.append(row)
    return pd.DataFrame(rows)


def records_to_wide_frame(records: list[MetricRecord]) -> pd.DataFrame:
    """Pivot metric records into a wide dataframe indexed by step."""
    if not records:
        return pd.DataFrame()
    by_step: dict[int, dict[str, float]] = {}
    for record in records:
        by_step.setdefault(record.step, {})[record.name] = record.value
    frame = pd.DataFrame.from_dict(by_step, orient="index").sort_index()
    frame.index.name = "step"
    return frame


def downsample_frame(frame: pd.DataFrame, *, max_rows: int = DEFAULT_MAX_CHART_ROWS) -> pd.DataFrame:
    """Return a row-limited frame that preserves first and last samples."""
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame
    stride = max(1, len(frame) // max_rows)
    sampled = frame.iloc[::stride]
    if sampled.empty or sampled.index[-1] == frame.index[-1]:
        return sampled
    return pd.concat([sampled, frame.tail(1)])


def tail_frame(frame: pd.DataFrame, *, max_rows: int = DEFAULT_MAX_TABLE_ROWS) -> pd.DataFrame:
    """Return the last ``max_rows`` rows for responsive tables."""
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame
    return frame.tail(max_rows)


def default_metric_prefixes(prefixes: Iterable[str]) -> list[str]:
    """Return a small default prefix subset for the metrics explorer."""
    available = list(dict.fromkeys(prefixes))
    preferred = [prefix for prefix in DEFAULT_METRIC_PREFIXES if prefix in available]
    return preferred or available[:3]


def default_metric_choices(
    names: Iterable[str] | Mapping[str, float],
) -> list[str]:
    """Pick a small default metric set present in ``names``."""
    available = set(names)
    preferred = [*DEFAULT_TRAIN_METRICS, *DEFAULT_EVAL_METRICS]
    selected: list[str] = []
    for name in preferred:
        if name in available and name not in selected:
            selected.append(name)
    return selected or sorted(available)[:3]
