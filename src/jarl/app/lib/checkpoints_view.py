"""Checkpoint listing helpers for the Runner Lab UI."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from jarl.experiments.io.checkpoints import (
    CHECKPOINT_ALIAS_BEST,
    CHECKPOINT_ALIAS_FINAL,
    CHECKPOINT_ALIAS_LATEST,
    CheckpointRecord,
    CheckpointRef,
    CheckpointStatus,
)
from jarl.experiments.node import NodeWorkspace


@dataclass(frozen=True, slots=True)
class CheckpointRow:
    """One checkpoint row for UI tables."""

    checkpoint_step: int
    node_step: int
    status: str
    global_step: int | None
    train_return: float | None
    eval_return: float | None
    is_latest_alias: bool
    is_final_alias: bool
    is_best_alias: bool


def list_checkpoint_rows(workspace: NodeWorkspace) -> list[CheckpointRow]:
    """Return checkpoint records formatted for display."""
    latest = workspace.resolve_checkpoint_alias(CHECKPOINT_ALIAS_LATEST)
    final = workspace.resolve_checkpoint_alias(CHECKPOINT_ALIAS_FINAL)
    best = workspace.resolve_checkpoint_alias(CHECKPOINT_ALIAS_BEST)
    rows: list[CheckpointRow] = []
    for record in sorted(workspace.list_checkpoints(), key=lambda item: item.checkpoint_step):
        if record.status != CheckpointStatus.SAVED:
            continue
        rows.append(_to_row(record, latest=latest, final=final, best=best))
    return rows


def checkpoint_table(rows: list[CheckpointRow]) -> pd.DataFrame:
    """Convert checkpoint rows to a dataframe."""
    return pd.DataFrame(
        [
            {
                "step": row.checkpoint_step,
                "node_step": row.node_step,
                "global_step": row.global_step,
                "train_return": row.train_return,
                "eval_return": row.eval_return,
                "aliases": _alias_label(row),
            }
            for row in rows
        ]
    )


def checkpoint_ref(workspace: NodeWorkspace, checkpoint_step: int) -> CheckpointRef:
    """Build a ``CheckpointRef`` for fork/extend from a parent node."""
    record = _saved_checkpoint_record(workspace, checkpoint_step)
    checkpoint_path = workspace.path / record.relative_path
    if not checkpoint_path.exists():
        msg = f"Checkpoint step {checkpoint_step} is registered but missing on disk: {checkpoint_path}"
        raise ValueError(msg)
    return CheckpointRef(node_id=workspace.id, checkpoint_step=checkpoint_step)


def available_checkpoint_aliases(workspace: NodeWorkspace) -> list[str]:
    """Return checkpoint aliases that resolve for ``workspace``."""
    return [
        alias
        for alias in (CHECKPOINT_ALIAS_LATEST, CHECKPOINT_ALIAS_BEST, CHECKPOINT_ALIAS_FINAL)
        if workspace.resolve_checkpoint_alias(alias) is not None
    ]


def _saved_checkpoint_record(workspace: NodeWorkspace, checkpoint_step: int) -> CheckpointRecord:
    """Return a saved checkpoint record without restoring Orbax payloads."""
    for record in workspace.list_checkpoints():
        if record.checkpoint_step == checkpoint_step and record.status == CheckpointStatus.SAVED:
            return record
    msg = f"Checkpoint step {checkpoint_step} is not registered as saved for node {workspace.id!r}."
    raise ValueError(msg)


def format_checkpoint_option(row: CheckpointRow) -> str:
    """Format a checkpoint row for selectbox labels."""
    alias = _alias_label(row)
    metrics = []
    if row.train_return is not None:
        metrics.append(f"train={row.train_return:.3g}")
    if row.eval_return is not None:
        metrics.append(f"eval={row.eval_return:.3g}")
    metric_text = f" ({', '.join(metrics)})" if metrics else ""
    alias_text = f" [{alias}]" if alias else ""
    return f"step {row.checkpoint_step}{alias_text}{metric_text}"


def _to_row(
    record: CheckpointRecord,
    *,
    latest: CheckpointRecord | None,
    final: CheckpointRecord | None,
    best: CheckpointRecord | None,
) -> CheckpointRow:
    metrics = record.metrics
    return CheckpointRow(
        checkpoint_step=record.checkpoint_step,
        node_step=record.node_step,
        status=record.status.value,
        global_step=record.global_step,
        train_return=_optional_float(metrics.get("rollout/episode_return")),
        eval_return=_optional_float(metrics.get("eval/episode_return")),
        is_latest_alias=latest is not None and latest.checkpoint_step == record.checkpoint_step,
        is_final_alias=final is not None and final.checkpoint_step == record.checkpoint_step,
        is_best_alias=best is not None and best.checkpoint_step == record.checkpoint_step,
    )


def _alias_label(row: CheckpointRow) -> str:
    labels: list[str] = []
    if row.is_latest_alias:
        labels.append("latest")
    if row.is_final_alias:
        labels.append("final")
    if row.is_best_alias:
        labels.append("best")
    return ", ".join(labels)


def _optional_float(value: float | None) -> float | None:
    return float(value) if value is not None else None
