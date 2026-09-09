"""Extend-chain metric concatenation across node lineages."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from jarl.experiments.metrics.snapshot import MetricsSnapshot

if TYPE_CHECKING:
    from jarl.experiments.graph import ExperimentGraph
    from jarl.experiments.node import NodeWorkspace

__all__ = [
    "concatenate_metric_snapshots",
    "describe_lineage_concat",
    "extend_chain_nodes",
]


def extend_chain_nodes(
    graph: ExperimentGraph[object],
    node_id: str,
) -> tuple[NodeWorkspace, ...]:
    """Return root-to-target nodes on the same branch (extend chain, excludes fork prefixes)."""
    lineage = graph.get_lineage(node_id)
    if not lineage:
        return ()
    target_branch = lineage[-1].branch
    chain: list[NodeWorkspace] = []
    for workspace in reversed(lineage):
        if workspace.branch != target_branch:
            break
        chain.append(workspace)
    chain.reverse()
    return tuple(chain)


def concatenate_metric_snapshots(
    segments: Sequence[tuple[NodeWorkspace, MetricsSnapshot]],
) -> MetricsSnapshot:
    """Merge per-node metrics into one timeline using ``parent_checkpoint_step`` offsets."""
    by_step: dict[int, dict[str, float]] = {}
    names: set[str] = set()
    record_count = sum(snapshot.record_count for _, snapshot in segments)
    best: dict[str, tuple[int, float]] = {}

    for index, (_workspace, snapshot) in enumerate(segments):
        if snapshot.record_count == 0:
            continue
        offset = 0 if index == 0 else int(_workspace.parent_checkpoint_step or 0)
        for step, metrics in snapshot.by_step.items():
            abs_step = int(step) + offset
            bucket = by_step.setdefault(abs_step, {})
            for name, value in metrics.items():
                bucket[name] = float(value)
                names.add(name)
                current = best.get(name)
                if current is None or abs_step >= current[0]:
                    best[name] = (abs_step, float(value))

    latest = {name: value for name, (_step, value) in best.items()}
    return MetricsSnapshot(
        latest=latest,
        names=tuple(sorted(names)),
        record_count=record_count,
        by_step=by_step,
    )


def describe_lineage_concat(
    segments: Sequence[tuple[NodeWorkspace, MetricsSnapshot]],
) -> list[str]:
    """Return human-readable offset notes for each segment in a concat chain."""
    lines: list[str] = []
    for index, (workspace, snapshot) in enumerate(segments):
        if snapshot.record_count == 0:
            lines.append(f"{workspace.id} (sin métricas)")
            continue
        if index == 0:
            lines.append(f"{workspace.id} (base)")
            continue
        offset = int(workspace.parent_checkpoint_step or 0)
        lines.append(f"{workspace.id} (+{offset:,} steps)")
    return lines
