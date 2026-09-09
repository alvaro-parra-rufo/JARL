"""Tests for the diverging best/latest Navix root experiment case."""

from __future__ import annotations

from pathlib import Path

from jarl.experiments.cases.builtin.navix_root_best_vs_latest import (
    BEST_CHECKPOINT_STEP,
    CASE,
    LATEST_CHECKPOINT_STEP,
)
from jarl.experiments.io.checkpoints import CHECKPOINT_ALIAS_BEST, CHECKPOINT_ALIAS_LATEST


def test_materialize_diverging_best_and_latest(tmp_path: Path) -> None:
    context = CASE.materialize(tmp_path / "case")
    graph = context.reload_graph()
    workspace = graph.get_node(context.node_id("root"))

    best = workspace.resolve_checkpoint_alias(CHECKPOINT_ALIAS_BEST)
    latest = workspace.resolve_checkpoint_alias(CHECKPOINT_ALIAS_LATEST)

    assert best is not None
    assert latest is not None
    assert best.checkpoint_step == BEST_CHECKPOINT_STEP
    assert latest.checkpoint_step == LATEST_CHECKPOINT_STEP
    assert best.checkpoint_step != latest.checkpoint_step
