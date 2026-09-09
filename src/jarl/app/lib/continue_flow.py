"""Continuation wizard helpers for the Runner Lab."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from jarl.app.lib.graph_ops import (
    checkout_node,
    create_prepared_root,
    extend_branch,
    fork_branch,
)
from jarl.experiments.io.checkpoints import CheckpointRef
from jarl.experiments.node import NodeWorkspace
from jarl.training.config import RLRunConfig
from jarl.training.presets import RunFormPayload, child_config_overrides_from_payload

ContinuationKind = Literal["extend", "fork"]


class CheckpointPick(StrEnum):
    """How the operator selects a parent checkpoint."""

    NONE = "none"
    LATEST = "latest"
    BEST = "best"
    FINAL = "final"
    STEP = "step"


@dataclass(frozen=True, slots=True)
class ContinuationRequest:
    """Parameters for creating a child node from the wizard."""

    kind: ContinuationKind
    from_node: str
    branch: str
    label: str
    prepare: bool
    checkpoint_pick: CheckpointPick
    checkpoint_step: int | None = None
    config_overrides: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ContinuationResult:
    """Outcome of a continuation operation."""

    child_id: str
    parent_checkpoint_step: int | None
    checkpoint_ref: CheckpointRef | None


def default_extend_label(parent: NodeWorkspace) -> str:
    """Suggest a label when extending on the same branch."""
    return f"extend_{parent.node_metadata.step + 1}"


def default_fork_branch(parent: NodeWorkspace) -> str:
    """Suggest a new branch name for a fork."""
    return f"{parent.branch}_fork"


def run_continuation(
    exp_dir: Path,
    *,
    request: ContinuationRequest,
    checkpoint_ref: CheckpointRef | None,
) -> ContinuationResult:
    """Create an extend or fork child according to the wizard request."""
    overrides = request.config_overrides or {}
    if request.kind == "extend":
        child_id = extend_branch(
            exp_dir,
            from_node=request.from_node,
            branch=request.branch or None,
            label=request.label,
            config_overrides=overrides,
            from_checkpoint=checkpoint_ref,
            prepare=request.prepare,
        )
    else:
        child_id = fork_branch(
            exp_dir,
            from_node=request.from_node,
            branch=request.branch,
            label=request.label,
            config_overrides=overrides,
            from_checkpoint=checkpoint_ref,
            prepare=request.prepare,
        )
    parent_step = checkpoint_ref.checkpoint_step if checkpoint_ref is not None else None
    return ContinuationResult(
        child_id=child_id,
        parent_checkpoint_step=parent_step,
        checkpoint_ref=checkpoint_ref,
    )


def create_root_only(
    *,
    experiment_dir: Path,
    config: RLRunConfig,
    label: str,
    branch: str = "main",
) -> str:
    """Create a prepared root node without launching training."""
    return create_prepared_root(experiment_dir, config=config, label=label, branch=branch)


def overrides_from_payload(parent_config: RLRunConfig, payload: RunFormPayload) -> dict[str, Any]:
    """Build sparse overrides for fork/extend from the shared form."""
    return child_config_overrides_from_payload(parent_config, payload)


def activate_created_node(exp_dir: Path, node_id: str) -> None:
    """Checkout the newly created node as the active context."""
    checkout_node(exp_dir, node_id)
