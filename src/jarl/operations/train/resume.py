"""Resume training in-process through the global runner."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, ClassVar

from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.node import RESUMABLE_NODE_STATUSES
from jarl.operations.contracts.compact import dataclass_to_compact_dict
from jarl.operations.train._helpers import resolve_train_overrides
from jarl.operations.train.run import RunResponse, ScheduleSummary, build_run_response
from jarl.training.config import RLRunConfig
from jarl.training.launch import build_resume_command, spawn_training, write_run_config
from jarl.training.presets import RunFormPayload
from jarl.training.runner import resume_training
from jarl.training.schedule import apply_training_schedule, compute_training_schedule
from jarl.training.trainer import EnvFactory, Trainer

__all__ = ["ResumeRequest", "ResumeResponse", "resume"]


@dataclass(frozen=True, slots=True)
class ResumeRequest:
    """Inputs for resuming training on a node."""

    node_id: str | None = None
    payload: RunFormPayload | None = None
    config_overrides: dict[str, Any] | None = None
    detach: bool = False


@dataclass(frozen=True, slots=True)
class ResumeResponse:
    """Outcome of ``resume``."""

    node_id: str
    status: str
    schedule: ScheduleSummary
    checkpoint_step: int | None = None
    detached: bool | None = None
    pid: int | None = None
    log_path: str | None = None

    _compact_include: ClassVar[frozenset[str] | None] = None

    def to_compact_dict(self) -> dict[str, object]:
        """Return a compact projection for tools and agents."""
        return dataclass_to_compact_dict(self, include=self._compact_include)


def resume(
    graph: ExperimentGraph[RLRunConfig],
    request: ResumeRequest,
    *,
    trainer: Trainer | None = None,
    env_factory: EnvFactory | None = None,
) -> ResumeResponse:
    """Resume training on a failed or interrupted node."""
    if request.detach:
        return _resume_detached(graph, request)
    if trainer is None:
        msg = "trainer is required unless detach is true."
        raise ValueError(msg)
    node_id = request.node_id or graph.current_node.id
    base_config = graph.resolve_config(graph.get_node(node_id))
    overrides = resolve_train_overrides(
        base_config,
        payload=request.payload,
        config_overrides=request.config_overrides,
    )
    result = resume_training(
        trainer=trainer,
        experiment_dir=graph.layout.root,
        graph=graph,
        node=node_id,
        config_overrides=overrides,
        env_factory=env_factory,
    )
    run_response = build_run_response(result.workspace, result.schedule)
    return _resume_response(run_response)


def _resume_detached(graph: ExperimentGraph[RLRunConfig], request: ResumeRequest) -> ResumeResponse:
    """Validate resume eligibility, spawn the CLI, and return immediately."""
    node_id = request.node_id or graph.current_node.id
    workspace = graph.get_node(node_id)
    if workspace.status not in RESUMABLE_NODE_STATUSES:
        msg = f"Node {workspace.id!r} is not resumable from status {workspace.status!r}."
        raise RuntimeError(msg)
    base_config = graph.resolve_config(workspace)
    overrides = resolve_train_overrides(
        base_config,
        payload=request.payload,
        config_overrides=request.config_overrides,
    )
    resolved = base_config.apply_overrides(overrides) if overrides else base_config
    schedule = compute_training_schedule(resolved.environment, resolved.algorithm)
    resolved_with_schedule = apply_training_schedule(resolved)
    workspace.save_resolved_config(resolved_with_schedule)
    experiment_dir = graph.layout.root
    config_path = experiment_dir / f".pending_resume_{workspace.id}.json"
    write_run_config(resolved_with_schedule, config_path)
    spawned = spawn_training(
        command=build_resume_command(
            experiment_dir=experiment_dir,
            config_path=config_path,
            node_id=workspace.id,
            python_executable=sys.executable,
        ),
        experiment_dir=experiment_dir,
        config_path=config_path,
    )
    run_response = build_run_response(
        workspace,
        schedule,
        detached=True,
        pid=spawned.pid,
        log_path=spawned.log_path.as_posix(),
    )
    return _resume_response(run_response)


def _resume_response(run_response: RunResponse) -> ResumeResponse:
    """Copy the shared train payload into a resume response."""
    return ResumeResponse(
        node_id=run_response.node_id,
        status=run_response.status,
        schedule=run_response.schedule,
        checkpoint_step=run_response.checkpoint_step,
        detached=run_response.detached,
        pid=run_response.pid,
        log_path=run_response.log_path,
    )
