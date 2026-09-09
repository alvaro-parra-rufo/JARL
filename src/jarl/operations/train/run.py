"""Run training in-process through the global runner."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, ClassVar

from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.io.checkpoints import CHECKPOINT_ALIAS_LATEST
from jarl.experiments.node import NodeWorkspace, validate_training_entry
from jarl.operations.contracts.compact import dataclass_to_compact_dict
from jarl.operations.train._helpers import resolve_train_overrides
from jarl.training.config import RLRunConfig
from jarl.training.launch import (
    build_train_current_command,
    spawn_training,
    write_run_config,
)
from jarl.training.presets import (
    RunFormPayload,
    form_values_to_run_config,
    preset_run_config,
)
from jarl.training.runner import run_training
from jarl.training.schedule import TrainingSchedule, apply_training_schedule, compute_training_schedule
from jarl.training.trainer import EnvFactory, Trainer

__all__ = ["RunRequest", "RunResponse", "ScheduleSummary", "build_run_response", "run"]


@dataclass(frozen=True, slots=True)
class RunRequest:
    """Inputs for running training on a node."""

    payload: RunFormPayload | None = None
    node_id: str | None = None
    config_overrides: dict[str, Any] | None = None
    create_root: bool = False
    branch: str = "main"
    label: str = ""
    detach: bool = False


@dataclass(frozen=True, slots=True)
class ScheduleSummary:
    """Compact schedule counters returned to agents."""

    actual_rollout_updates: int
    actual_optimizer_updates: int
    actual_total_timesteps: int


@dataclass(frozen=True, slots=True)
class RunResponse:
    """Outcome of ``run``."""

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


def run(
    graph: ExperimentGraph[RLRunConfig],
    request: RunRequest,
    *,
    trainer: Trainer | None = None,
    env_factory: EnvFactory | None = None,
) -> RunResponse:
    """Run training on the target node through ``jarl.training.runner``."""
    if request.detach:
        return _run_detached(graph, request)
    if trainer is None:
        msg = "trainer is required unless detach is true."
        raise ValueError(msg)
    config = None
    if request.create_root:
        config = form_values_to_run_config(request.payload) if request.payload else preset_run_config()
        base_config = config
    else:
        workspace = graph.get_node(request.node_id) if request.node_id is not None else graph.current_node
        base_config = graph.resolve_config(workspace)
    overrides = resolve_train_overrides(
        base_config,
        payload=request.payload,
        config_overrides=request.config_overrides,
    )
    result = run_training(
        trainer=trainer,
        experiment_dir=graph.layout.root,
        graph=graph,
        node=request.node_id,
        config=config,
        config_overrides=overrides,
        env_factory=env_factory,
        create_root=request.create_root,
        branch=request.branch,
        label=request.label,
    )
    return build_run_response(result.workspace, result.schedule)


def build_run_response(
    workspace: NodeWorkspace,
    schedule: TrainingSchedule,
    *,
    detached: bool | None = None,
    pid: int | None = None,
    log_path: str | None = None,
) -> RunResponse:
    """Build the standard train operation response."""
    latest = workspace.resolve_checkpoint_alias(CHECKPOINT_ALIAS_LATEST)
    return RunResponse(
        node_id=workspace.id,
        status=workspace.status.value,
        schedule=ScheduleSummary(
            actual_rollout_updates=schedule.actual_rollout_updates,
            actual_optimizer_updates=schedule.actual_optimizer_updates,
            actual_total_timesteps=schedule.actual_total_timesteps,
        ),
        checkpoint_step=latest.checkpoint_step if latest is not None else None,
        detached=detached,
        pid=pid,
        log_path=log_path,
    )


def _run_detached(graph: ExperimentGraph[RLRunConfig], request: RunRequest) -> RunResponse:
    """Validate config, spawn ``python -m jarl.training.run``, and return immediately."""
    if request.create_root and graph.as_networkx().number_of_nodes() == 0:
        config = form_values_to_run_config(request.payload) if request.payload else preset_run_config()
        workspace = graph.create_root(
            config=config,
            branch=request.branch,
            label=request.label,
            prepare=True,
        )
        graph.save()
        base_config = config
    else:
        workspace = graph.get_node(request.node_id) if request.node_id is not None else graph.current_node
        base_config = graph.resolve_config(workspace)
    validate_training_entry(workspace.status)
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
    config_path = experiment_dir / f".pending_train_{workspace.id}.json"
    write_run_config(resolved_with_schedule, config_path)
    spawned = spawn_training(
        command=build_train_current_command(
            experiment_dir=experiment_dir,
            config_path=config_path,
            python_executable=sys.executable,
            node_id=workspace.id,
        ),
        experiment_dir=experiment_dir,
        config_path=config_path,
    )
    return build_run_response(
        workspace,
        schedule,
        detached=True,
        pid=spawned.pid,
        log_path=spawned.log_path.as_posix(),
    )
