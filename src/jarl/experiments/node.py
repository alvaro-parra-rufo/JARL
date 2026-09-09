"""Node-level workspace and metadata for experiment tree nodes."""

from __future__ import annotations

import importlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self

import jax
import orbax.checkpoint as ocp

if TYPE_CHECKING:
    from flax.metrics.tensorboard import SummaryWriter

    from jarl.training.config import RLRunConfig
    from jarl.training.schedule import TrainingSchedule

from jarl.config import BaseConfig
from jarl.experiments.checkpoint_state import TrainingCheckpoint
from jarl.experiments.io import (
    ARTIFACT_KIND_MODEL,
    ArtifactRecord,
    ArtifactRegistry,
    JsonlMetricReader,
    NodeLayout,
    NodeMetricWriter,
    make_scalar_metric_log_callback,
    save_resolved_config,
)
from jarl.experiments.io.checkpoint_policy import (
    CheckpointPolicyState,
    build_checkpoint_preservation_policy,
    should_save_checkpoint,
)
from jarl.experiments.io.checkpoints import (
    CHECKPOINT_ALIAS_LATEST,
    CheckpointOrigin,
    CheckpointRecord,
    CheckpointRegistry,
    CheckpointStatus,
)
from jarl.experiments.io.execution_attempts import (
    ExecutionAttemptRecord,
    ExecutionAttemptRegistry,
    ExecutionAttemptStatus,
)
from jarl.experiments.io.model_archive import (
    MODEL_ALIAS_LATEST,
    build_model_artifact_record,
    load_model_archive,
    make_model_archive_save_callback,
    save_model_archive,
)
from jarl.experiments.run_config import CheckpointConfig, TrackingConfig
from jarl.experiments.wandb_tracking import WandbRunTracker, build_wandb_run_config
from jarl.logging import attach_file_handler
from jarl.metadata import now_iso
from jarl.utils import write_text_atomic

__all__ = [
    "NodeMetadata",
    "NodeStatus",
    "NodeWorkspace",
]


class NodeStatus(StrEnum):
    """Lifecycle state of an experiment tree node."""

    CREATED = "created"
    PREPARED = "prepared"
    TRAINING = "training"
    PAUSED = "paused"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    FAILED = "failed"


TERMINAL_NODE_STATUSES = frozenset({NodeStatus.COMPLETED})
"""Statuses that forbid training on the same node id."""

TRAINING_ENTRY_NODE_STATUSES = frozenset(
    {
        NodeStatus.CREATED,
        NodeStatus.PREPARED,
        NodeStatus.FAILED,
        NodeStatus.INTERRUPTED,
    }
)
"""Statuses that may enter the training context manager."""

RESUMABLE_NODE_STATUSES = frozenset({NodeStatus.FAILED, NodeStatus.INTERRUPTED})
"""Statuses that resume from the latest restorable checkpoint when training starts."""


def _standard_orbax_restore_args() -> ocp.args.StandardRestore:
    """Return Orbax restore args for the active JAX default device.

    Checkpoints may record CPU topology metadata even when a later run prefers
    GPU (or vice versa). ``fallback_sharding`` restores arrays on the current
    default device without forcing the saved topology.
    """
    device = jax.devices()[0]
    sharding = jax.sharding.SingleDeviceSharding(device)
    return ocp.args.StandardRestore(fallback_sharding=sharding)


def validate_training_entry(status: NodeStatus) -> None:
    """Validate that a node may enter the training context manager.

    Args:
        status: Current node lifecycle status.

    Raises:
        RuntimeError: If training cannot start from ``status``.
    """
    if status in TERMINAL_NODE_STATUSES:
        msg = "Cannot re-enter a completed node. Use graph.extend() or graph.fork() to continue training."
        raise RuntimeError(msg)
    if status == NodeStatus.TRAINING:
        msg = "Node is already in training."
        raise RuntimeError(msg)
    if status not in TRAINING_ENTRY_NODE_STATUSES:
        msg = f"Cannot start training from status {status!r}."
        raise RuntimeError(msg)


def validate_prepare_transition(status: NodeStatus) -> None:
    """Validate that a node may transition to ``prepared``.

    Args:
        status: Current node lifecycle status.

    Raises:
        RuntimeError: If the node cannot be prepared from ``status``.
    """
    if status != NodeStatus.CREATED:
        msg = f"Cannot prepare node in status {status!r}."
        raise RuntimeError(msg)


@dataclass
class NodeMetadata:
    """Metadata for a single node in the experiment tree.

    Args:
        id: Computed node identifier (`{branch}_{label?}_{uuid8}`).
        parent_id: ID of the parent node (`None` for root).
        branch: Branch name this node belongs to.
        step: Training step reported by the trainer (node-relative, 0-based).
        label: Optional human/LLM-readable name.
        description: Optional explanation of what/why this fork exists.
        metadata: Extensible dictionary for workflow customization.
        config_overrides: Sparse dict — true diff from parent config.
        status: Lifecycle state of this node.
        created_at: ISO-8601 creation timestamp.
        updated_at: ISO-8601 last-update timestamp.
        content_hash: Informational hash of resolved config + parent + timestamp.
        workspace_cls: Fully qualified class name for reconstruction.
        parent_checkpoint_step: Orbax step restored from the parent on fork/extend.
    """

    id: str = ""
    parent_id: str | None = None
    branch: str = "main"
    step: int = 0
    label: str = ""
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    config_overrides: dict[str, Any] = field(default_factory=dict)
    status: NodeStatus = NodeStatus.CREATED
    created_at: str = ""
    updated_at: str = ""
    content_hash: str = ""
    workspace_cls: str = ""
    parent_checkpoint_step: int | None = None

    def save(self, path: str | Path) -> Path:
        """Serialize node metadata to a JSON file.

        Args:
            path: Destination file path.

        Returns:
            The path written to.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = dict(self.__dict__)
        write_text_atomic(path, json.dumps(data, indent=2, default=str))
        return path

    @classmethod
    def load(cls, path: str | Path) -> NodeMetadata:
        """Deserialize node metadata from a JSON file.

        Args:
            path: Source file path.

        Returns:
            Reconstructed `NodeMetadata` instance.
        """
        data = json.loads(Path(path).read_text())
        if "status" in data:
            data["status"] = NodeStatus(data["status"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class NodeWorkspace:
    """Workspace for a single node in the experiment tree.

    Manages the per-node directory structure (checkpoints, TensorBoard events,
    logs, metrics) and provides a context manager for training lifecycle.

    Subclass this to customize enter/exit behavior for experiment-specific needs
    (e.g., auto-flushing TensorBoard on exit, emergency checkpoint on failure).

    Args:
        node_dir: Root directory for this node's artifacts.
        node_metadata: Metadata describing this node.
        parent_checkpoint_path: Fallback checkpoint path from the parent node.
        logger_name: Logger name for file handler attachment.
        tracking: Tracking settings controlling TensorBoard and future backends.
    """

    def __init__(
        self,
        node_dir: str | Path,
        node_metadata: NodeMetadata,
        parent_checkpoint_path: Path | None = None,
        parent_checkpoint_step: int | None = None,
        logger_name: str = "jarl",
        tracking: TrackingConfig | None = None,
    ) -> None:
        """Initialize the node workspace."""
        self._dir = Path(node_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._layout = NodeLayout(self._dir)
        self._meta = node_metadata
        self._parent_ckpt_path = parent_checkpoint_path
        self._parent_ckpt_step = (
            parent_checkpoint_step if parent_checkpoint_step is not None else node_metadata.parent_checkpoint_step
        )
        self._model_archive_algorithm_name: str | None = None
        self._model_archive_env_id: str | None = None
        self._logger_name = logger_name
        self._tracking = tracking or TrackingConfig()
        self._ckpt_manager: ocp.CheckpointManager | None = None
        self._log_handler: logging.FileHandler | None = None
        self._run_metric_writer: NodeMetricWriter | None = None
        self._adhoc_metric_writer: NodeMetricWriter | None = None
        self._wandb_tracker: WandbRunTracker | None = None
        self._wandb_run_config: dict[str, Any] | None = None
        self._artifact_registry: ArtifactRegistry | None = None
        self._checkpoint_registry: CheckpointRegistry | None = None
        self._checkpoint_policy_state = CheckpointPolicyState()
        self._execution_attempt_registry: ExecutionAttemptRegistry | None = None
        self._active_attempt_id: int | None = None

    @classmethod
    def from_import_path(cls, import_path: str) -> type[NodeWorkspace]:
        """Load a `NodeWorkspace` subclass from a fully qualified import path.

        Args:
            import_path: Fully qualified name (`module.qualname`) stored in
                node metadata.

        Returns:
            The resolved workspace class.

        Raises:
            ImportError: If the module or attribute cannot be imported.
            TypeError: If the resolved object is not a `NodeWorkspace` subclass.
        """
        if not import_path:
            return cls
        module_path, _, qualname = import_path.rpartition(".")
        if not module_path:
            msg = f"Invalid workspace import path: {import_path!r}"
            raise ImportError(msg)
        module = importlib.import_module(module_path)
        obj: object = module
        for attr in qualname.split("."):
            obj = getattr(obj, attr)
        if not isinstance(obj, type) or not issubclass(obj, cls):
            msg = f"Import path {import_path!r} is not a {cls.__name__} subclass"
            raise TypeError(msg)
        return obj

    def __repr__(self) -> str:
        """Return a string representation of the node workspace."""
        return f"NodeWorkspace(id={self._meta.id!r}, branch={self._meta.branch!r}, status={self._meta.status!r})"

    # ---- Context manager ---- #

    def bind_wandb_run_context(self, config: RLRunConfig, schedule: TrainingSchedule) -> None:
        """Attach resolved config and schedule for enriched W&B initialization.

        Call before entering the training context (for example from ``run_training``)
        so ``WandbRunTracker.init`` receives hyperparameters and derived schedule
        counters in addition to node lineage metadata.

        Args:
            config: Fully resolved run config persisted for this attempt.
            schedule: Derived training schedule counters for the run.
        """
        self._wandb_run_config = build_wandb_run_config(
            node_metadata=self._meta,
            node_layout=self._layout,
            config=config,
            schedule=schedule,
        )

    def __enter__(self) -> Self:
        """Enter training context: mark node as training, attach log handler."""
        validate_training_entry(self._meta.status)
        resume_of_attempt_id = None
        checkpoint_step_at_start = None
        if self._meta.status in RESUMABLE_NODE_STATUSES:
            resume_record = self.resolve_resume_checkpoint_record()
            if resume_record is not None:
                resume_of_attempt_id = self._execution_attempts().latest_closed_attempt_id()
                checkpoint_step_at_start = resume_record.checkpoint_step
            elif any(record.status == CheckpointStatus.SAVED for record in self._checkpoints().list()):
                msg = "No restorable checkpoint available to resume this node."
                raise RuntimeError(msg)
        attempt = self._execution_attempts().open_attempt(
            resume_of_attempt_id=resume_of_attempt_id,
            checkpoint_step_at_start=checkpoint_step_at_start,
        )
        self._active_attempt_id = attempt.attempt_id
        self._execution_attempts().save()
        self._meta.status = NodeStatus.TRAINING
        self._meta.updated_at = now_iso()
        self._save_metadata()
        self._log_handler = attach_file_handler(self._logger_name, self.log_path)
        self._run_metric_writer = NodeMetricWriter(
            jsonl_path=self.metrics_jsonl_path,
            tensorboard_dir=self.tensorboard_dir,
            enable_tensorboard=self._tracking.track_tensorboard,
        )
        self._run_metric_writer.open()
        if self._tracking.track_wandb:
            wandb_config = self._wandb_run_config
            if wandb_config is None:
                wandb_config = {
                    "node_id": self._meta.id,
                    "branch": self._meta.branch,
                    "parent_id": self._meta.parent_id,
                }
            self._wandb_tracker = WandbRunTracker(
                tracking=self._tracking,
                wandb_dir=self.wandb_dir,
                run_name=self._meta.id,
                config=wandb_config,
            )
            self._wandb_tracker.init()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit training context: update status, detach log handler, flush writers."""
        if exc_type is KeyboardInterrupt:
            self._meta.status = NodeStatus.INTERRUPTED
            attempt_status = ExecutionAttemptStatus.INTERRUPTED
        elif exc_type is not None:
            self._meta.status = NodeStatus.FAILED
            attempt_status = ExecutionAttemptStatus.FAILED
        else:
            self._meta.status = NodeStatus.COMPLETED
            attempt_status = ExecutionAttemptStatus.COMPLETED
        self._meta.updated_at = now_iso()
        self._close_active_execution_attempt(
            status=attempt_status,
            exc_type=exc_type,
            exc_val=exc_val,
        )
        self._save_metadata()

        if self._wandb_tracker is not None:
            self._wandb_tracker.finish()
            self._wandb_tracker = None

        if self._run_metric_writer is not None:
            self._run_metric_writer.close()
            self._run_metric_writer = None

        if self._log_handler is not None:
            logger = logging.getLogger(self._logger_name)
            logger.removeHandler(self._log_handler)
            self._log_handler.close()
            self._log_handler = None

        self._finalize_checkpoint_aliases(exc_type is None)

    @property
    def tracking(self) -> TrackingConfig:
        """Tracking settings for this node workspace."""
        return self._tracking

    @property
    def id(self) -> str:
        """Node identifier."""
        return self._meta.id

    @property
    def node_metadata(self) -> NodeMetadata:
        """Full node metadata."""
        return self._meta

    @property
    def branch(self) -> str:
        """Branch this node belongs to."""
        return self._meta.branch

    @property
    def status(self) -> NodeStatus:
        """Current lifecycle status."""
        return self._meta.status

    # ---- Lifecycle ---- #

    def prepare(self) -> Path:
        """Mark the node as prepared without opening training writers.

        Returns:
            Path to the written node metadata file.

        Raises:
            RuntimeError: If the node is not in ``created`` status.
        """
        validate_prepare_transition(self._meta.status)
        self._meta.status = NodeStatus.PREPARED
        self._meta.updated_at = now_iso()
        return self._save_metadata()

    def mark_interrupted(self) -> Path:
        """Mark a stale ``training`` node as ``interrupted`` after operator reconciliation.

        Returns:
            Path to the written node metadata file.

        Raises:
            RuntimeError: If the node is not currently ``training``.
        """
        if self._meta.status != NodeStatus.TRAINING:
            msg = f"Cannot mark node as interrupted from status {self._meta.status!r}."
            raise RuntimeError(msg)
        self._meta.status = NodeStatus.INTERRUPTED
        self._meta.updated_at = now_iso()
        return self._save_metadata()

    def resolve_resume_checkpoint_record(self) -> CheckpointRecord | None:
        """Return the newest restorable saved checkpoint for resume, if any.

        Walks saved checkpoints from highest to lowest step and returns the
        first entry that passes registry and on-disk restore validation.

        Returns:
            Restorable checkpoint record, or ``None`` when no candidate qualifies.
        """
        candidates = sorted(
            (record for record in self._checkpoints().list() if record.status == CheckpointStatus.SAVED),
            key=lambda record: record.checkpoint_step,
            reverse=True,
        )
        for record in candidates:
            try:
                return self.validate_saved_checkpoint(record.checkpoint_step)
            except ValueError:
                continue
        return None

    def load_resume_checkpoint(self) -> object:
        """Load the newest restorable checkpoint for this node.

        Call this after re-entering a failed or interrupted node and before
        continuing training so trainer state matches ``checkpoint_step_at_start``
        recorded on the new execution attempt.

        Returns:
            Restored checkpoint payload.

        Raises:
            ValueError: If no restorable checkpoint exists locally or on the parent.
        """
        record = self.resolve_resume_checkpoint_record()
        if record is None:
            msg = f"No restorable checkpoint available for node {self._meta.id!r}."
            raise ValueError(msg)
        return self.load_checkpoint(record.checkpoint_step)

    def list_execution_attempts(self) -> list[ExecutionAttemptRecord]:
        """Return all execution attempts registered for this node."""
        return self._execution_attempts().list()

    def latest_execution_attempt(self) -> ExecutionAttemptRecord | None:
        """Return the most recent execution attempt, if any."""
        return self._execution_attempts().latest()

    # ---- Path accessors ---- #

    @property
    def path(self) -> Path:
        """Root path of this node workspace."""
        return self._dir

    @property
    def layout(self) -> NodeLayout:
        """Path layout for this node workspace."""
        return self._layout

    @property
    def config_path(self) -> Path:
        """Path to the resolved config snapshot for this node."""
        return self._layout.config_path

    @property
    def checkpoint_dir(self) -> Path:
        """Path to Orbax checkpoint storage."""
        return self._layout.checkpoint_dir

    @property
    def checkpoints_registry_path(self) -> Path:
        """Path to the checkpoint metadata index."""
        return self._layout.checkpoints_registry_path

    @property
    def parent_checkpoint_step(self) -> int | None:
        """Orbax step restored from the parent when this node was forked."""
        return self._parent_ckpt_step

    @property
    def tensorboard_dir(self) -> Path:
        """Path to per-node TensorBoard events."""
        return self._layout.tensorboard_dir

    @property
    def wandb_dir(self) -> Path:
        """Path to Weights & Biases run data."""
        return self._layout.wandb_dir

    @property
    def models_dir(self) -> Path:
        """Path to saved model archives."""
        return self._layout.models_dir

    @property
    def rollouts_dir(self) -> Path:
        """Path to materialized checkpoint rollouts."""
        return self._layout.rollouts_dir

    @property
    def videos_dir(self) -> Path:
        """Path to rendered rollout videos."""
        return self._layout.videos_dir

    @property
    def log_path(self) -> Path:
        """Path to the per-node training log file."""
        return self._layout.log_path

    @property
    def metrics_jsonl_path(self) -> Path:
        """Path to the primary incremental metrics log."""
        return self._layout.metrics_jsonl_path

    @property
    def video_metrics_jsonl_path(self) -> Path:
        """Path to video-associated metrics log."""
        return self._layout.video_metrics_jsonl_path

    @property
    def config_overrides_path(self) -> Path:
        """Path to the per-node config overrides JSON file."""
        return self._layout.config_overrides_path

    @property
    def metadata_path(self) -> Path:
        """Path to the per-node metadata JSON file."""
        return self._layout.metadata_path

    @property
    def artifacts_registry_path(self) -> Path:
        """Path to the exportable artifact registry."""
        return self._layout.artifacts_registry_path

    def save_resolved_config(self, config: BaseConfig) -> Path:
        """Persist a fully resolved config snapshot for this node.

        Args:
            config: Resolved configuration provided by the caller.

        Returns:
            Path to the written ``config.json`` file.
        """
        return save_resolved_config(self.config_path, config)

    # ---- Metrics ---- #

    @property
    def metric_writer(self) -> NodeMetricWriter:
        """Unified JSONL and optional TensorBoard metric writer."""
        if self._run_metric_writer is not None:
            return self._run_metric_writer
        if self._adhoc_metric_writer is None:
            self._adhoc_metric_writer = NodeMetricWriter(
                jsonl_path=self.metrics_jsonl_path,
                tensorboard_dir=self.tensorboard_dir,
                enable_tensorboard=False,
            )
            self._adhoc_metric_writer.open()
        return self._adhoc_metric_writer

    @property
    def tb_writer(self) -> SummaryWriter:
        """Flax TensorBoard writer when ``track_tensorboard`` is enabled.

        Raises:
            RuntimeError: If TensorBoard tracking is disabled or the writer is
                not active.
        """
        if not self._tracking.track_tensorboard:
            msg = "TensorBoard tracking is disabled for this workspace."
            raise RuntimeError(msg)
        summary_writer = self.metric_writer.summary_writer
        if summary_writer is None:
            msg = "TensorBoard writer is not available."
            raise RuntimeError(msg)
        return summary_writer

    # ---- Checkpointing ---- #

    def init_checkpoint_manager(
        self,
        max_to_keep: int | None = None,
        save_interval_steps: int = 1,
    ) -> ocp.CheckpointManager:
        """Initialize the orbax `CheckpointManager` for this node.

        Pinned checkpoint steps are preserved even when ``max_to_keep`` would
        otherwise delete older checkpoints.

        Args:
            max_to_keep: Maximum checkpoints to retain (`None` = unlimited).
            save_interval_steps: Save every N steps.

        Returns:
            Configured `CheckpointManager` instance.
        """
        options = ocp.CheckpointManagerOptions(
            save_interval_steps=save_interval_steps,
            preservation_policy=build_checkpoint_preservation_policy(
                max_to_keep=max_to_keep,
                get_pinned_steps=self._pinned_checkpoint_steps,
            ),
            enable_async_checkpointing=False,
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._ckpt_manager = ocp.CheckpointManager(
            directory=self.checkpoint_dir.resolve(),
            options=options,
        )
        return self._ckpt_manager

    def _pinned_checkpoint_steps(self) -> set[int]:
        """Return Orbax steps pinned for downstream fork/extend operations."""
        if not self.checkpoints_registry_path.exists():
            return set()
        return self._checkpoints().pinned_checkpoint_steps

    @property
    def ckpt_manager(self) -> ocp.CheckpointManager:
        """The orbax `CheckpointManager` (must be initialized via `init_checkpoint_manager`).

        Raises:
            RuntimeError: If not yet initialized.
        """
        if self._ckpt_manager is None:
            msg = "Checkpoint manager not initialized. Call init_checkpoint_manager() first."
            raise RuntimeError(msg)
        return self._ckpt_manager

    def bind_model_archive_context(self, *, algorithm_name: str, env_id: str) -> None:
        """Store manifest metadata used by host-side model archive saves."""
        self._model_archive_algorithm_name = algorithm_name
        self._model_archive_env_id = env_id

    def save_checkpoint(
        self,
        step: int,
        state: dict[str, Any] | TrainingCheckpoint,
        metrics: dict | None = None,
        force: bool = True,
        *,
        global_step: int | None = None,
        optimizer_updates: int | None = None,
    ) -> bool:
        """Save a checkpoint at the given step.

        This method always attempts to persist immediately and bypasses Orbax
        interval policy unless ``force=False`` is passed explicitly.

        Args:
            step: Node-relative training step.
            state: Pytree state to persist.
            metrics: Optional metrics associated with this checkpoint.
            force: Save even if Orbax interval policy would skip.
            global_step: Optional global training step supplied by the caller.
            optimizer_updates: Optional optimizer update counter supplied by the caller.

        Returns:
            `True` if saved, `False` if skipped.
        """
        payload = state.to_pytree() if isinstance(state, TrainingCheckpoint) else state
        self._meta.step = step
        self._meta.updated_at = now_iso()
        saved = self.ckpt_manager.save(step, args=ocp.args.StandardSave(payload), metrics=metrics, force=force)
        if saved:
            self._register_saved_checkpoint(
                step=step,
                metrics=metrics,
                global_step=global_step,
                optimizer_updates=optimizer_updates,
            )
        return saved

    def save_checkpoint_if_due(
        self,
        step: int,
        state: dict[str, Any] | TrainingCheckpoint,
        *,
        policy: CheckpointConfig,
        total_steps: int | None = None,
        elapsed_seconds: float | None = None,
        metrics: dict | None = None,
        global_step: int | None = None,
        optimizer_updates: int | None = None,
        monotonic_now: float | None = None,
    ) -> bool:
        """Save a checkpoint when configured policies request it at this step.

        The caller must supply progress inputs such as ``total_steps`` and
        ``elapsed_seconds``; the workspace does not infer training schedules.

        Args:
            step: Current node-relative training step.
            state: Pytree state to persist when a save is due.
            policy: Checkpoint policy configuration to evaluate.
            total_steps: Planned total steps for percentage milestones.
            elapsed_seconds: Elapsed training time for time-based policies.
            metrics: Optional metrics associated with this checkpoint.
            global_step: Optional global training step supplied by the caller.
            optimizer_updates: Optional optimizer update counter supplied by the caller.
            monotonic_now: Optional monotonic clock override for tests.

        Returns:
            `True` if a checkpoint was saved, `False` otherwise.
        """
        now = monotonic_now if monotonic_now is not None else time.monotonic()
        if not should_save_checkpoint(
            step,
            config=policy,
            state=self._checkpoint_policy_state,
            total_steps=total_steps,
            elapsed_seconds=elapsed_seconds,
            monotonic_now=now,
        ):
            return False
        return self.save_checkpoint(
            step,
            state,
            metrics=metrics,
            force=True,
            global_step=global_step,
            optimizer_updates=optimizer_updates,
        )

    def save_training_checkpoint(
        self,
        step: int,
        checkpoint: TrainingCheckpoint,
        *,
        metrics: dict | None = None,
        force: bool = True,
        global_step: int | None = None,
        optimizer_updates: int | None = None,
    ) -> bool:
        """Save a typed training checkpoint at the given step."""
        return self.save_checkpoint(
            step,
            checkpoint,
            metrics=metrics,
            force=force,
            global_step=global_step if global_step is not None else checkpoint.global_step,
            optimizer_updates=optimizer_updates if optimizer_updates is not None else checkpoint.optimizer_updates,
        )

    def has_saved_checkpoint(self, checkpoint_step: int) -> bool:
        """Return whether a saved checkpoint record exists for ``checkpoint_step``."""
        record = self._checkpoints().get(checkpoint_step)
        return record is not None and record.status == CheckpointStatus.SAVED

    def save_training_checkpoint_if_absent(
        self,
        step: int,
        checkpoint: TrainingCheckpoint,
        *,
        metrics: dict | None = None,
        global_step: int | None = None,
        optimizer_updates: int | None = None,
    ) -> bool:
        """Save a typed checkpoint unless that step is already on disk.

        Trainers use this for the post-loop final save when the last periodic
        checkpoint may already have been written at the same global step.
        """
        if self.has_saved_checkpoint(step):
            return False
        return self.save_training_checkpoint(
            step,
            checkpoint,
            metrics=metrics,
            force=True,
            global_step=global_step,
            optimizer_updates=optimizer_updates,
        )

    def save_training_checkpoint_if_due(
        self,
        step: int,
        checkpoint: TrainingCheckpoint,
        *,
        policy: CheckpointConfig,
        total_steps: int | None = None,
        elapsed_seconds: float | None = None,
        metrics: dict | None = None,
        global_step: int | None = None,
        optimizer_updates: int | None = None,
        monotonic_now: float | None = None,
    ) -> bool:
        """Save a typed checkpoint when configured policies request it at this step."""
        now = monotonic_now if monotonic_now is not None else time.monotonic()
        if not should_save_checkpoint(
            step,
            config=policy,
            state=self._checkpoint_policy_state,
            total_steps=total_steps,
            elapsed_seconds=elapsed_seconds,
            monotonic_now=now,
        ):
            return False
        return self.save_training_checkpoint(
            step,
            checkpoint,
            metrics=metrics,
            force=True,
            global_step=global_step,
            optimizer_updates=optimizer_updates,
        )

    def load_checkpoint(self, step: int | None = None) -> object:
        """Load a checkpoint, falling back to parent if no local checkpoints exist.

        Tries local checkpoints first. If none exist and a parent checkpoint
        path was provided, loads from the parent using ``parent_checkpoint_step``
        when set. Lazily initializes a local checkpoint manager when ``checkpoint/``
        already contains saved steps.

        Args:
            step: Step to restore. `None` restores the latest local checkpoint.

        Returns:
            Restored pytree state.

        Raises:
            ValueError: If no checkpoints exist locally or from parent.
        """
        manager = self._ensure_local_checkpoint_manager()
        if manager is not None:
            latest = manager.latest_step()
            if latest is not None:
                target = step if step is not None else latest
                return manager.restore(target, args=_standard_orbax_restore_args())

        return self.load_parent_checkpoint(self._parent_ckpt_step)

    def load_training_checkpoint[T: TrainingCheckpoint](
        self,
        checkpoint_cls: type[T],
        step: int | None = None,
    ) -> T:
        """Load and deserialize a typed training checkpoint."""
        tree = self.load_checkpoint(step)
        if not isinstance(tree, dict):
            msg = "Checkpoint payload must be a mapping."
            raise TypeError(msg)
        loaded = checkpoint_cls.from_pytree(tree)
        if not isinstance(loaded, checkpoint_cls):
            msg = f"Checkpoint payload did not deserialize to {checkpoint_cls.__name__}."
            raise TypeError(msg)
        return loaded

    def load_parent_checkpoint(self, step: int | None = None) -> object:
        """Load the parent node's checkpoint.

        Args:
            step: Orbax step to restore. `None` uses ``parent_checkpoint_step``
                when set, otherwise the latest parent checkpoint.

        Returns:
            Restored pytree state from the parent.

        Raises:
            ValueError: If no parent checkpoint path was provided or parent has no checkpoints.
        """
        if self._parent_ckpt_path is None:
            msg = "No parent checkpoint path available."
            raise ValueError(msg)
        parent_mgr = ocp.CheckpointManager(directory=self._parent_ckpt_path.resolve())
        target = step if step is not None else self._parent_ckpt_step
        if target is None:
            target = parent_mgr.latest_step()
        if target is None:
            msg = f"No checkpoints found in parent path: {self._parent_ckpt_path}"
            raise ValueError(msg)
        return parent_mgr.restore(target, args=_standard_orbax_restore_args())

    def validate_saved_checkpoint(self, checkpoint_step: int) -> CheckpointRecord:
        """Validate that a checkpoint is registered, saved, and restorable.

        Args:
            checkpoint_step: Orbax step identifier to validate.

        Returns:
            Matching saved checkpoint record.

        Raises:
            ValueError: If metadata or on-disk storage is missing or unrestorable.
            KeyError: If the checkpoint record is missing or not saved.
        """
        try:
            record = self._checkpoints().require_saved(checkpoint_step)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
        if not self.checkpoint_dir.exists():
            msg = f"Checkpoint storage missing for step {checkpoint_step} under {self.checkpoint_dir}"
            raise ValueError(msg)
        manager = self._ensure_local_checkpoint_manager()
        if manager is None:
            msg = f"Checkpoint manager unavailable for step {checkpoint_step} under {self.checkpoint_dir}"
            raise ValueError(msg)
        try:
            manager.restore(checkpoint_step, args=_standard_orbax_restore_args())
        except Exception as exc:
            msg = f"Checkpoint step {checkpoint_step} could not be restored from {self.checkpoint_dir}"
            raise ValueError(msg) from exc
        return record

    def pin_checkpoint(self, checkpoint_step: int) -> Path:
        """Pin a checkpoint step so downstream nodes can reference it safely.

        Args:
            checkpoint_step: Orbax step identifier to pin.

        Returns:
            Path to the written checkpoint registry file.
        """
        self.validate_saved_checkpoint(checkpoint_step)
        registry = self._checkpoints()
        registry.pin(checkpoint_step)
        return registry.save()

    def promote_checkpoint_best(
        self,
        checkpoint_step: int,
        *,
        metric_name: str,
        metric_value: float,
        reason: str,
    ) -> Path:
        """Promote a checkpoint to the ``best`` alias with caller metadata.

        Args:
            checkpoint_step: Orbax step identifier to promote.
            metric_name: Metric used for the promotion decision.
            metric_value: Metric value recorded with the promotion.
            reason: Caller-provided explanation for the promotion.

        Returns:
            Path to the written checkpoint registry file.

        Raises:
            KeyError: If the checkpoint record does not exist or is not saved.
        """
        registry = self._checkpoints()
        registry.promote_best(
            checkpoint_step,
            metric_name=metric_name,
            metric_value=metric_value,
            reason=reason,
        )
        return registry.save()

    def resolve_checkpoint_alias(self, alias: str) -> CheckpointRecord | None:
        """Resolve a checkpoint alias such as ``latest`` or ``best``.

        Args:
            alias: Alias name stored in ``checkpoints.json``.

        Returns:
            Matching checkpoint record, or ``None`` when the alias is unset.
        """
        return self._checkpoints().resolve_alias(alias)

    def list_checkpoints(self) -> list[CheckpointRecord]:
        """Return all checkpoint records registered for this node."""
        return self._checkpoints().list()

    def save_model_archive(
        self,
        name: str,
        step: int,
        *,
        policy: object,
        critic: object,
        alias: str | None = None,
        metadata: dict[str, Any] | None = None,
        algorithm_name: str | None = None,
        env_id: str | None = None,
    ) -> Path:
        """Save policy and critic pytrees into a ``.model`` archive.

        Args:
            name: Model identifier used in ``artifacts.json``.
            step: Training step associated with the archive.
            policy: Policy pytree payload.
            critic: Critic pytree payload.
            alias: Optional alias such as ``latest`` to also register under.
            metadata: Optional extra artifact metadata.
            algorithm_name: Optional algorithm identifier stored in the manifest.
            env_id: Optional environment identifier stored in the manifest.

        Returns:
            Path to the written ``.model`` archive.
        """
        self.models_dir.mkdir(parents=True, exist_ok=True)
        archive_path = save_model_archive(
            self.models_dir,
            name=name,
            step=step,
            components={"policy": policy, "critic": critic},
            algorithm_name=algorithm_name or self._model_archive_algorithm_name,
            env_id=env_id or self._model_archive_env_id,
        )
        registry = self._artifacts()
        record = build_model_artifact_record(name=name, step=step, metadata=metadata)
        registry.register(record)
        if alias is not None:
            registry.set_model_alias(alias, name=name, step=step)
        registry.save()
        return archive_path

    def save_model_archive_from_host(
        self,
        step: int,
        name: str,
        policy: object,
        critic: object,
    ) -> Path:
        """Host-side helper bound to JIT callbacks for model archive saves.

        Args:
            step: Training step associated with the archive.
            name: Model identifier used in ``artifacts.json``.
            policy: Policy pytree payload.
            critic: Critic pytree payload.

        Returns:
            Path to the written ``.model`` archive.
        """
        policy_payload = policy.params if hasattr(policy, "params") else policy
        critic_payload = critic.params if hasattr(critic, "params") else critic
        return self.save_model_archive(
            name,
            step,
            policy=policy_payload,
            critic=critic_payload,
            alias=MODEL_ALIAS_LATEST,
        )

    def make_model_archive_save_callback(self) -> Callable[[Any, Any, Any, Any], None]:
        """Return a ``jax.debug.callback`` helper bound to this workspace.

        Returns:
            Host callback that delegates model archive IO to
            ``save_model_archive_from_host``.
        """
        return make_model_archive_save_callback(self.save_model_archive_from_host)

    def load_model_archive(self, name: str, step: int) -> dict[str, object]:
        """Restore a registered ``.model`` archive by artifact identity.

        Args:
            name: Model identifier used in ``artifacts.json``.
            step: Training step associated with the archive.

        Returns:
            Mapping from component name to restored pytree payload.

        Raises:
            KeyError: If no matching model artifact is registered.
        """
        archive_path = self.resolve_artifact_path(ARTIFACT_KIND_MODEL, name, step)
        return load_model_archive(archive_path)

    def promote_model_best(
        self,
        name: str,
        step: int,
        *,
        metric_name: str,
        metric_value: float,
        reason: str,
    ) -> Path:
        """Promote a canonical model archive to the ``best`` alias pointer.

        Args:
            name: Canonical model identifier.
            step: Training step associated with the canonical archive.
            metric_name: Metric used for the promotion decision.
            metric_value: Metric value recorded with the promotion.
            reason: Caller-provided explanation for the promotion.

        Returns:
            Path to the written artifact registry file.
        """
        registry = self._artifacts()
        registry.promote_model_best(
            name=name,
            step=step,
            metric_name=metric_name,
            metric_value=metric_value,
            reason=reason,
        )
        return registry.save()

    def resolve_model_alias(self, alias: str) -> ArtifactRecord | None:
        """Resolve a model alias pointer to its canonical artifact record.

        Args:
            alias: Alias name such as ``latest``, ``final``, or ``best``.

        Returns:
            Canonical artifact record, or ``None`` when the alias is unset.
        """
        return self._artifacts().resolve_model_alias(alias)

    def _register_artifact(self, record: ArtifactRecord) -> Path:
        """Register an artifact from a subclass hook.

        Args:
            record: Artifact entry keyed by ``(kind, name, step)``.

        Returns:
            Path to the written registry file.
        """
        return self.register_artifact(record)

    # ---- Metrics access ---- #

    @property
    def wandb_active(self) -> bool:
        """Return whether W&B tracking is active for this run."""
        return self._wandb_tracker is not None and self._wandb_tracker.active

    def log_scalar(self, step: int, name: str, value: float) -> None:
        """Append one scalar metric to the node's JSONL log.

        This method performs host-side IO and must be called from outside
        ``jax.jit``. For full-JIT training loops use
        ``make_scalar_metric_log_callback`` with ``jax.debug.callback``.

        Args:
            step: Training step associated with the measurement.
            name: Metric identifier.
            value: Scalar metric value.
        """
        self.metric_writer.log_scalar(step, name, value)
        if self._wandb_tracker is not None:
            self._wandb_tracker.log_scalar(step, name, value)

    def make_scalar_metric_log_callback(self) -> Callable[[Any, Any, Any], None]:
        """Return a ``jax.debug.callback`` logger bound to this workspace.

        Create the callback on the host before entering compiled training code.
        The returned function accepts ``(step, name, value)`` scalars only and
        must not be passed a ``NodeWorkspace`` from inside ``jax.jit``.

        Returns:
            Host callback that delegates metric IO to ``log_scalar``.
        """
        return make_scalar_metric_log_callback(self.log_scalar)

    def log_scalars(self, step: int, **metrics: float) -> None:
        """Append multiple scalar metrics at the same training step.

        Args:
            step: Training step associated with the measurements.
            **metrics: Metric name-value pairs.
        """
        for name, metric_value in metrics.items():
            self.log_scalar(step, name, metric_value)

    def latest_metrics(self) -> dict[str, float]:
        """Return the latest scalar metrics recorded for this node.

        Returns:
            Mapping from metric name to latest scalar value.
        """
        if self.metrics_jsonl_path.exists():
            return JsonlMetricReader(self.metrics_jsonl_path).latest()
        return {}

    def save_metrics(self, values: dict[str, float] | None = None) -> Path:
        """Record metric values to the node's JSONL log.

        Prefer `log_scalar` or `log_scalars` for incremental logging. This
        helper uses the node metadata step and writes to `metrics.jsonl`.

        Args:
            values: Optional new metric values to record before flushing.

        Returns:
            Path to the JSONL metrics file.
        """
        if values:
            self.log_scalars(self._meta.step, **values)
        self.metric_writer.flush()
        return self.metrics_jsonl_path

    # ---- Artifacts ---- #

    def register_artifact(self, record: ArtifactRecord) -> Path:
        """Register an exportable artifact and persist ``artifacts.json``.

        Args:
            record: Artifact entry keyed by ``(kind, name, step)``.

        Returns:
            Path to the written registry file.
        """
        registry = self._artifacts()
        registry.register(record)
        return registry.save()

    def list_artifacts(self) -> list[ArtifactRecord]:
        """Return all artifacts registered for this node."""
        return self._artifacts().list()

    def get_artifact(self, kind: str, name: str, step: int) -> ArtifactRecord | None:
        """Return one artifact by ``(kind, name, step)`` when present.

        Args:
            kind: Artifact category.
            name: Artifact identifier within ``kind``.
            step: Training step associated with the artifact.

        Returns:
            Matching record, or ``None`` when not registered.
        """
        return self._artifacts().get(kind, name, step)

    def resolve_artifact_path(self, kind: str, name: str, step: int) -> Path:
        """Resolve the absolute path for a registered artifact.

        Args:
            kind: Artifact category.
            name: Artifact identifier within ``kind``.
            step: Training step associated with the artifact.

        Returns:
            Absolute path under the node directory.

        Raises:
            KeyError: If no artifact matches the identity.
            ValueError: If the stored relative path escapes the node root.
        """
        path = self._artifacts().resolve_path(kind, name, step)
        if path is None:
            msg = f"No artifact registered for identity ({kind!r}, {name!r}, {step})"
            raise KeyError(msg)
        return path

    # ---- Config overrides ---- #

    def save_config_overrides(self, overrides: dict[str, Any]) -> Path:
        """Persist config overrides (sparse delta from parent) to disk.

        Args:
            overrides: The config override dictionary.

        Returns:
            Path to the saved file.
        """
        self._meta.config_overrides = overrides
        path = self.config_overrides_path
        write_text_atomic(path, json.dumps(overrides, indent=2, default=str))
        self._save_metadata()
        return path

    def load_config_overrides(self) -> dict[str, Any]:
        """Load config overrides from disk.

        Returns:
            The override dictionary, or empty dict if file doesn't exist.
        """
        if self.config_overrides_path.exists():
            return json.loads(self.config_overrides_path.read_text())
        return {}

    # ---- Internal ---- #

    def _artifacts(self) -> ArtifactRegistry:
        """Return the lazily loaded artifact registry for this node."""
        if self._artifact_registry is None:
            registry = ArtifactRegistry(self.artifacts_registry_path, root=self._dir)
            registry.load()
            self._artifact_registry = registry
        return self._artifact_registry

    def _checkpoints(self) -> CheckpointRegistry:
        """Return the lazily loaded checkpoint registry for this node."""
        if self._checkpoint_registry is None:
            registry = CheckpointRegistry(self.checkpoints_registry_path)
            registry.load()
            self._checkpoint_registry = registry
        return self._checkpoint_registry

    def _execution_attempts(self) -> ExecutionAttemptRegistry:
        """Return the lazily loaded execution attempt registry for this node."""
        if self._execution_attempt_registry is None:
            registry = ExecutionAttemptRegistry(self._layout.execution_attempts_path)
            registry.load()
            self._execution_attempt_registry = registry
        return self._execution_attempt_registry

    def _close_active_execution_attempt(
        self,
        *,
        status: ExecutionAttemptStatus,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
    ) -> None:
        if self._active_attempt_id is None:
            return
        checkpoint_step_at_end = None
        latest = self.resolve_checkpoint_alias(CHECKPOINT_ALIAS_LATEST)
        if latest is not None and latest.status == CheckpointStatus.SAVED:
            checkpoint_step_at_end = latest.checkpoint_step
        self._execution_attempts().close_attempt(
            self._active_attempt_id,
            status=status,
            checkpoint_step_at_end=checkpoint_step_at_end,
            error_type=exc_type.__name__ if exc_type is not None else None,
            error_message=str(exc_val) if exc_val is not None else None,
        )
        self._execution_attempts().save()
        self._active_attempt_id = None

    def _register_saved_checkpoint(
        self,
        *,
        step: int,
        metrics: dict | None,
        global_step: int | None,
        optimizer_updates: int | None,
    ) -> None:
        snapshot = dict(metrics) if metrics is not None else self.latest_metrics()
        origin = None
        if self._meta.parent_id and self._parent_ckpt_step is not None and self._checkpoints().list() == []:
            origin = CheckpointOrigin(
                node_id=self._meta.parent_id,
                checkpoint_step=self._parent_ckpt_step,
            )
        record = CheckpointRecord.for_step(
            node_step=step,
            checkpoint_step=step,
            metrics={key: float(value) for key, value in snapshot.items()},
            global_step=global_step,
            optimizer_updates=optimizer_updates,
            origin=origin,
            status=CheckpointStatus.SAVED,
        )
        registry = self._checkpoints()
        registry.register(record)
        registry.set_alias_latest(step)
        registry.save()
        self._save_metadata()

    def _ensure_local_checkpoint_manager(self) -> ocp.CheckpointManager | None:
        """Initialize a checkpoint manager when local storage already exists."""
        if self._ckpt_manager is not None:
            return self._ckpt_manager
        if not self.checkpoint_dir.exists():
            return None
        if not any(self.checkpoint_dir.iterdir()):
            return None
        return self.init_checkpoint_manager(max_to_keep=None, save_interval_steps=1)

    def _finalize_checkpoint_aliases(self, run_succeeded: bool) -> None:
        if not run_succeeded:
            return
        registry = self._checkpoint_registry
        if registry is None and not self.checkpoints_registry_path.exists():
            return
        active_registry = self._checkpoints()
        latest = active_registry.get_alias_step(CHECKPOINT_ALIAS_LATEST)
        if latest is not None:
            active_registry.set_alias_final(latest)
            active_registry.save()

    def _save_metadata(self) -> Path:
        """Persist node metadata to node.json."""
        return self._meta.save(self.metadata_path)
