"""Optional Weights & Biases tracking for experiment node workspaces."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jarl.experiments.io.layout import NodeLayout
    from jarl.experiments.node import NodeMetadata
    from jarl.experiments.run_config import TrackingConfig
    from jarl.training.config import RLRunConfig
    from jarl.training.schedule import TrainingSchedule

__all__ = [
    "DEFAULT_WANDB_PROJECT",
    "WandbRunTracker",
    "build_wandb_run_config",
    "configure_wandb_metrics",
    "log_wandb_scalar",
]

DEFAULT_WANDB_PROJECT = "jarl"
LOGGER = logging.getLogger(__name__)


def build_wandb_run_config(
    *,
    node_metadata: NodeMetadata,
    node_layout: NodeLayout,
    config: RLRunConfig,
    schedule: TrainingSchedule,
) -> dict[str, Any]:
    """Build a compact W&B config payload for a training node run.

    Args:
        node_metadata: Node lineage metadata for the active workspace.
        node_layout: Layout helper exposing canonical artifact paths.
        config: Fully resolved RL run config for the attempt.
        schedule: Derived training schedule counters.

    Returns:
        JSON-serializable dict passed to ``wandb.init(config=...)``.
    """
    algorithm = config.algorithm
    environment = config.environment
    return {
        "node_id": node_metadata.id,
        "branch": node_metadata.branch,
        "parent_id": node_metadata.parent_id,
        "label": node_metadata.label,
        "algorithm_name": algorithm.name,
        "env_id": environment.env_id,
        "seed": environment.seed,
        "nr_envs": environment.nr_envs,
        "max_episode_steps": environment.max_episode_steps,
        "total_timesteps": algorithm.total_timesteps,
        "requested_total_timesteps": schedule.requested_total_timesteps,
        "actual_total_timesteps": schedule.actual_total_timesteps,
        "actual_rollout_updates": schedule.actual_rollout_updates,
        "actual_optimizer_updates": schedule.actual_optimizer_updates,
        "batch_size": schedule.batch_size,
        "nr_minibatches": schedule.nr_minibatches,
        "effective_evaluation_and_save_frequency": schedule.effective_evaluation_and_save_frequency,
        "nr_steps": algorithm.nr_steps,
        "nr_epochs": algorithm.nr_epochs,
        "minibatch_size": algorithm.minibatch_size,
        "learning_rate": algorithm.learning_rate,
        "gamma": algorithm.gamma,
        "evaluation_active": algorithm.evaluation_active,
        "obs_encoding_dim": algorithm.obs_encoding_dim,
        "gru_hidden_dim": algorithm.gru_hidden_dim,
        "config_path": str(node_layout.config_path),
        "metrics_path": str(node_layout.metrics_jsonl_path),
        "checkpoint_dir": str(node_layout.checkpoint_dir),
        "models_dir": str(node_layout.models_dir),
        "tensorboard_dir": str(node_layout.tensorboard_dir),
        "wandb_dir": str(node_layout.wandb_dir),
    }


def configure_wandb_metrics() -> None:
    """Configure W&B charts to use training steps as the x-axis."""
    import wandb

    if wandb.run is None:
        return
    wandb.define_metric("global_step")
    wandb.define_metric("*", step_metric="global_step")


def log_wandb_scalar(step: int, name: str, value: float) -> None:
    """Log one scalar metric to the active W&B run when available.

    Args:
        step: Training step associated with the measurement.
        name: Metric identifier.
        value: Scalar metric value.
    """
    import wandb

    if wandb.run is None:
        return
    wandb.log({name: value, "global_step": int(step)}, step=int(step))


@dataclass
class WandbRunTracker:
    """Manage W&B init, scalar logging, and finish for one node run.

    Args:
        tracking: Tracking settings controlling W&B behavior.
        wandb_dir: Directory where W&B stores local run data.
        run_name: Display name for the W&B run.
        config: Optional config payload attached to the run.
    """

    tracking: TrackingConfig
    wandb_dir: Path
    run_name: str
    config: dict[str, Any] | None = None
    _initialized: bool = field(default=False, init=False, repr=False)
    _disabled: bool = field(default=False, init=False, repr=False)

    @property
    def active(self) -> bool:
        """Return whether W&B tracking is initialized and still enabled."""
        return self._initialized and not self._disabled

    def init(self) -> None:
        """Initialize a W&B run when tracking is enabled.

        Failures are logged and tracking is disabled for the remainder of the
        run without raising.
        """
        if not self.tracking.track_wandb or self._disabled:
            return

        import wandb

        self.wandb_dir.mkdir(parents=True, exist_ok=True)
        os.environ["WANDB_CACHE_DIR"] = str(self.wandb_dir / "cache")
        os.environ["WANDB_DATA_DIR"] = str(self.wandb_dir / "data")
        os.environ["WANDB_CONFIG_DIR"] = str(self.wandb_dir / "config")

        project = self.tracking.wandb_project or DEFAULT_WANDB_PROJECT
        mode = self.tracking.wandb_mode or os.environ.get("WANDB_MODE", "offline")
        if mode == "online" and not os.environ.get("WANDB_API_KEY"):
            LOGGER.warning("W&B is configured in online mode, but WANDB_API_KEY is not set.")

        init_kwargs: dict[str, Any] = {
            "project": project,
            "name": self.run_name,
            "dir": str(self.wandb_dir),
            "mode": mode,
            "config": self.config or {},
        }
        if self.tracking.wandb_group:
            init_kwargs["group"] = self.tracking.wandb_group
        if self.tracking.wandb_entity:
            init_kwargs["entity"] = self.tracking.wandb_entity
        if self.tracking.wandb_tags:
            init_kwargs["tags"] = list(self.tracking.wandb_tags)

        try:
            wandb.init(**init_kwargs)
            configure_wandb_metrics()
            self._initialized = True
        except Exception as exc:
            LOGGER.warning("Disabling W&B for this run because initialization failed: %s", exc)
            self._disabled = True

    def log_scalar(self, step: int, name: str, value: float) -> None:
        """Log one scalar metric when W&B tracking is active.

        Args:
            step: Training step associated with the measurement.
            name: Metric identifier.
            value: Scalar metric value.
        """
        if not self.active:
            return
        log_wandb_scalar(step, name, value)

    def finish(self) -> None:
        """Finish the active W&B run if one was initialized."""
        if not self._initialized:
            return

        import wandb

        if wandb.run is not None:
            wandb.finish()
        self._initialized = False
