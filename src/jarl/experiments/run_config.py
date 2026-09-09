"""Default run configuration for jarl experiments."""

from __future__ import annotations

import os
import time
from typing import Annotated, ClassVar, Self

from pydantic import Field, model_validator

from jarl.config import BaseConfig

__all__ = [
    "CheckpointConfig",
    "JaxConfig",
    "LoggingConfig",
    "RunConfig",
    "TrackingConfig",
]


class CheckpointConfig(BaseConfig):
    """Checkpoint management settings."""

    max_to_keep: Annotated[int | None, Field(description="Maximum checkpoints to retain (`None` for unlimited).")] = (
        None
    )
    save_interval_steps: Annotated[
        int,
        Field(description="Save a checkpoint every N training steps.", ge=1),
    ] = 1
    save_interval_seconds: Annotated[
        float | None,
        Field(description="Save a checkpoint every N seconds of elapsed training time."),
    ] = None
    save_at_progress_fractions: Annotated[
        list[float],
        Field(description="Save checkpoints when progress crosses these fractions in ``[0, 1]``."),
    ] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_progress_fractions(self) -> Self:
        """Ensure progress fractions stay within ``[0, 1]``."""
        for fraction in self.save_at_progress_fractions:
            if not 0.0 <= fraction <= 1.0:
                msg = f"Progress fraction must be in [0, 1], got {fraction}"
                raise ValueError(msg)
        return self


class JaxConfig(BaseConfig):
    """JAX runtime settings.

    Values default to environment variables where available, falling back to
    sensible defaults. See https://docs.jax.dev/en/latest/config_options.html.
    """

    compilation_cache_dir: Annotated[
        str,
        Field(description="Directory for JAX compilation cache."),
    ] = os.environ.get("JAX_COMPILATION_CACHE_DIR", "/tmp/jax_cache")  # noqa: S108
    default_matmul_precision: Annotated[
        str,
        Field(description="Precision for 32-bit matmul and conv operations."),
    ] = os.environ.get("JAX_DEFAULT_MATMUL_PRECISION", "bfloat16")
    # - Control the default matmul and conv precision for 32bit inputs.
    #   The levels roughly describe the precision at which scalar products are computed.
    #   - The 'bfloat16' option is the fastest and least precise
    #   - The 'float32' option is similar to full float32 precision
    #   - The 'tensorfloat32' option is intermediate
    exec_time_optimization_effort: Annotated[
        float,
        Field(description="Effort for minimizing execution time ([-1.0, 1.0])."),
    ] = float(os.environ.get("JAX_EXEC_TIME_OPTIMIZATION_EFFORT", "0.0"))
    memory_fitting_effort: Annotated[
        float,
        Field(description="Effort for minimizing memory usage ([-1.0, 1.0])."),
    ] = float(os.environ.get("JAX_MEMORY_FITTING_EFFORT", "1.0"))


class TrackingConfig(BaseConfig):
    """Experiment tracking settings."""

    track_console: Annotated[
        bool,
        Field(description="Whether to log metrics to the console."),
    ] = False
    track_tensorboard: Annotated[
        bool,
        Field(description="Whether to write TensorBoard event files."),
    ] = False
    track_wandb: Annotated[
        bool,
        Field(description="Whether to log metrics to Weights & Biases."),
    ] = False
    wandb_project: Annotated[
        str,
        Field(description="W&B project name (defaults to ``jarl`` when empty)."),
    ] = ""
    wandb_entity: Annotated[
        str | None,
        Field(description="W&B entity (team or user)."),
    ] = None
    wandb_group: Annotated[
        str,
        Field(description="W&B run group."),
    ] = ""
    wandb_mode: Annotated[
        str | None,
        Field(
            description=(
                "W&B mode: ``offline``, ``online``, or ``disabled``. "
                "When unset, uses ``WANDB_MODE`` from the environment (default ``offline``)."
            ),
        ),
    ] = None
    wandb_tags: Annotated[
        list[str],
        Field(description="Tags attached to the W&B run."),
    ] = Field(default_factory=list)


class LoggingConfig(BaseConfig):
    """Logging settings."""

    logger_name: Annotated[
        str,
        Field(description="Name of the Python logger to use."),
    ] = "jarl"
    lockfile: Annotated[
        str | None,
        Field(description="Path to a dependency lockfile for reproducibility."),
    ] = os.environ.get("JARL_ENV_LOCK_FILE")


class RunConfig(BaseConfig):
    """Top-level experiment run configuration.

    Composes topic-specific nested configs for checkpointing, JAX, tracking,
    and logging. Experiment-specific configs should subclass this and add
    their own fields.

    Example:
        ```python
        class MNISTConfig(RunConfig):
            DEFAULT_PROJECT_NAME = "learning"
            DEFAULT_EXP_NAME = "mnist"
            learning_rate: float = 0.1

        config = MNISTConfig()  # project/exp filled from class defaults
        ```
    """

    DEFAULT_PROJECT_NAME: ClassVar[str] = "unknown_project"
    DEFAULT_EXP_NAME: ClassVar[str] = "unknown_experiment"

    IMMUTABLE_PATHS: ClassVar[frozenset[str]] = frozenset({"project_name", "exp_name"})

    project_name: Annotated[
        str,
        Field(description="Project name grouping related experiments."),
    ] = ""
    exp_name: Annotated[
        str,
        Field(description="Experiment name within the project."),
    ] = ""
    run_name: Annotated[
        str,
        Field(description="Unique run identifier (auto-generated when empty)."),
    ] = ""
    notes: Annotated[
        str,
        Field(description="Free-form notes about this run."),
    ] = ""
    checkpoint: Annotated[
        CheckpointConfig,
        Field(description="Checkpoint management settings."),
    ] = CheckpointConfig()
    jax: Annotated[
        JaxConfig,
        Field(description="JAX runtime settings."),
    ] = JaxConfig()
    tracking: Annotated[
        TrackingConfig,
        Field(description="Experiment tracking settings."),
    ] = TrackingConfig()
    logging: Annotated[
        LoggingConfig,
        Field(description="Logging settings."),
    ] = LoggingConfig()

    @model_validator(mode="after")
    def _ensure_defaults(self) -> Self:
        """Apply class-level defaults and generate a run name when missing."""
        cls = type(self)
        if not self.project_name:
            object.__setattr__(self, "project_name", cls.DEFAULT_PROJECT_NAME)
        if not self.exp_name:
            object.__setattr__(self, "exp_name", cls.DEFAULT_EXP_NAME)
        if not self.run_name:
            ts = time.strftime("%Y_%m_%d_%H_%M_%S_", time.localtime()) + f"{int((time.time() % 1) * 1e6):06d}"
            object.__setattr__(self, "run_name", ts)
        return self
