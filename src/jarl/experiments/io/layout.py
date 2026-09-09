"""On-disk layout conventions for a single experiment/node.

For experiments, the root directory follows:

| Path | Purpose |
|------|---------|
| ``config.json`` | Immutable base config snapshot |
| ``run_metadata.json`` | Experiment-level run metadata |
| ``case.json`` | Optional experiment-case metadata snapshot |
| ``manifest.json`` | Experiment tree manifest |
| ``nodes/`` | Directory containing node workspaces |

Each node directory follows:

| Path | Purpose |
|------|---------|
| ``node.json`` | Node metadata and lifecycle state |
| ``config.json`` | Fully resolved training config snapshot for this node |
| ``config_overrides.json`` | Sparse config delta from the parent node |
| ``metrics.jsonl`` | Primary incremental scalar metrics log |
| ``artifacts.json`` | Exportable artifact registry |
| ``train.log`` | Per-node training log file |
| ``checkpoint/`` | Orbax checkpoint storage |
| ``checkpoints.json`` | Typed checkpoint metadata index |
| ``execution_attempts.json`` | Training execution attempt registry |
| ``tensorboard/`` | TensorBoard event files |
| ``wandb/`` | Weights & Biases run data |
| ``models/`` | Saved model archives |
| ``rollouts/`` | Materialized checkpoint rollout traces |
| ``videos/`` | Rendered rollout videos |
| ``video_metrics.jsonl`` | Metrics associated with recorded videos |
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jarl.experiments.manifest import MANIFEST_FILENAME

RUN_METADATA_FILENAME = "run_metadata.json"
"""Experiment-level run metadata filename."""

CASE_METADATA_FILENAME = "case.json"
"""Experiment-level case metadata filename."""

NODES_DIRNAME = "nodes"
"""Directory name for per-node workspaces under an experiment root."""

ARTIFACTS_FILENAME = "artifacts.json"
"""Artifact registry filename for a node."""

CONFIG_OVERRIDES_FILENAME = "config_overrides.json"
"""Sparse config override filename for a node."""

CONFIG_FILENAME = "config.json"
"""Resolved config snapshot filename for a node."""

METRICS_JSONL_FILENAME = "metrics.jsonl"
"""Primary incremental metrics log filename for a node."""

NODE_METADATA_FILENAME = "node.json"
"""Node metadata filename."""

TRAIN_LOG_FILENAME = "train.log"
"""Training log filename for a node."""

VIDEO_METRICS_JSONL_FILENAME = "video_metrics.jsonl"
"""Video-associated metrics log filename for a node."""

CHECKPOINT_DIRNAME = "checkpoint"
"""Orbax checkpoint storage directory name for a node."""

CHECKPOINTS_INDEX_FILENAME = "checkpoints.json"
"""Checkpoint metadata index filename for a node."""

EXECUTION_ATTEMPTS_FILENAME = "execution_attempts.json"
"""Execution attempt registry filename for a node."""

MODELS_DIRNAME = "models"
"""Saved model archives directory name for a node."""

ROLLOUTS_DIRNAME = "rollouts"
"""Materialized checkpoint rollout directory name for a node."""

TENSORBOARD_DIRNAME = "tensorboard"
"""TensorBoard event files directory name for a node."""

VIDEOS_DIRNAME = "videos"
"""Rendered rollout videos directory name for a node."""

WANDB_DIRNAME = "wandb"
"""Weights & Biases run data directory name for a node."""


@dataclass(frozen=True, slots=True)
class ExperimentLayout:
    """Path accessors for an experiment root directory.

    All properties return paths without creating files or directories.

    Args:
        root: Root directory of the experiment.
    """

    root: Path

    @property
    def config_path(self) -> Path:
        """Path to the immutable base config snapshot."""
        return self.root / CONFIG_FILENAME

    @property
    def run_metadata_path(self) -> Path:
        """Path to experiment-level run metadata."""
        return self.root / RUN_METADATA_FILENAME

    @property
    def case_metadata_path(self) -> Path:
        """Path to experiment-level case metadata."""
        return self.root / CASE_METADATA_FILENAME

    @property
    def manifest_path(self) -> Path:
        """Path to the experiment tree manifest."""
        return self.root / MANIFEST_FILENAME

    @property
    def nodes_dir(self) -> Path:
        """Path to the directory containing node workspaces."""
        return self.root / NODES_DIRNAME

    def node_dir(self, node_id: str) -> Path:
        """Path to a single node workspace directory."""
        return self.nodes_dir / node_id


@dataclass(frozen=True, slots=True)
class NodeLayout:
    """Path accessors for a node workspace directory.

    All properties return paths without creating files or directories.

    Args:
        root: Root directory of the node workspace.
    """

    root: Path

    @property
    def config_path(self) -> Path:
        """Path to the resolved config snapshot."""
        return self.root / CONFIG_FILENAME

    @property
    def config_overrides_path(self) -> Path:
        """Path to sparse config overrides."""
        return self.root / CONFIG_OVERRIDES_FILENAME

    @property
    def metadata_path(self) -> Path:
        """Path to node metadata."""
        return self.root / NODE_METADATA_FILENAME

    @property
    def metrics_jsonl_path(self) -> Path:
        """Path to the primary incremental metrics log."""
        return self.root / METRICS_JSONL_FILENAME

    @property
    def artifacts_registry_path(self) -> Path:
        """Path to the artifact registry file."""
        return self.root / ARTIFACTS_FILENAME

    @property
    def log_path(self) -> Path:
        """Path to the per-node training log."""
        return self.root / TRAIN_LOG_FILENAME

    @property
    def video_metrics_jsonl_path(self) -> Path:
        """Path to video-associated metrics log."""
        return self.root / VIDEO_METRICS_JSONL_FILENAME

    @property
    def checkpoint_dir(self) -> Path:
        """Path to Orbax checkpoint storage."""
        return self.root / CHECKPOINT_DIRNAME

    @property
    def checkpoints_registry_path(self) -> Path:
        """Path to the checkpoint metadata index."""
        return self.root / CHECKPOINTS_INDEX_FILENAME

    @property
    def execution_attempts_path(self) -> Path:
        """Path to the execution attempt registry."""
        return self.root / EXECUTION_ATTEMPTS_FILENAME

    @property
    def tensorboard_dir(self) -> Path:
        """Path to TensorBoard event files."""
        return self.root / TENSORBOARD_DIRNAME

    @property
    def wandb_dir(self) -> Path:
        """Path to Weights & Biases run data."""
        return self.root / WANDB_DIRNAME

    @property
    def models_dir(self) -> Path:
        """Path to saved model archives."""
        return self.root / MODELS_DIRNAME

    @property
    def rollouts_dir(self) -> Path:
        """Path to materialized checkpoint rollouts."""
        return self.root / ROLLOUTS_DIRNAME

    @property
    def videos_dir(self) -> Path:
        """Path to rendered rollout videos."""
        return self.root / VIDEOS_DIRNAME

    def rollout_dir(self, rollout_id: str) -> Path:
        """Path to one materialized checkpoint rollout."""
        return self.rollouts_dir / rollout_id
