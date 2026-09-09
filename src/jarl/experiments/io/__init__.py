"""IO utilities for experiment node workspaces."""

from typing import TYPE_CHECKING

from jarl.experiments.io.artifacts import (
    ARTIFACT_KIND_MODEL,
    ARTIFACT_KIND_ROLLOUT,
    ARTIFACT_KIND_VIDEO,
    ARTIFACT_KIND_VIDEO_METRICS,
    MODEL_ALIAS_BEST,
    MODEL_ALIAS_FINAL,
    MODEL_ALIAS_LATEST,
    ArtifactIdentity,
    ArtifactRecord,
    ArtifactRegistry,
    BestModelArtifactAlias,
    ModelArtifactAlias,
)
from jarl.experiments.io.checkpoint_policy import CheckpointPolicyState, should_save_checkpoint
from jarl.experiments.io.checkpoints import (
    CHECKPOINT_ALIAS_BEST,
    CHECKPOINT_ALIAS_FINAL,
    CHECKPOINT_ALIAS_LATEST,
    CHECKPOINT_BEST_LENGTH_METRIC,
    CHECKPOINT_BEST_RETURN_METRIC,
    BestCheckpointAlias,
    CheckpointOrigin,
    CheckpointRecord,
    CheckpointRef,
    CheckpointRegistry,
    CheckpointStatus,
    select_best_checkpoint,
)
from jarl.experiments.io.config import save_resolved_config
from jarl.experiments.io.execution_attempts import (
    ExecutionAttemptRecord,
    ExecutionAttemptRegistry,
    ExecutionAttemptStatus,
)
from jarl.experiments.io.jit_metrics import MetricLogFn, ScalarMetricLogPayload, make_scalar_metric_log_callback
from jarl.experiments.io.layout import (
    NODES_DIRNAME,
    ROLLOUTS_DIRNAME,
    RUN_METADATA_FILENAME,
    ExperimentLayout,
    NodeLayout,
)
from jarl.experiments.io.metric_writer import NodeMetricWriter
from jarl.experiments.io.metrics import JsonlMetricReader, JsonlMetricWriter, MetricRecord
from jarl.experiments.io.model_archive import (
    MODEL_ARCHIVE_FORMAT_VERSION,
    MODEL_MANIFEST_FILENAME,
    build_model_artifact_record,
    canonical_model_filename,
    load_model_archive,
    make_model_archive_save_callback,
    save_model_archive,
)

if TYPE_CHECKING:
    from jarl.experiments.io.rollouts import (
        ROLLOUT_MANIFEST_FILENAME,
        ROLLOUT_SCHEMA_VERSION,
        ROLLOUT_TRACE_FILENAME,
        ROLLOUT_VIDEO_FILENAME,
        LoadedRolloutArtifact,
        RolloutArtifactFiles,
        RolloutArtifactManifest,
        RolloutArtifactRef,
        TraceArrayMetadata,
        WriteRolloutArtifactResult,
        compute_rollout_id,
        load_cached_rollout_artifact,
        load_rollout_artifact,
        write_rollout_artifact,
    )

__all__ = [
    "ARTIFACT_KIND_MODEL",
    "ARTIFACT_KIND_ROLLOUT",
    "ARTIFACT_KIND_VIDEO",
    "ARTIFACT_KIND_VIDEO_METRICS",
    "CHECKPOINT_ALIAS_BEST",
    "CHECKPOINT_ALIAS_FINAL",
    "CHECKPOINT_ALIAS_LATEST",
    "CHECKPOINT_BEST_LENGTH_METRIC",
    "CHECKPOINT_BEST_RETURN_METRIC",
    "MODEL_ALIAS_BEST",
    "MODEL_ALIAS_FINAL",
    "MODEL_ALIAS_LATEST",
    "MODEL_ARCHIVE_FORMAT_VERSION",
    "MODEL_MANIFEST_FILENAME",
    "NODES_DIRNAME",
    "ROLLOUTS_DIRNAME",
    "ROLLOUT_MANIFEST_FILENAME",
    "ROLLOUT_SCHEMA_VERSION",
    "ROLLOUT_TRACE_FILENAME",
    "ROLLOUT_VIDEO_FILENAME",
    "RUN_METADATA_FILENAME",
    "ArtifactIdentity",
    "ArtifactRecord",
    "ArtifactRegistry",
    "BestCheckpointAlias",
    "BestModelArtifactAlias",
    "CheckpointOrigin",
    "CheckpointPolicyState",
    "CheckpointRecord",
    "CheckpointRef",
    "CheckpointRegistry",
    "CheckpointStatus",
    "ExecutionAttemptRecord",
    "ExecutionAttemptRegistry",
    "ExecutionAttemptStatus",
    "ExperimentLayout",
    "JsonlMetricReader",
    "JsonlMetricWriter",
    "LoadedRolloutArtifact",
    "MetricLogFn",
    "MetricRecord",
    "ModelArtifactAlias",
    "NodeLayout",
    "NodeMetricWriter",
    "RolloutArtifactFiles",
    "RolloutArtifactManifest",
    "RolloutArtifactRef",
    "ScalarMetricLogPayload",
    "TraceArrayMetadata",
    "WriteRolloutArtifactResult",
    "build_model_artifact_record",
    "canonical_model_filename",
    "compute_rollout_id",
    "load_cached_rollout_artifact",
    "load_model_archive",
    "load_rollout_artifact",
    "make_model_archive_save_callback",
    "make_scalar_metric_log_callback",
    "save_model_archive",
    "save_resolved_config",
    "select_best_checkpoint",
    "should_save_checkpoint",
    "write_rollout_artifact",
]


def __getattr__(name: str) -> object:
    """Lazily expose rollout IO without coupling generic workspace imports."""
    if name not in {
        "ROLLOUT_MANIFEST_FILENAME",
        "ROLLOUT_SCHEMA_VERSION",
        "ROLLOUT_TRACE_FILENAME",
        "ROLLOUT_VIDEO_FILENAME",
        "LoadedRolloutArtifact",
        "RolloutArtifactFiles",
        "RolloutArtifactManifest",
        "RolloutArtifactRef",
        "TraceArrayMetadata",
        "WriteRolloutArtifactResult",
        "compute_rollout_id",
        "load_cached_rollout_artifact",
        "load_rollout_artifact",
        "write_rollout_artifact",
    }:
        raise AttributeError(name)
    from jarl.experiments.io import rollouts

    return getattr(rollouts, name)
