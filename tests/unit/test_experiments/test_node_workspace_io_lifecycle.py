"""Integration tests for NodeWorkspace IO across the training lifecycle."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from jarl.experiments.io.artifacts import ARTIFACT_KIND_VIDEO, ARTIFACT_KIND_VIDEO_METRICS, ArtifactRecord
from jarl.experiments.io.layout import VIDEO_METRICS_JSONL_FILENAME
from jarl.experiments.io.metrics import JsonlMetricReader
from jarl.experiments.node import NodeMetadata, NodeWorkspace
from jarl.experiments.run_config import TrackingConfig


class TestNodeWorkspaceIoLifecycle:
    """End-to-end checks for JSONL metrics, artifacts, and JIT callbacks."""

    def test_training_context_writes_jsonl_registers_artifacts_and_jit_callback(
        self,
        tmp_path: Path,
    ) -> None:
        workspace = NodeWorkspace(
            node_dir=tmp_path / "node",
            node_metadata=NodeMetadata(id="main_rollout_ab12cd34", branch="main"),
            tracking=TrackingConfig(track_tensorboard=False, track_wandb=False),
        )
        host_log = workspace.make_scalar_metric_log_callback()

        with workspace:

            @jax.jit
            def train_step(step: jax.Array, loss: jax.Array) -> jax.Array:
                jax.debug.callback(host_log, step, "loss", loss)
                return step + 1

            result = train_step(jnp.int32(0), jnp.float32(0.42))

            workspace.register_artifact(ArtifactRecord.video("policy", step=int(result)))
            workspace.register_artifact(ArtifactRecord.video_metrics(step=int(result)))

        assert int(result) == 1
        assert JsonlMetricReader(workspace.metrics_jsonl_path).latest() == {"loss": pytest.approx(0.42)}
        assert not workspace.tensorboard_dir.exists()
        assert not workspace.wandb_dir.exists()
        assert workspace.artifacts_registry_path.exists()
        assert len(workspace.list_artifacts()) == 2
        assert workspace.get_artifact(ARTIFACT_KIND_VIDEO, "policy", 1) is not None
        assert (
            workspace.get_artifact(
                ARTIFACT_KIND_VIDEO_METRICS,
                VIDEO_METRICS_JSONL_FILENAME,
                1,
            )
            is not None
        )
