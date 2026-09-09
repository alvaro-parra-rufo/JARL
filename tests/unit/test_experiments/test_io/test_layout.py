"""Tests for experiment and node path layouts."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarl.experiments.io.layout import (
    CASE_METADATA_FILENAME,
    CONFIG_FILENAME,
    RUN_METADATA_FILENAME,
    ExperimentLayout,
    NodeLayout,
)
from jarl.experiments.manifest import MANIFEST_FILENAME


@pytest.fixture()
def node_root(tmp_path: Path) -> Path:
    """Empty node root directory."""
    root = tmp_path / "nodes" / "main_baseline_ab12cd34"
    root.mkdir(parents=True)
    return root


@pytest.fixture()
def node_layout(node_root: Path) -> NodeLayout:
    """Layout for an empty node root."""
    return NodeLayout(node_root)


@pytest.fixture()
def experiment_root(tmp_path: Path) -> Path:
    """Empty experiment root directory."""
    root = tmp_path / "experiment"
    root.mkdir(parents=True)
    return root


@pytest.fixture()
def experiment_layout(experiment_root: Path) -> ExperimentLayout:
    """Layout for an empty experiment root."""
    return ExperimentLayout(experiment_root)


class TestExperimentLayoutPaths:
    """Tests for side-effect-free ``ExperimentLayout`` accessors."""

    @pytest.mark.parametrize(
        ("property_name", "expected_suffix"),
        [
            pytest.param("config_path", CONFIG_FILENAME, id="config"),
            pytest.param("run_metadata_path", RUN_METADATA_FILENAME, id="run_metadata"),
            pytest.param("case_metadata_path", CASE_METADATA_FILENAME, id="case_metadata"),
            pytest.param("manifest_path", MANIFEST_FILENAME, id="manifest"),
            pytest.param("nodes_dir", "nodes", id="nodes_dir"),
        ],
    )
    def test_path_access_does_not_create_entries(
        self,
        experiment_layout: ExperimentLayout,
        experiment_root: Path,
        property_name: str,
        expected_suffix: str,
    ) -> None:
        path = getattr(experiment_layout, property_name)

        assert path == experiment_root / expected_suffix
        assert not path.exists()

    def test_node_dir(self, experiment_layout: ExperimentLayout, experiment_root: Path) -> None:
        path = experiment_layout.node_dir("node_a")

        assert path == experiment_root / "nodes" / "node_a"
        assert not path.exists()


class TestNodeLayoutPaths:
    """Tests for side-effect-free ``NodeLayout`` accessors."""

    @pytest.mark.parametrize(
        ("property_name", "expected_suffix"),
        [
            pytest.param("config_path", "config.json", id="config"),
            pytest.param("config_overrides_path", "config_overrides.json", id="config_overrides"),
            pytest.param("metadata_path", "node.json", id="metadata"),
            pytest.param("metrics_jsonl_path", "metrics.jsonl", id="metrics_jsonl"),
            pytest.param("artifacts_registry_path", "artifacts.json", id="artifacts"),
            pytest.param("log_path", "train.log", id="log"),
            pytest.param("video_metrics_jsonl_path", "video_metrics.jsonl", id="video_metrics_jsonl"),
            pytest.param("checkpoint_dir", "checkpoint", id="checkpoint_dir"),
            pytest.param("checkpoints_registry_path", "checkpoints.json", id="checkpoints_registry"),
            pytest.param("execution_attempts_path", "execution_attempts.json", id="execution_attempts"),
            pytest.param("tensorboard_dir", "tensorboard", id="tensorboard_dir"),
            pytest.param("wandb_dir", "wandb", id="wandb_dir"),
            pytest.param("models_dir", "models", id="models_dir"),
            pytest.param("rollouts_dir", "rollouts", id="rollouts_dir"),
            pytest.param("videos_dir", "videos", id="videos_dir"),
        ],
    )
    def test_path_access_does_not_create_entries(
        self,
        node_layout: NodeLayout,
        node_root: Path,
        property_name: str,
        expected_suffix: str,
    ) -> None:
        path = getattr(node_layout, property_name)

        assert path == node_root / expected_suffix
        assert not path.exists()

    def test_rollout_dir(self, node_layout: NodeLayout, node_root: Path) -> None:
        path = node_layout.rollout_dir("abcd1234")

        assert path == node_root / "rollouts" / "abcd1234"
        assert not path.exists()
