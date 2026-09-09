"""Smoke tests for PPO full-JAX trainer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarl.agents.ppo.metric_schema import PPO_EVAL_METRIC_SCHEMA, PPO_TRAIN_METRIC_SCHEMA
from jarl.experiments.io.metrics import JsonlMetricReader
from jarl.experiments.node import NodeStatus
from jarl.training.config import RLRunConfig
from tests.helpers.jax_training_subprocess import run_create_and_train_in_subprocess
from tests.helpers.minimal_navix_configs import minimal_ppo_run_config


@pytest.fixture()
def tiny_ppo_config() -> RLRunConfig:
    """Minimal Navix PPO config for a short compiled training run."""
    return minimal_ppo_run_config()


@pytest.mark.slow
class TestPPOFullJaxTrainer:
    """Minimal real PPO smoke run through the training CLI subprocess."""

    def test_ppo_trainer_logs_metrics_and_writes_checkpoints(
        self,
        tmp_path: Path,
        tiny_ppo_config: RLRunConfig,
    ) -> None:
        workspace = run_create_and_train_in_subprocess(tmp_path / "exp", tiny_ppo_config)

        metrics = list(JsonlMetricReader(workspace.metrics_jsonl_path).iter_records())
        metric_names = {record.name for record in metrics}
        saved_config = json.loads(workspace.config_path.read_text(encoding="utf-8"))
        registry = json.loads(workspace.checkpoints_registry_path.read_text(encoding="utf-8"))

        assert workspace.status == NodeStatus.COMPLETED
        assert saved_config["algorithm"]["actual_total_timesteps"] == 512
        assert PPO_TRAIN_METRIC_SCHEMA.all_names[0] in metric_names
        assert PPO_EVAL_METRIC_SCHEMA.all_names[0] in metric_names
        assert registry["checkpoints"]
        assert all(record["status"] == "saved" for record in registry["checkpoints"])
