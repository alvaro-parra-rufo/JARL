"""Tests for forwarding resolved JAX settings to environment setup."""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from jarl.experiments.run_config import JaxConfig
from jarl.training.config import (
    AlgorithmConfig,
    EnvironmentConfig,
    RLRunConfig,
    VideoConfig,
)
from jarl.training.runner import run_training


@pytest.fixture()
def minimal_config(tmp_path: Path) -> RLRunConfig:
    """Return a minimal config suitable for runner JAX tests."""
    return RLRunConfig(
        jax=JaxConfig(
            default_matmul_precision="float32",
            exec_time_optimization_effort=0.5,
            memory_fitting_effort=0.25,
            compilation_cache_dir=str(tmp_path / "jax_cache"),
        ),
        environment=EnvironmentConfig(nr_envs=1, seed=7),
        algorithm=AlgorithmConfig(
            total_timesteps=1,
            nr_steps=1,
            nr_epochs=1,
            minibatch_size=1,
            evaluation_and_save_frequency=-1,
            evaluation_active=False,
        ),
        video=VideoConfig(record_video=False, record_final_video=False),
    )


class TestRunnerJaxConfig:
    """Specifications for applying ``JaxConfig`` before device discovery."""

    def test_runner_passes_resolved_jax_config(
        self,
        tmp_path: Path,
        minimal_config: RLRunConfig,
        mocker: MockerFixture,
    ) -> None:
        trainer = mocker.Mock()
        setup_jax = mocker.patch("jarl.training.runner.setup_jax")

        result = run_training(
            trainer=trainer,
            experiment_dir=tmp_path / "experiment",
            config=minimal_config,
            create_root=True,
        )

        setup_jax.assert_called_once_with(result.config.jax)
