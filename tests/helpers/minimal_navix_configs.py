"""Tiny Navix full-JAX configs for slow smoke and CLI integration tests."""

from __future__ import annotations

import sys
from pathlib import Path

from jarl.training.config import AlgorithmConfig, EnvironmentConfig, RLRunConfig, VideoConfig

__all__ = [
    "MINIMAL_NAVIX_ENV_ID",
    "minimal_cli_argv_ppo",
    "minimal_cli_argv_ppo_gru",
    "minimal_ppo_gru_run_config",
    "minimal_ppo_run_config",
]

MINIMAL_NAVIX_ENV_ID = "Navix-Empty-5x5-v0"
"""Small Navix map used across runner smoke tests."""

_MINIMAL_ENV = EnvironmentConfig(
    env_id=MINIMAL_NAVIX_ENV_ID,
    nr_envs=4,
    seed=0,
    max_episode_steps=16,
)

_MINIMAL_ALGO = AlgorithmConfig(
    total_timesteps=512,
    nr_steps=8,
    minibatch_size=32,
    nr_epochs=1,
    evaluation_and_save_frequency=128,
    evaluation_active=True,
)

_MINIMAL_VIDEO = VideoConfig(
    record_video=False,
    record_final_video=False,
)


def minimal_ppo_run_config() -> RLRunConfig:
    """Minimal PPO full-JAX config for short compiled training runs."""
    return RLRunConfig(
        environment=_MINIMAL_ENV.model_copy(),
        algorithm=_MINIMAL_ALGO.model_copy(),
        video=_MINIMAL_VIDEO.model_copy(),
    )


def minimal_ppo_gru_run_config() -> RLRunConfig:
    """Minimal PPO-GRU full-JAX config for short compiled training runs."""
    return RLRunConfig(
        environment=_MINIMAL_ENV.model_copy(),
        algorithm=_MINIMAL_ALGO.model_copy(
            update={
                "name": "ppo_gru.full_jax.navix",
                "obs_encoding_dim": 16,
                "gru_hidden_dim": 8,
            }
        ),
        video=_MINIMAL_VIDEO.model_copy(),
    )


def _base_cli_argv(exp_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "jarl.training.run",
        "--experiment-dir",
        str(exp_dir),
        "--create-root",
        "--env-id",
        MINIMAL_NAVIX_ENV_ID,
        "--total-timesteps",
        "512",
        "--nr-envs",
        "4",
        "--nr-steps",
        "8",
        "--minibatch-size",
        "32",
        "--eval-frequency",
        "128",
        "--no-wandb",
    ]


def minimal_cli_argv_ppo(exp_dir: Path) -> list[str]:
    """CLI argv for a minimal PPO run resolved from the default registry name."""
    return _base_cli_argv(exp_dir)


def minimal_cli_argv_ppo_gru(exp_dir: Path) -> list[str]:
    """CLI argv for a minimal PPO-GRU run resolved from ``ppo_gru.full_jax.navix``."""
    return [
        *_base_cli_argv(exp_dir),
        "--algorithm-name",
        "ppo_gru.full_jax.navix",
        "--obs-encoding-dim",
        "16",
        "--gru-hidden-dim",
        "8",
    ]
