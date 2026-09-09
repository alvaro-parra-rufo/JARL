"""RL training orchestration for jarl experiments.

Public entrypoints for config, schedule derivation, and the global runner.
Typical flow:

    experiment graph → target node → resolved ``RLRunConfig`` →
    ``apply_training_schedule`` → ``run_training(trainer=...)``

Trainers implement the ``Trainer`` protocol; the runner persists schedule
fields on ``config.json``, opens ``NodeWorkspace``, and saves the graph.
CLI flags map to nested config overrides via ``python -m jarl.training.run``.
"""

from jarl.training.cli import build_config_overrides, main, parse_args, resolve_trainer
from jarl.training.config import (
    AlgorithmConfig,
    EnvironmentConfig,
    RewardWeightsConfig,
    RLRunConfig,
    RLRunnerConfig,
    VideoConfig,
)
from jarl.training.env_factory import navix_full_jit_env_factory
from jarl.training.override_models import (
    ForkConfigOverrides,
    RetrainConfigOverrides,
    SparseConfigOverrides,
)
from jarl.training.registry import list_trainer_names, resolve_trainer_from_name
from jarl.training.runner import RunTrainingResult, resume_training, run_training
from jarl.training.schedule import (
    TrainingSchedule,
    apply_training_schedule,
    compute_training_schedule,
    format_training_schedule_message,
)
from jarl.training.trainer import EnvFactory, ScheduleBuilder, Trainer

__all__ = [
    "AlgorithmConfig",
    "EnvFactory",
    "EnvironmentConfig",
    "ForkConfigOverrides",
    "RLRunConfig",
    "RLRunnerConfig",
    "RetrainConfigOverrides",
    "RewardWeightsConfig",
    "RunTrainingResult",
    "ScheduleBuilder",
    "SparseConfigOverrides",
    "Trainer",
    "TrainingSchedule",
    "VideoConfig",
    "apply_training_schedule",
    "build_config_overrides",
    "compute_training_schedule",
    "format_training_schedule_message",
    "list_trainer_names",
    "main",
    "navix_full_jit_env_factory",
    "parse_args",
    "resolve_trainer",
    "resolve_trainer_from_name",
    "resume_training",
    "run_training",
]
