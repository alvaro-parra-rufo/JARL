"""Build and validate ``RLRunConfig`` instances from training form payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from jarl.envs.navix.catalog import assert_transfer_compatible
from jarl.training.config import (
    AlgorithmConfig,
    EnvironmentConfig,
    RLRunConfig,
    RLRunnerConfig,
    VideoConfig,
    rollout_batch_divisibility_error,
)
from jarl.training.config import (
    rollout_batch_size as compute_rollout_batch_size,
)
from jarl.training.run_overrides import (
    ensure_validated_config_overrides,
    mutable_overrides_from_config,
    retrain_overrides_from_config,
)

MINIMAL_NAVIX_ENV_ID = "Navix-Empty-5x5-v0"
PPO_ALGORITHM_NAME = "ppo.full_jax.navix"
PPO_GRU_ALGORITHM_NAME = "ppo_gru.full_jax.navix"

# Default for new forms; the UI toggle can still turn it off.
DEFAULT_SAVE_MODEL = True

_LONG_RUN_MIN_TIMESTEPS = 100_000

PresetKind = Literal["fast", "custom"]
CustomBaseKind = Literal["simple", "medium", "advanced"]
AlgorithmChoice = Literal["ppo", "ppo_gru"]
DEFAULT_CUSTOM_BASE: CustomBaseKind = "advanced"

__all__ = [
    "DEFAULT_CUSTOM_BASE",
    "DEFAULT_SAVE_MODEL",
    "MINIMAL_NAVIX_ENV_ID",
    "PPO_ALGORITHM_NAME",
    "PPO_GRU_ALGORITHM_NAME",
    "AlgorithmChoice",
    "CustomBaseKind",
    "PresetKind",
    "RunFormPayload",
    "RunFormValues",
    "algorithm_choice_from_name",
    "child_config_overrides_from_payload",
    "custom_base_form_values",
    "default_form_values",
    "form_values_from_run_config",
    "form_values_to_run_config",
    "preset_run_config",
    "production_run_config",
    "recommended_video_frequency",
    "rollout_batch_size",
    "run_overrides_from_payload",
    "validate_run_form_values",
]


@dataclass(frozen=True, slots=True)
class RunFormValues:
    """User-facing training parameters collected from a form or agent payload."""

    algorithm: AlgorithmChoice
    preset: PresetKind
    env_id: str
    seed: int
    total_timesteps: int
    nr_envs: int
    nr_steps: int
    minibatch_size: int
    eval_frequency: int
    learning_rate: float
    obs_encoding_dim: int
    gru_hidden_dim: int
    track_wandb: bool
    track_tensorboard: bool
    wandb_project: str
    wandb_mode: str
    record_video: bool
    gamma: float
    gae_lambda: float
    clip_range: float
    entropy_coef: float
    nr_epochs: int
    save_model: bool
    evaluation_active: bool
    checkpoint_save_interval_steps: int
    wandb_entity: str
    wandb_group: str
    wandb_tags: str
    max_episode_steps: int | None = None


@dataclass(frozen=True, slots=True)
class RunFormPayload:
    """Form values plus video scheduling fields."""

    values: RunFormValues
    video_frequency: int
    record_final_video: bool
    reference_nr_envs: int | None = None


def production_run_config(*, algorithm: AlgorithmChoice = "ppo") -> RLRunConfig:
    """Production-scale defaults aligned with ``RLRunConfig`` model defaults."""
    algorithm_config = AlgorithmConfig(
        name=PPO_GRU_ALGORITHM_NAME if algorithm == "ppo_gru" else PPO_ALGORITHM_NAME,
    )
    return RLRunConfig(
        environment=EnvironmentConfig(),
        algorithm=algorithm_config,
        video=VideoConfig(record_video=False, record_final_video=True),
        runner=RLRunnerConfig(),
    )


def default_form_values(
    *,
    algorithm: AlgorithmChoice = "ppo",
    preset: PresetKind = "custom",
) -> RunFormValues:
    """Return default form values for production-scale training runs."""
    production = production_run_config(algorithm=algorithm)
    return _form_values_from_config(production, algorithm=algorithm, preset=preset)


def custom_base_form_values(
    *,
    algorithm: AlgorithmChoice = "ppo",
    base: CustomBaseKind = DEFAULT_CUSTOM_BASE,
) -> RunFormValues:
    """Return editable form values for a named custom base profile."""
    production = production_run_config(algorithm=algorithm)
    if base == "advanced":
        config = production
    elif base == "medium":
        config = production.apply_overrides(
            {
                "algorithm.total_timesteps": 1_000_000,
                "algorithm.nr_steps": 64,
                "algorithm.minibatch_size": 1024,
                "algorithm.nr_epochs": 3,
                "algorithm.evaluation_and_save_frequency": 32_768,
                "environment.nr_envs": 32,
                "checkpoint.save_interval_steps": 32_768,
            }
        )
    else:
        config = production.apply_overrides(
            {
                "environment.env_id": MINIMAL_NAVIX_ENV_ID,
                "environment.nr_envs": 8,
                "environment.seed": 0,
                "environment.max_episode_steps": 16,
                "algorithm.total_timesteps": 100_000,
                "algorithm.nr_steps": 32,
                "algorithm.minibatch_size": 128,
                "algorithm.nr_epochs": 2,
                "algorithm.evaluation_and_save_frequency": 4_096,
                "checkpoint.save_interval_steps": 4_096,
            }
        )
    return _form_values_from_config(config, algorithm=algorithm, preset="custom")


def _form_values_from_config(
    config: RLRunConfig,
    *,
    algorithm: AlgorithmChoice,
    preset: PresetKind,
    use_config_tracking: bool = False,
) -> RunFormValues:
    """Project a run config into editable form fields."""
    algorithm_config = config.algorithm
    environment = config.environment
    tracking = config.tracking
    return RunFormValues(
        algorithm=algorithm,
        preset=preset,
        env_id=environment.env_id,
        seed=environment.seed,
        total_timesteps=algorithm_config.total_timesteps,
        nr_envs=environment.nr_envs,
        nr_steps=algorithm_config.nr_steps,
        minibatch_size=algorithm_config.minibatch_size,
        eval_frequency=algorithm_config.evaluation_and_save_frequency,
        learning_rate=algorithm_config.learning_rate,
        obs_encoding_dim=algorithm_config.obs_encoding_dim,
        gru_hidden_dim=algorithm_config.gru_hidden_dim,
        track_wandb=tracking.track_wandb if use_config_tracking else False,
        track_tensorboard=tracking.track_tensorboard if use_config_tracking else True,
        wandb_project=tracking.wandb_project if use_config_tracking else "jarl",
        wandb_mode=tracking.wandb_mode or "offline",
        record_video=config.video.record_video if use_config_tracking else False,
        gamma=algorithm_config.gamma,
        gae_lambda=algorithm_config.gae_lambda,
        clip_range=algorithm_config.clip_range,
        entropy_coef=algorithm_config.entropy_coef,
        nr_epochs=algorithm_config.nr_epochs,
        save_model=config.runner.save_model if use_config_tracking else DEFAULT_SAVE_MODEL,
        evaluation_active=algorithm_config.evaluation_active,
        checkpoint_save_interval_steps=config.checkpoint.save_interval_steps,
        wandb_entity=(tracking.wandb_entity or "") if use_config_tracking else "",
        wandb_group=tracking.wandb_group if use_config_tracking else "",
        wandb_tags=",".join(tracking.wandb_tags) if use_config_tracking else "",
        max_episode_steps=environment.max_episode_steps,
    )


def algorithm_choice_from_name(algorithm_name: str) -> AlgorithmChoice:
    """Map a canonical algorithm name to the UI radio choice."""
    if "gru" in algorithm_name:
        return "ppo_gru"
    return "ppo"


def form_values_from_run_config(config: RLRunConfig) -> RunFormValues:
    """Project a resolved node config into persisted form fields."""
    algorithm = algorithm_choice_from_name(config.algorithm.name)
    return _form_values_from_config(
        config,
        algorithm=algorithm,
        preset="custom",
        use_config_tracking=True,
    )


def preset_run_config(*, algorithm: AlgorithmChoice = "ppo") -> RLRunConfig:
    """Build the short Navix preset used for local demos."""
    environment = EnvironmentConfig(
        env_id=MINIMAL_NAVIX_ENV_ID,
        nr_envs=4,
        seed=0,
        max_episode_steps=16,
    )
    algorithm_config = AlgorithmConfig(
        total_timesteps=512,
        nr_steps=8,
        minibatch_size=32,
        nr_epochs=1,
        evaluation_and_save_frequency=128,
        evaluation_active=True,
    )
    if algorithm == "ppo_gru":
        algorithm_config = algorithm_config.model_copy(
            update={
                "name": PPO_GRU_ALGORITHM_NAME,
                "obs_encoding_dim": 16,
                "gru_hidden_dim": 8,
            }
        )
    return RLRunConfig(
        environment=environment,
        algorithm=algorithm_config,
        video=VideoConfig(record_video=False, record_final_video=False),
        runner=RLRunnerConfig(save_model=True),
    )


def recommended_video_frequency(*, total_timesteps: int, eval_frequency: int) -> int:
    """Pick a video cadence that fires at least once for the run length."""
    if eval_frequency > 0:
        return min(eval_frequency, total_timesteps)
    if total_timesteps >= _LONG_RUN_MIN_TIMESTEPS:
        return max(65_536, total_timesteps // 20)
    return max(32, total_timesteps // 2)


def form_values_to_run_config(payload: RunFormPayload) -> RLRunConfig:
    """Convert form values into a resolved ``RLRunConfig``."""
    values = payload.values
    if values.preset == "fast":
        base = preset_run_config(algorithm=values.algorithm)
        total_timesteps = base.algorithm.total_timesteps
        eval_frequency = base.algorithm.evaluation_and_save_frequency
        checkpoint_save_interval_steps = base.checkpoint.save_interval_steps
        save_model = values.save_model
    else:
        algorithm_name = PPO_ALGORITHM_NAME if values.algorithm == "ppo" else PPO_GRU_ALGORITHM_NAME
        total_timesteps = values.total_timesteps
        eval_frequency = values.eval_frequency
        checkpoint_save_interval_steps = values.checkpoint_save_interval_steps
        save_model = values.save_model
        base = RLRunConfig(
            environment=EnvironmentConfig(
                env_id=values.env_id,
                nr_envs=values.nr_envs,
                seed=values.seed,
                max_episode_steps=values.max_episode_steps,
            ),
            algorithm=AlgorithmConfig(
                name=algorithm_name,
                total_timesteps=values.total_timesteps,
                nr_steps=values.nr_steps,
                minibatch_size=values.minibatch_size,
                nr_epochs=values.nr_epochs,
                learning_rate=values.learning_rate,
                gamma=values.gamma,
                gae_lambda=values.gae_lambda,
                clip_range=values.clip_range,
                entropy_coef=values.entropy_coef,
                evaluation_and_save_frequency=values.eval_frequency,
                evaluation_active=values.evaluation_active,
                obs_encoding_dim=values.obs_encoding_dim,
                gru_hidden_dim=values.gru_hidden_dim,
            ),
            video=VideoConfig(
                record_video=values.record_video,
                record_final_video=payload.record_final_video,
            ),
            runner=RLRunnerConfig(save_model=values.save_model),
        )
    video_frequency = payload.video_frequency
    if video_frequency <= 0:
        if values.record_video:
            video_frequency = recommended_video_frequency(
                total_timesteps=total_timesteps,
                eval_frequency=eval_frequency,
            )
        else:
            video_frequency = base.video.video_frequency
    overrides: dict[str, object] = {
        "tracking.track_wandb": values.track_wandb,
        "tracking.track_tensorboard": values.track_tensorboard,
        "tracking.wandb_project": values.wandb_project,
        "tracking.wandb_mode": values.wandb_mode,
        "tracking.wandb_entity": values.wandb_entity or None,
        "tracking.wandb_group": values.wandb_group,
        "tracking.wandb_tags": _parse_tags(values.wandb_tags),
        "video.record_video": values.record_video,
        "video.record_final_video": payload.record_final_video,
        "video.video_frequency": video_frequency,
        "video.video_episodes": 1,
        "checkpoint.save_interval_steps": checkpoint_save_interval_steps,
        "runner.save_model": save_model,
    }
    if values.preset == "fast" and values.max_episode_steps is not None:
        overrides["environment.max_episode_steps"] = values.max_episode_steps
    return base.apply_overrides(overrides)


def run_overrides_from_payload(
    payload: RunFormPayload,
    *,
    base_config: RLRunConfig | None = None,
) -> dict[str, object]:
    """Sparse overrides for training an existing node (mutable paths only)."""
    overrides = retrain_overrides_from_config(form_values_to_run_config(payload))
    validation_base = base_config or _reference_base_config(payload)
    if validation_base is not None:
        ensure_validated_config_overrides(validation_base, overrides, policy="retrain")
    return overrides


def child_config_overrides_from_payload(
    parent_config: RLRunConfig,
    payload: RunFormPayload,
) -> dict[str, object]:
    """Sparse overrides for fork/extend relative to the parent resolved config."""
    config = form_values_to_run_config(payload)
    overrides = mutable_overrides_from_config(config)
    target_env_id = payload.values.env_id
    if target_env_id != parent_config.environment.env_id:
        assert_transfer_compatible(parent_config.environment.env_id, target_env_id)
        overrides["environment.env_id"] = target_env_id
    target_horizon = payload.values.max_episode_steps
    if target_horizon != parent_config.environment.max_episode_steps:
        overrides["environment.max_episode_steps"] = target_horizon
    ensure_validated_config_overrides(parent_config, overrides, policy="fork")
    return overrides


def _reference_base_config(payload: RunFormPayload) -> RLRunConfig | None:
    """Return a synthetic base config for legacy ``reference_nr_envs`` validation."""
    if payload.reference_nr_envs is None:
        return None
    reference_base = preset_run_config()
    return reference_base.model_copy(
        update={
            "environment": reference_base.environment.model_copy(
                update={"nr_envs": payload.reference_nr_envs},
            ),
        },
    )


def _parse_tags(raw: str) -> list[str]:
    return [tag.strip() for tag in raw.split(",") if tag.strip()]


def rollout_batch_size(values: RunFormValues, *, environment_nr_envs: int | None = None) -> int:
    """Return rollout batch size implied by form values."""
    nr_envs = environment_nr_envs if environment_nr_envs is not None else values.nr_envs
    return compute_rollout_batch_size(nr_envs=nr_envs, nr_steps=values.nr_steps)


def validate_run_form_values(
    values: RunFormValues,
    *,
    environment_nr_envs: int | None = None,
) -> list[str]:
    """Return validation messages for run form values."""
    messages: list[str] = []
    nr_envs = environment_nr_envs if environment_nr_envs is not None else values.nr_envs
    batch_error = rollout_batch_divisibility_error(
        nr_envs=nr_envs,
        nr_steps=values.nr_steps,
        minibatch_size=values.minibatch_size,
    )
    if batch_error is not None:
        messages.append(batch_error)
    if values.eval_frequency == 0:
        messages.append("evaluation_and_save_frequency debe ser -1 o positivo.")
    if values.track_wandb and values.wandb_mode == "online" and not values.wandb_project:
        messages.append("wandb_project no puede estar vacío con W&B online.")
    if values.max_episode_steps is not None and values.max_episode_steps <= 0:
        messages.append("max_episode_steps debe ser positivo o null.")
    return messages
