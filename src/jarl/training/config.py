"""RL training configuration models."""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal, Self

from pydantic import Field, model_validator

from jarl.config import BaseConfig
from jarl.envs.navix.reward_config import RewardWeightsConfig
from jarl.experiments.run_config import RunConfig

__all__ = [
    "AlgorithmConfig",
    "EnvironmentConfig",
    "RLRunConfig",
    "RLRunnerConfig",
    "RewardWeightsConfig",
    "VideoConfig",
    "rollout_batch_divisibility_error",
    "rollout_batch_size",
]


def rollout_batch_size(*, nr_envs: int, nr_steps: int) -> int:
    """Return ``nr_envs * nr_steps`` for PPO rollout batch sizing."""
    return nr_envs * nr_steps


def rollout_batch_divisibility_error(
    *,
    nr_envs: int,
    nr_steps: int,
    minibatch_size: int,
) -> str | None:
    """Return a validation message when ``minibatch_size`` does not divide the rollout batch."""
    batch_size = rollout_batch_size(nr_envs=nr_envs, nr_steps=nr_steps)
    if batch_size % minibatch_size != 0:
        return (
            f"minibatch_size ({minibatch_size}) debe dividir nr_envs * nr_steps "
            f"({batch_size} = {nr_envs} * {nr_steps})."
        )
    return None


class EnvironmentConfig(BaseConfig):
    """Navix full-JIT environment settings."""

    IMMUTABLE_PATHS: ClassVar[frozenset[str]] = frozenset()

    env_id: Annotated[
        str,
        Field(description="Gymnasium environment identifier (for example ``Navix-LavaGapS5-v0``)."),
    ] = "Navix-LavaGapS5-v0"
    nr_envs: Annotated[
        int,
        Field(description="Number of parallel vectorized environments.", ge=1),
    ] = 64
    seed: Annotated[
        int,
        Field(description="Random seed for environment and action-space initialization."),
    ] = 42
    max_episode_steps: Annotated[
        int | None,
        Field(description="Maximum steps per episode (`None` uses the environment default)."),
    ] = None
    render: Annotated[
        bool,
        Field(description="Whether to enable environment rendering during training."),
    ] = False
    render_callback_type: Annotated[
        str,
        Field(description="JAX render callback backend (for example ``io_callback``)."),
    ] = "io_callback"
    reward: Annotated[
        RewardWeightsConfig | None,
        Field(
            description=(
                "Named weights for the configurable Navix reward mix. ``None`` keeps the environment's native reward."
            ),
        ),
    ] = None
    scenario_reward_id: Annotated[
        str | None,
        Field(description="Internal scenario overlay identifier, or ``None`` when unused."),
    ] = None
    scenario_reward_version: Annotated[
        int | None,
        Field(description="Internal scenario overlay version, or ``None`` when unused."),
    ] = None

    @model_validator(mode="after")
    def _validate_scenario_reward(self) -> Self:
        """Require overlay id and version together, and a custom mix when an overlay is set."""
        has_id = self.scenario_reward_id is not None
        has_version = self.scenario_reward_version is not None
        if has_id != has_version:
            msg = "scenario_reward_id and scenario_reward_version must both be set or both be None."
            raise ValueError(msg)
        if has_id and self.reward is None:
            msg = "scenario_reward_id requires environment.reward to be set."
            raise ValueError(msg)
        return self


class AlgorithmConfig(BaseConfig):
    """Shared PPO / PPO-GRU algorithm settings and derived schedule fields."""

    IMMUTABLE_PATHS: ClassVar[frozenset[str]] = frozenset(
        {
            "name",
            "obs_encoding_dim",
            "gru_hidden_dim",
            "gru_obs_combine_method",
            "share_gru_obs_encoder",
        }
    )

    name: Annotated[
        str,
        Field(description="Algorithm identifier recorded in run metadata."),
    ] = "ppo.full_jax.navix"
    total_timesteps: Annotated[
        int,
        Field(description="Requested environment-step budget for training.", ge=1),
    ] = 5_000_000
    nr_steps: Annotated[
        int,
        Field(description="Rollout length per environment before each PPO update.", ge=1),
    ] = 128
    nr_epochs: Annotated[
        int,
        Field(description="Number of optimization epochs per rollout batch.", ge=1),
    ] = 4
    minibatch_size: Annotated[
        int,
        Field(description="PPO minibatch size in environment steps.", ge=1),
    ] = 2048
    learning_rate: Annotated[
        float,
        Field(description="Initial PPO learning rate.", gt=0.0),
    ] = 2.5e-4
    anneal_learning_rate: Annotated[
        bool,
        Field(description="Whether to linearly anneal the learning rate during training."),
    ] = True
    gamma: Annotated[
        float,
        Field(description="Discount factor.", gt=0.0, le=1.0),
    ] = 0.99
    gae_lambda: Annotated[
        float,
        Field(description="GAE lambda for advantage estimation.", ge=0.0, le=1.0),
    ] = 0.95
    clip_range: Annotated[
        float,
        Field(description="PPO clip range.", gt=0.0),
    ] = 0.2
    entropy_coef: Annotated[
        float,
        Field(description="Entropy bonus coefficient."),
    ] = 0.05
    critic_coef: Annotated[
        float,
        Field(description="Value-loss coefficient."),
    ] = 0.5
    max_grad_norm: Annotated[
        float,
        Field(description="Maximum gradient norm for clipping.", gt=0.0),
    ] = 0.5
    std_dev: Annotated[
        float,
        Field(description="Initial policy standard deviation for continuous control.", gt=0.0),
    ] = 1.0
    action_clipping_and_rescaling: Annotated[
        bool,
        Field(description="Whether to clip and rescale continuous actions."),
    ] = False
    obs_encoding_dim: Annotated[
        int,
        Field(description="Observation encoder width for PPO-GRU policy networks.", ge=1),
    ] = 128
    gru_hidden_dim: Annotated[
        int,
        Field(description="GRU hidden state size for PPO-GRU policy networks.", ge=1),
    ] = 64
    gru_obs_combine_method: Annotated[
        Literal["concat", "film"],
        Field(description="How GRU latent and observation latent are fused before the policy torso."),
    ] = "concat"
    share_gru_obs_encoder: Annotated[
        bool,
        Field(description="Whether the GRU input encoder is reused as the observation encoder."),
    ] = False
    evaluation_and_save_frequency: Annotated[
        int,
        Field(
            description=(
                "Environment steps between evaluation and checkpoint events. "
                "Use ``-1`` to auto-align with the rollout batch size."
            ),
            ge=-1,
        ),
    ] = 65_536
    evaluation_active: Annotated[
        bool,
        Field(description="Whether periodic evaluation runs during training."),
    ] = True
    requested_total_timesteps: Annotated[
        int | None,
        Field(
            description=(
                "Copy of ``total_timesteps`` before schedule rounding. "
                "Populated by ``apply_training_schedule``; do not set manually."
            ),
        ),
    ] = None
    actual_total_timesteps: Annotated[
        int | None,
        Field(
            description=(
                "Effective environment-step budget after schedule rounding: ``actual_rollout_updates * batch_size``."
            ),
        ),
    ] = None
    actual_rollout_updates: Annotated[
        int | None,
        Field(
            description=(
                "Number of rollout updates executed during training: ``multi_iterations * updates_per_eval``."
            ),
        ),
    ] = None
    actual_optimizer_updates: Annotated[
        int | None,
        Field(
            description=("Total optimizer steps: ``actual_rollout_updates * nr_epochs * nr_minibatches``."),
        ),
    ] = None
    batch_size: Annotated[
        int | None,
        Field(
            description="Rollout batch size in environment steps: ``nr_envs * nr_steps``.",
        ),
    ] = None
    nr_minibatches: Annotated[
        int | None,
        Field(
            description="Minibatches per rollout batch: ``batch_size // minibatch_size``.",
        ),
    ] = None
    effective_evaluation_and_save_frequency: Annotated[
        int | None,
        Field(
            description=(
                "Resolved evaluation/checkpoint interval. When ``evaluation_and_save_frequency`` "
                "is ``-1``, equals ``batch_size * (total_timesteps // batch_size)``."
            ),
        ),
    ] = None

    @model_validator(mode="after")
    def _validate_evaluation_frequency(self) -> Self:
        """Reject a zero evaluation frequency."""
        if self.evaluation_and_save_frequency == 0:
            msg = "evaluation_and_save_frequency must be -1 or a positive integer."
            raise ValueError(msg)
        return self


class RLRunnerConfig(BaseConfig):
    """RL runner execution settings.

    Experiment tracking flags live on ``RunConfig.tracking``; this block covers
    model export handled by the training runner.
    """

    save_model: Annotated[
        bool,
        Field(description="Whether to persist model checkpoints to disk."),
    ] = True


class VideoConfig(BaseConfig):
    """Policy video recording settings."""

    record_video: Annotated[
        bool,
        Field(description="Whether to record intermediate rollout videos during training."),
    ] = True
    record_final_video: Annotated[
        bool,
        Field(description="Whether to record a final evaluation video after training."),
    ] = True
    video_frequency: Annotated[
        int,
        Field(description="Environment steps between intermediate video recordings.", ge=1),
    ] = 1_000_000
    video_episodes: Annotated[
        int,
        Field(description="Number of episodes captured per intermediate video event.", ge=1),
    ] = 54
    video_fps: Annotated[
        int,
        Field(description="Frames per second for rendered videos.", ge=1),
    ] = 15
    video_max_steps: Annotated[
        int,
        Field(description="Maximum steps recorded per episode.", ge=1),
    ] = 500
    video_view_mode: Annotated[
        Literal["full", "first_person"],
        Field(description="Navix camera mode used when rendering videos."),
    ] = "full"
    video_scale: Annotated[
        int,
        Field(description="Integer upscaling factor applied to rendered frames.", ge=1),
    ] = 4
    final_video_episodes: Annotated[
        int,
        Field(description="Number of episodes captured in the final evaluation video.", ge=1),
    ] = 5
    max_pending_videos: Annotated[
        int,
        Field(description="Maximum number of intermediate videos queued before blocking.", ge=1),
    ] = 1
    video_dir: Annotated[
        str,
        Field(description="Relative directory name for saved videos within the node layout."),
    ] = "videos"


class RLRunConfig(RunConfig):
    """Top-level RL experiment configuration.

    Extends ``RunConfig`` with nested blocks for environment, algorithm, runner,
    and video settings shared by PPO and PPO-GRU trainers.
    """

    DEFAULT_PROJECT_NAME: ClassVar[str] = "navix"
    DEFAULT_EXP_NAME: ClassVar[str] = "ppo_full_jax"

    RETRAIN_MUTABLE_PATHS: ClassVar[frozenset[str]] = frozenset(
        {
            "tracking.track_wandb",
            "tracking.track_tensorboard",
            "tracking.wandb_project",
            "tracking.wandb_mode",
            "algorithm.total_timesteps",
            "algorithm.nr_steps",
            "algorithm.minibatch_size",
            "algorithm.learning_rate",
            "algorithm.gamma",
            "algorithm.gae_lambda",
            "algorithm.clip_range",
            "algorithm.entropy_coef",
            "algorithm.nr_epochs",
            "algorithm.evaluation_and_save_frequency",
            "algorithm.evaluation_active",
            "video.record_video",
            "video.record_final_video",
            "video.video_frequency",
            "video.video_episodes",
            "checkpoint.save_interval_steps",
            "runner.save_model",
        }
    )
    """Dotted paths allowed when re-training or resuming an existing node."""

    FORK_EXTRA_MUTABLE_PATHS: ClassVar[frozenset[str]] = frozenset(
        {"environment.env_id", "environment.max_episode_steps"}
    )
    """Additional dotted paths allowed when forking or extending a child node."""

    environment: Annotated[
        EnvironmentConfig,
        Field(description="Navix full-JIT environment settings."),
    ] = EnvironmentConfig()
    algorithm: Annotated[
        AlgorithmConfig,
        Field(description="Shared PPO algorithm and derived schedule fields."),
    ] = AlgorithmConfig()
    runner: Annotated[
        RLRunnerConfig,
        Field(description="Runner model IO settings."),
    ] = RLRunnerConfig()
    video: Annotated[
        VideoConfig,
        Field(description="Policy video recording settings."),
    ] = VideoConfig()

    @model_validator(mode="after")
    def _validate_rollout_batch(self) -> Self:
        """Ensure the PPO minibatch divides the rollout batch size."""
        error = rollout_batch_divisibility_error(
            nr_envs=self.environment.nr_envs,
            nr_steps=self.algorithm.nr_steps,
            minibatch_size=self.algorithm.minibatch_size,
        )
        if error is not None:
            raise ValueError(error)
        return self
