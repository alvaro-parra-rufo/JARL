"""PPO-GRU full-JAX trainer for ``jarl`` experiment nodes."""

from __future__ import annotations

import logging
from collections.abc import Callable

import jax
import optax
from flax.training.train_state import TrainState

from jarl.agents.ppo.checkpoint_state import PPOGrUCheckpoint
from jarl.agents.ppo.core.hyperparameters import PPOHyperparameters
from jarl.agents.ppo.core.learning_rate import make_linear_annealed_learning_rate_schedule
from jarl.agents.ppo.gru_loop import build_ppo_gru_train_fn
from jarl.agents.ppo.loop_config import PPOGrULoopConfig
from jarl.agents.ppo.networks import create_gru_critic_network, create_gru_policy_network
from jarl.agents.ppo.trainer import PPOTrainHost
from jarl.agents.ppo.training_restore import prepare_training_restore
from jarl.agents.ppo.video import AsyncNavixVideoRecorder
from jarl.agents.ppo.video.recurrent_rollout import NavixGreedyRecurrentVideoRollout
from jarl.env_setup import log_jax_training_devices
from jarl.envs.types import ActionSpaceType
from jarl.experiments.io.artifacts import ArtifactRecord
from jarl.experiments.node import NodeWorkspace
from jarl.training.config import RLRunConfig
from jarl.training.env_factory import navix_full_jit_env_factory
from jarl.training.schedule import TrainingSchedule
from jarl.training.trainer import EnvFactory

LOGGER = logging.getLogger(__name__)

__all__ = ["ppo_gru_full_jax_trainer"]


def _create_gru_train_states(
    *,
    policy: object,
    critic: object,
    sample_observation: jax.Array,
    loop_config: PPOGrULoopConfig,
    policy_key: jax.Array,
    critic_key: jax.Array,
) -> tuple[TrainState, TrainState]:
    if loop_config.anneal_learning_rate and loop_config.nr_updates > 0:
        learning_rate: float | Callable[[jax.Array], jax.Array] = make_linear_annealed_learning_rate_schedule(
            base_learning_rate=loop_config.learning_rate,
            nr_updates=loop_config.nr_updates,
            nr_minibatches=loop_config.nr_minibatches,
            nr_epochs=loop_config.nr_epochs,
        )
    else:
        learning_rate = loop_config.learning_rate

    optimizer = optax.chain(
        optax.clip_by_global_norm(loop_config.max_grad_norm),
        optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate),
    )
    dummy_policy_carry = policy.initialize_carry(loop_config.nr_envs)
    dummy_critic_carry = critic.initialize_carry(loop_config.nr_envs)
    policy_state = TrainState.create(
        apply_fn=policy.apply,
        params=policy.init(
            policy_key,
            sample_observation,
            dummy_policy_carry,
            method=policy.apply_one_step,
        ),
        tx=optimizer,
    )
    critic_state = TrainState.create(
        apply_fn=critic.apply,
        params=critic.init(
            critic_key,
            sample_observation,
            dummy_critic_carry,
            method=critic.apply_one_step,
        ),
        tx=optimizer,
    )
    return policy_state, critic_state


def ppo_gru_full_jax_trainer(
    workspace: NodeWorkspace,
    config: RLRunConfig,
    schedule: TrainingSchedule,
    *,
    env_factory: EnvFactory | None = None,
) -> None:
    """Run PPO-GRU full-JAX training inside an active node workspace."""
    factory = env_factory or navix_full_jit_env_factory
    train_env = factory(config)
    eval_env = factory(config)
    workspace.init_checkpoint_manager(
        max_to_keep=config.checkpoint.max_to_keep,
        save_interval_steps=config.checkpoint.save_interval_steps,
    )
    workspace.bind_model_archive_context(
        algorithm_name=config.algorithm.name,
        env_id=config.environment.env_id,
    )
    is_discrete = train_env.general_properties.action_space_type == ActionSpaceType.DISCRETE
    hyperparameters = PPOHyperparameters.from_algorithm_config(config.algorithm)
    policy, process_action = create_gru_policy_network(config.algorithm, train_env)
    critic = create_gru_critic_network(config.algorithm, train_env)

    video_rollout = None
    video_recorder = None
    if config.video.record_video or config.video.record_final_video:
        workspace.videos_dir.mkdir(parents=True, exist_ok=True)
        video_recorder = AsyncNavixVideoRecorder(
            workspace.videos_dir,
            workspace.video_metrics_jsonl_path,
            max_pending_videos=config.video.max_pending_videos,
        )

        def apply_policy_one_step(
            policy_params: object,
            observation: jax.Array,
            carry: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            return policy.apply(policy_params, observation, carry, method=policy.apply_one_step)

        video_rollout = NavixGreedyRecurrentVideoRollout(
            env_id=config.environment.env_id,
            preprocess_observation=train_env.preprocess_observation,
            policy_apply_one_step=apply_policy_one_step,
            initialize_carry=policy.initialize_carry,
            max_steps=config.video.video_max_steps,
            view_mode=config.video.video_view_mode,
        )

    key = jax.random.PRNGKey(config.environment.seed)
    key, policy_key, critic_key, reset_key = jax.random.split(key, 4)
    reset_keys = jax.random.split(reset_key, config.environment.nr_envs)
    env_state = train_env.reset(reset_keys, False)
    bootstrap_loop_config = PPOGrULoopConfig.from_run(config, schedule, horizon=train_env.horizon)
    policy_state, critic_state = _create_gru_train_states(
        policy=policy,
        critic=critic,
        sample_observation=env_state.next_observation,
        loop_config=bootstrap_loop_config,
        policy_key=policy_key,
        critic_key=critic_key,
    )
    zero_policy_carry = policy.initialize_carry(config.environment.nr_envs)
    zero_critic_carry = critic.initialize_carry(config.environment.nr_envs)
    restored = prepare_training_restore(
        workspace=workspace,
        config=config,
        schedule=schedule,
        policy_template=policy_state,
        critic_template=critic_state,
        zero_carry=zero_policy_carry,
        key=key,
    )
    config = restored.config
    schedule = restored.schedule
    key = restored.key
    policy_state = restored.policy_state
    critic_state = restored.critic_state
    loop_config = PPOGrULoopConfig.from_run(config, schedule, horizon=train_env.horizon)

    host = PPOTrainHost(
        workspace=workspace,
        loop_config=loop_config,
        seed=config.environment.seed,
        is_discrete=is_discrete,
        record_video=config.video.record_video,
        record_final_video=config.video.record_final_video,
        video_frequency=config.video.video_frequency,
        video_episodes=config.video.video_episodes,
        video_fps=config.video.video_fps,
        video_scale=config.video.video_scale,
        final_video_episodes=config.video.final_video_episodes,
        video_rollout=video_rollout,
        video_recorder=video_recorder,
        _next_video_step=config.video.video_frequency,
    )

    train_fn = build_ppo_gru_train_fn(
        train_env=train_env,
        eval_env=eval_env,
        policy=policy,
        critic=critic,
        process_action=process_action,
        loop_config=loop_config,
        hyperparameters=hyperparameters,
        is_discrete=is_discrete,
        log_train_metrics=host.make_train_log_callback(),
        log_eval_metrics=host.make_eval_log_callback(),
        save_model=workspace.make_model_archive_save_callback(),
        save_checkpoint=host.make_checkpoint_save_callback(
            algorithm_name=config.algorithm.name,
            checkpoint_policy=config.checkpoint,
            total_timesteps=config.algorithm.total_timesteps,
            is_gru=True,
        ),
    )
    train_fn = jax.jit(train_fn)
    log_jax_training_devices(LOGGER)
    LOGGER.info("Starting PPO-GRU full-JAX training")
    try:
        policy_state, critic_state = jax.block_until_ready(
            train_fn(key, policy_state, critic_state),
        )
        if config.video.record_final_video:
            host.record_final_videos(policy_state)
    finally:
        if video_recorder is not None:
            video_recorder.shutdown(wait=True)

    final_step = (
        loop_config.nr_multi_eval_iterations
        * loop_config.nr_updates_per_eval
        * loop_config.nr_steps
        * loop_config.nr_envs
    )
    final_optimizer_updates = (
        loop_config.nr_multi_eval_iterations
        * loop_config.nr_updates_per_eval
        * loop_config.nr_epochs
        * loop_config.nr_minibatches
    )
    workspace.save_training_checkpoint_if_absent(
        final_step,
        PPOGrUCheckpoint.build(
            algorithm_name=config.algorithm.name,
            global_step=final_step,
            optimizer_updates=final_optimizer_updates,
            rng_key=key,
            policy_state=policy_state,
            critic_state=critic_state,
            policy_carry=zero_policy_carry,
            critic_carry=zero_critic_carry,
        ),
    )

    if config.runner.save_model:
        archive_path = workspace.save_model_archive(
            "latest",
            final_step,
            policy=policy_state.params,
            critic=critic_state.params,
            alias="latest",
            algorithm_name=config.algorithm.name,
            env_id=config.environment.env_id,
        )
        workspace.register_artifact(ArtifactRecord.model(name="latest", step=final_step, filename=archive_path.name))
