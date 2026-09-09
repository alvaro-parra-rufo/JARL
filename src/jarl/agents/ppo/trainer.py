"""PPO full-JAX feedforward trainer for ``jarl`` experiment nodes."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import numpy as np
import optax
from flax.training.train_state import TrainState

from jarl.agents.ppo.callbacks import make_metric_bundle_callback
from jarl.agents.ppo.checkpoint_state import PPOCheckpoint, PPOGrUCheckpoint
from jarl.agents.ppo.core.hyperparameters import PPOHyperparameters
from jarl.agents.ppo.core.learning_rate import make_linear_annealed_learning_rate_schedule
from jarl.agents.ppo.feedforward_loop import build_ppo_feedforward_train_fn
from jarl.agents.ppo.loop_config import PPOFeedforwardLoopConfig, PPOGrULoopConfig
from jarl.agents.ppo.metric_schema import (
    PPO_EVAL_METRIC_SCHEMA,
    PPO_TRAIN_METRIC_SCHEMA,
    assemble_metric_bundle,
)
from jarl.agents.ppo.networks import create_critic_network, create_policy_network
from jarl.agents.ppo.training_restore import prepare_training_restore
from jarl.agents.ppo.video import AsyncNavixVideoRecorder, NavixGreedyVideoRollout, NavixVideoJob
from jarl.env_setup import log_jax_training_devices
from jarl.envs.types import ActionSpaceType
from jarl.experiments.io.artifacts import ArtifactRecord
from jarl.experiments.node import NodeWorkspace
from jarl.experiments.run_config import CheckpointConfig
from jarl.training.config import RLRunConfig
from jarl.training.env_factory import navix_full_jit_env_factory
from jarl.training.schedule import TrainingSchedule
from jarl.training.trainer import EnvFactory

LOGGER = logging.getLogger(__name__)

__all__ = ["PPOTrainHost", "ppo_full_jax_trainer"]


def _zero_gru_carry(carry: Any | None) -> Any | None:
    """Return a zero GRU carry of the same shape, keeping the checkpoint field."""
    if carry is None:
        return None
    return jax.numpy.zeros_like(jax.numpy.asarray(carry))


@dataclass
class PPOTrainHost:
    """Host-side services invoked from ``jax.debug.callback`` during PPO training."""

    workspace: NodeWorkspace
    loop_config: PPOFeedforwardLoopConfig | PPOGrULoopConfig
    seed: int
    is_discrete: bool
    record_video: bool
    record_final_video: bool
    video_frequency: int
    video_episodes: int
    video_fps: int
    video_scale: int
    final_video_episodes: int
    video_rollout: NavixGreedyVideoRollout | object | None = None
    video_recorder: AsyncNavixVideoRecorder | None = None
    _last_log_time: float = field(default_factory=time.time)
    _next_video_step: int = 0
    _latest_policy_state: TrainState | None = field(default=None, repr=False)

    def make_train_log_callback(self) -> Callable[..., None]:
        """Build JIT callback for train-phase metric bundles."""
        bundle_log = make_metric_bundle_callback(self.workspace.log_scalars, PPO_TRAIN_METRIC_SCHEMA)

        def callback(combined_rollout_step: Any, metric_body: Any, policy_state: TrainState) -> None:
            self._latest_policy_state = policy_state
            rollout_step = int(combined_rollout_step)
            body = np.asarray(metric_body, dtype=np.float64)
            global_step = rollout_step * self.loop_config.nr_steps * self.loop_config.nr_envs
            current_time = time.time()
            elapsed = max(current_time - self._last_log_time, 1e-8)
            self._last_log_time = current_time
            steps_per_second = int((self.loop_config.nr_steps * self.loop_config.nr_envs) / elapsed)
            optimizer_updates = rollout_step * self.loop_config.nr_epochs * self.loop_config.nr_minibatches
            host_values = {
                "time/sps": float(steps_per_second),
                "steps/nr_env_steps": float(global_step),
                "steps/nr_rollout_updates": float(rollout_step),
                "steps/nr_optimizer_updates": float(optimizer_updates),
                "steps/nr_updates": float(optimizer_updates),
            }
            full = assemble_metric_bundle(body, host_values, PPO_TRAIN_METRIC_SCHEMA)
            bundle_log(global_step, full)
            if self.record_video and self.video_frequency > 0 and global_step >= self._next_video_step:
                self.record_video_progress(global_step, policy_state)
                while self._next_video_step <= global_step:
                    self._next_video_step += self.video_frequency

        return callback

    def make_eval_log_callback(self) -> Callable[..., None]:
        """Build JIT callback for evaluation metrics."""
        return make_metric_bundle_callback(self.workspace.log_scalars, PPO_EVAL_METRIC_SCHEMA)

    def make_checkpoint_save_callback(
        self,
        *,
        algorithm_name: str,
        checkpoint_policy: CheckpointConfig,
        total_timesteps: int,
        is_gru: bool = False,
    ) -> Callable[..., None]:
        """Build JIT callback that persists typed training checkpoints on the host."""

        def callback(
            checkpoint_step: Any,
            global_step: Any,
            optimizer_updates: Any,
            rng_key: Any,
            policy_state: TrainState,
            critic_state: TrainState,
            policy_carry: Any | None = None,
            critic_carry: Any | None = None,
        ) -> None:
            key = jax.numpy.asarray(rng_key)
            if is_gru:
                checkpoint = PPOGrUCheckpoint.build(
                    algorithm_name=algorithm_name,
                    global_step=int(global_step),
                    optimizer_updates=int(optimizer_updates),
                    rng_key=key,
                    policy_state=policy_state,
                    critic_state=critic_state,
                    policy_carry=_zero_gru_carry(policy_carry),
                    critic_carry=_zero_gru_carry(critic_carry),
                )
            else:
                checkpoint = PPOCheckpoint.build(
                    algorithm_name=algorithm_name,
                    global_step=int(global_step),
                    optimizer_updates=int(optimizer_updates),
                    rng_key=key,
                    policy_state=policy_state,
                    critic_state=critic_state,
                )
            self.workspace.save_training_checkpoint_if_due(
                int(checkpoint_step),
                checkpoint,
                policy=checkpoint_policy,
                total_steps=total_timesteps,
                global_step=int(global_step),
                optimizer_updates=int(optimizer_updates),
            )

        return callback

    def record_video_progress(self, global_step: int, policy_state: TrainState) -> None:
        """Schedule an intermediate greedy-policy video when recording is enabled."""
        if self.video_recorder is None or self.video_rollout is None or not self.is_discrete:
            return
        if not self.video_recorder.can_accept_intermediate():
            return
        job = self._generate_video_job(
            policy_params=policy_state.params,
            global_step=global_step,
            episodes=self.video_episodes,
            name_prefix=f"step_{global_step}",
            is_final=False,
        )
        if job is not None:
            self.video_recorder.submit_intermediate(job)

    def record_final_videos(self, policy_state: TrainState) -> None:
        """Schedule the final evaluation video."""
        if self.video_recorder is None or self.video_rollout is None or not self.record_final_video:
            return
        global_step = (
            self.loop_config.nr_multi_eval_iterations
            * self.loop_config.nr_updates_per_eval
            * self.loop_config.nr_steps
            * self.loop_config.nr_envs
        )
        job = self._generate_video_job(
            policy_params=policy_state.params,
            global_step=global_step,
            episodes=self.final_video_episodes,
            name_prefix="final",
            is_final=True,
        )
        if job is not None:
            self.video_recorder.submit_final(job)

    def _generate_video_job(
        self,
        *,
        policy_params: object,
        global_step: int,
        episodes: int,
        name_prefix: str,
        is_final: bool,
    ) -> NavixVideoJob | None:
        if self.video_rollout is None:
            return None
        keys = jax.random.split(jax.random.PRNGKey(self.seed + int(global_step)), int(episodes))
        rollout_start = time.perf_counter()
        rollout_batch = self.video_rollout.rollout(policy_params, keys)
        jax.block_until_ready(rollout_batch.frames)
        rollout_seconds = time.perf_counter() - rollout_start
        transfer_start = time.perf_counter()
        rollout_batch_cpu = jax.device_get(rollout_batch)
        transfer_seconds = time.perf_counter() - transfer_start
        return NavixVideoJob(
            frames=np.asarray(rollout_batch_cpu.frames),
            episode_returns=np.asarray(rollout_batch_cpu.episode_returns),
            episode_lengths=np.asarray(rollout_batch_cpu.episode_lengths),
            global_step=int(global_step),
            name_prefix=name_prefix,
            fps=self.video_fps,
            scale=self.video_scale,
            rollout_seconds=rollout_seconds,
            transfer_seconds=transfer_seconds,
            is_final=is_final,
            env_id=getattr(self.video_rollout, "env_id", None),
            view_mode=getattr(self.video_rollout, "view_mode", None),
        )


def _create_train_states(
    *,
    policy: object,
    critic: object,
    sample_observation: jax.Array,
    loop_config: PPOFeedforwardLoopConfig,
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
    policy_state = TrainState.create(
        apply_fn=policy.apply,
        params=policy.init(policy_key, sample_observation),
        tx=optimizer,
    )
    critic_state = TrainState.create(
        apply_fn=critic.apply,
        params=critic.init(critic_key, sample_observation),
        tx=optimizer,
    )
    return policy_state, critic_state


def ppo_full_jax_trainer(
    workspace: NodeWorkspace,
    config: RLRunConfig,
    schedule: TrainingSchedule,
    *,
    env_factory: EnvFactory | None = None,
) -> None:
    """Run PPO feedforward full-JAX training inside an active node workspace."""
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
    policy, process_action = create_policy_network(config.algorithm, train_env)
    critic = create_critic_network(train_env)

    video_rollout = None
    video_recorder = None
    if config.video.record_video or config.video.record_final_video:
        workspace.videos_dir.mkdir(parents=True, exist_ok=True)
        video_recorder = AsyncNavixVideoRecorder(
            workspace.videos_dir,
            workspace.video_metrics_jsonl_path,
            max_pending_videos=config.video.max_pending_videos,
        )
        video_rollout = NavixGreedyVideoRollout(
            env_id=config.environment.env_id,
            preprocess_observation=train_env.preprocess_observation,
            policy_apply=policy.apply,
            max_steps=config.video.video_max_steps,
            view_mode=config.video.video_view_mode,
        )

    key = jax.random.PRNGKey(config.environment.seed)
    key, policy_key, critic_key, reset_key = jax.random.split(key, 4)
    reset_keys = jax.random.split(reset_key, config.environment.nr_envs)
    env_state = train_env.reset(reset_keys, False)
    bootstrap_loop_config = PPOFeedforwardLoopConfig.from_run(config, schedule, horizon=train_env.horizon)
    policy_state, critic_state = _create_train_states(
        policy=policy,
        critic=critic,
        sample_observation=env_state.next_observation,
        loop_config=bootstrap_loop_config,
        policy_key=policy_key,
        critic_key=critic_key,
    )
    restored = prepare_training_restore(
        workspace=workspace,
        config=config,
        schedule=schedule,
        policy_template=policy_state,
        critic_template=critic_state,
        key=key,
    )
    config = restored.config
    schedule = restored.schedule
    key = restored.key
    policy_state = restored.policy_state
    critic_state = restored.critic_state
    loop_config = PPOFeedforwardLoopConfig.from_run(config, schedule, horizon=train_env.horizon)

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

    train_fn = build_ppo_feedforward_train_fn(
        train_env=train_env,
        eval_env=eval_env,
        policy=policy,
        critic=critic,
        process_action=process_action,
        loop_config=loop_config,
        hyperparameters=hyperparameters,
        is_discrete=is_discrete,
        observation_shape=tuple(int(dim) for dim in train_env.single_observation_space.shape),
        action_shape=tuple(int(dim) for dim in train_env.single_action_space.shape),
        log_train_metrics=host.make_train_log_callback(),
        log_eval_metrics=host.make_eval_log_callback(),
        save_model=workspace.make_model_archive_save_callback(),
        save_checkpoint=host.make_checkpoint_save_callback(
            algorithm_name=config.algorithm.name,
            checkpoint_policy=config.checkpoint,
            total_timesteps=config.algorithm.total_timesteps,
        ),
    )
    train_fn = jax.jit(train_fn)
    log_jax_training_devices(LOGGER)
    LOGGER.info("Starting PPO full-JAX training")
    try:
        policy_state, critic_state = jax.block_until_ready(train_fn(key, policy_state, critic_state))
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
        PPOCheckpoint.build(
            algorithm_name=config.algorithm.name,
            global_step=final_step,
            optimizer_updates=final_optimizer_updates,
            rng_key=key,
            policy_state=policy_state,
            critic_state=critic_state,
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
