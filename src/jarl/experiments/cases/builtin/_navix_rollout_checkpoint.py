"""Shared helpers for uninitialized Navix rollout checkpoint experiment cases."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from jarl.agents.ppo.checkpoint_state import (
    CHECKPOINT_KIND_PPO,
    CHECKPOINT_KIND_PPO_GRU,
    PPOCheckpoint,
    PPOGrUCheckpoint,
)
from jarl.agents.ppo.networks import (
    create_critic_network,
    create_gru_critic_network,
    create_gru_policy_network,
    create_policy_network,
)
from jarl.experiments.cases.case import ExperimentCaseContext
from jarl.experiments.graph import ExperimentGraph
from jarl.operations.graph.create_root import CreateRootRequest, create_root
from jarl.training.config import AlgorithmConfig, EnvironmentConfig, RLRunConfig, VideoConfig
from jarl.training.env_factory import navix_full_jit_env_factory

__all__ = [
    "ROLLOUT_CHECKPOINT_STEP",
    "ROLLOUT_ENV_ID",
    "ROLLOUT_GLOBAL_STEP",
    "ROLLOUT_INIT_SEED",
    "ROLLOUT_MAX_STEPS",
    "ROLLOUT_OPTIMIZER_UPDATES",
    "ROLLOUT_SEED",
    "ROLLOUT_TRAINED_CHECKPOINT_EVAL_FREQUENCY",
    "ROLLOUT_TRAINED_ENV_ID",
    "ROLLOUT_TRAINED_MAX_STEPS",
    "build_ppo_checkpoint",
    "build_ppo_gru_checkpoint",
    "materialize_untrained_rollout_case",
    "minimal_rollout_run_config",
    "untrained_rollout_run_config",
]

ROLLOUT_CHECKPOINT_STEP = 8
"""Node-relative checkpoint step saved by untrained rollout experiment cases."""

ROLLOUT_GLOBAL_STEP = 128
"""Global step recorded on the synthetic checkpoint."""

ROLLOUT_OPTIMIZER_UPDATES = 4
"""Optimizer update counter recorded on the synthetic checkpoint."""

ROLLOUT_INIT_SEED = 0
"""Seed used to initialize policy parameters during materialization."""

ROLLOUT_SEED = 7
"""Default rollout episode seed used by agentic and integration tests."""

ROLLOUT_ENV_ID = "Navix-Empty-5x5-v0"
"""Small Navix map used by fast CPU rollout integration cases."""

ROLLOUT_MAX_STEPS = 8
"""Short episode horizon for fast 5x5 rollout integration."""

ROLLOUT_TRAINED_ENV_ID = "Navix-Empty-5x5-v0"
"""Empty map used by trained rollout agentic scenarios."""

ROLLOUT_TRAINED_MAX_STEPS = 16
"""Episode horizon for Empty-5x5 rollout analysis."""

ROLLOUT_TRAINED_CHECKPOINT_EVAL_FREQUENCY = 4_096
"""Environment steps between eval and checkpoint saves in trained rollout scenarios."""


def untrained_rollout_run_config(
    *,
    env_id: str,
    max_episode_steps: int,
    algorithm_name: str = CHECKPOINT_KIND_PPO,
    nr_envs: int = 4,
) -> RLRunConfig:
    """Return a tiny Navix config for uninitialized checkpoint materialization."""
    algorithm = AlgorithmConfig(
        name=algorithm_name,
        total_timesteps=512,
        nr_steps=8,
        minibatch_size=32,
        nr_epochs=1,
    )
    if algorithm_name == CHECKPOINT_KIND_PPO_GRU:
        algorithm = algorithm.model_copy(
            update={
                "obs_encoding_dim": 16,
                "gru_hidden_dim": 8,
            }
        )
    return RLRunConfig(
        environment=EnvironmentConfig(
            env_id=env_id,
            nr_envs=nr_envs,
            seed=ROLLOUT_INIT_SEED,
            max_episode_steps=max_episode_steps,
        ),
        algorithm=algorithm,
        video=VideoConfig(record_video=False, record_final_video=False),
    )


def minimal_rollout_run_config(*, algorithm_name: str = CHECKPOINT_KIND_PPO) -> RLRunConfig:
    """Return the fast 5x5 config used by CPU rollout integration cases."""
    return untrained_rollout_run_config(
        env_id=ROLLOUT_ENV_ID,
        max_episode_steps=ROLLOUT_MAX_STEPS,
        algorithm_name=algorithm_name,
    )


def build_ppo_checkpoint(
    config: RLRunConfig,
    *,
    init_seed: int = ROLLOUT_INIT_SEED,
) -> PPOCheckpoint:
    """Build one feed-forward PPO checkpoint without running training."""
    env = navix_full_jit_env_factory(config)
    policy, _process_action = create_policy_network(config.algorithm, env)
    del _process_action
    critic = create_critic_network(env)
    key = jax.random.PRNGKey(init_seed)
    policy_key, critic_key = jax.random.split(key)
    observation = _preprocessed_observation(env)
    policy_state = _empty_train_state(policy.init(policy_key, observation))
    critic_state = _empty_train_state(critic.init(critic_key, observation))
    return PPOCheckpoint.build(
        algorithm_name=CHECKPOINT_KIND_PPO,
        global_step=ROLLOUT_GLOBAL_STEP,
        optimizer_updates=ROLLOUT_OPTIMIZER_UPDATES,
        rng_key=key,
        policy_state=policy_state,
        critic_state=critic_state,
    )


def build_ppo_gru_checkpoint(
    config: RLRunConfig,
    *,
    init_seed: int = ROLLOUT_INIT_SEED,
) -> PPOGrUCheckpoint:
    """Build one recurrent PPO checkpoint without running training."""
    env = navix_full_jit_env_factory(config)
    policy, _process_action = create_gru_policy_network(config.algorithm, env)
    del _process_action
    critic = create_gru_critic_network(config.algorithm, env)
    key = jax.random.PRNGKey(init_seed)
    policy_key, critic_key = jax.random.split(key)
    observation = _preprocessed_observation(env)
    policy_carry = policy.initialize_carry(1)
    critic_carry = critic.initialize_carry(1)
    policy_state = _empty_train_state(
        policy.init(
            policy_key,
            observation,
            policy_carry,
            method=policy.apply_one_step,
        )
    )
    critic_state = _empty_train_state(
        critic.init(
            critic_key,
            observation,
            critic_carry,
            method=critic.apply_one_step,
        )
    )
    return PPOGrUCheckpoint.build(
        algorithm_name=CHECKPOINT_KIND_PPO_GRU,
        global_step=ROLLOUT_GLOBAL_STEP,
        optimizer_updates=ROLLOUT_OPTIMIZER_UPDATES,
        rng_key=key,
        policy_state=policy_state,
        critic_state=critic_state,
        policy_carry=policy_carry,
        critic_carry=critic_carry,
    )


def materialize_untrained_rollout_case(
    context: ExperimentCaseContext[RLRunConfig],
    *,
    config: RLRunConfig,
    algorithm_name: str,
    label: str,
) -> None:
    """Create a prepared root and persist one uninitialized typed checkpoint."""
    graph = ExperimentGraph(context.experiment_dir, base_config=config)
    response = create_root(
        graph,
        CreateRootRequest(
            label=label,
            branch="main",
            config=config,
        ),
    )
    workspace = graph.get_node(response.node_id)
    workspace.init_checkpoint_manager(max_to_keep=None, save_interval_steps=1)
    checkpoint = (
        build_ppo_gru_checkpoint(config) if algorithm_name == CHECKPOINT_KIND_PPO_GRU else build_ppo_checkpoint(config)
    )
    workspace.save_training_checkpoint(
        ROLLOUT_CHECKPOINT_STEP,
        checkpoint,
        force=True,
    )
    graph.save()
    context.register_alias("root", response.node_id)


def _preprocessed_observation(env: object) -> jax.Array:
    raw_shape = env.raw_observation_shape
    raw = jnp.zeros((1, *raw_shape), dtype=jnp.uint8)
    return env.preprocess_observation(raw)


def _empty_train_state(params: object) -> TrainState:
    return TrainState.create(
        apply_fn=lambda *_args, **_kwargs: None,
        params=params,
        tx=optax.adam(1e-3),
    )
