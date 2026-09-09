"""Tests for PPO inference policy loaders."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pytest_mock import MockerFixture

from jarl.agents.ppo.checkpoint_state import (
    CHECKPOINT_KIND_PPO,
    CHECKPOINT_KIND_PPO_GRU,
    PPOCheckpoint,
    PPOGrUCheckpoint,
)
from jarl.agents.ppo.inference.backend import (
    load_ppo_gru_inference_policy,
    load_ppo_inference_policy,
)
from jarl.agents.ppo.networks import create_gru_policy_network, create_policy_network
from jarl.experiments.checkpoint_state import TrainStateLeaf
from jarl.experiments.node import NodeWorkspace
from jarl.training.config import AlgorithmConfig, EnvironmentConfig, RLRunConfig
from jarl.training.env_factory import navix_full_jit_env_factory

# Compiles Navix XLA; `make test` runs slow markers serially.
pytestmark = pytest.mark.slow


class TestPPOInferenceBackend:
    """Feed-forward checkpoints become reusable greedy runtimes."""

    def test_load_policy_uses_checkpoint_params_and_matches_greedy_action(self, mocker: MockerFixture) -> None:
        config = _config(CHECKPOINT_KIND_PPO)
        env = navix_full_jit_env_factory(config)
        observation = _sample_observation(env)
        policy, _process_action = create_policy_network(config.algorithm, env)
        params = policy.init(jax.random.PRNGKey(3), observation)
        checkpoint = _ppo_checkpoint(params)
        workspace = _workspace(mocker, checkpoint)

        loaded = load_ppo_inference_policy(workspace, config, 12, env)
        policy_state = loaded.runtime.initial_state(1)
        action, next_policy_state = loaded.runtime.greedy_step(
            loaded.params,
            observation,
            policy_state,
        )

        expected_action = jnp.argmax(policy.apply(params, observation), axis=-1).astype(jnp.int32)
        np.testing.assert_array_equal(action, expected_action)
        assert policy_state == ()
        assert next_policy_state == ()
        assert loaded.params is params
        assert loaded.source_node_id == "node_a"
        assert loaded.checkpoint_step == 12
        assert loaded.global_step == 64
        assert loaded.optimizer_updates == 9
        workspace.load_training_checkpoint.assert_called_once_with(PPOCheckpoint, step=12)

    def test_runtime_preprocessing_matches_environment(self, mocker: MockerFixture) -> None:
        config = _config(CHECKPOINT_KIND_PPO)
        env = navix_full_jit_env_factory(config)
        observation = _sample_observation(env)
        policy, _process_action = create_policy_network(config.algorithm, env)
        checkpoint = _ppo_checkpoint(policy.init(jax.random.PRNGKey(5), observation))
        workspace = _workspace(mocker, checkpoint)
        raw_observation = jnp.full((1, *env.raw_observation_shape), 127, dtype=jnp.uint8)

        loaded = load_ppo_inference_policy(workspace, config, 12, env)
        actual = loaded.runtime.preprocess_observation(raw_observation)

        np.testing.assert_allclose(actual, env.preprocess_observation(raw_observation))


class TestPPOGrUInferenceBackend:
    """Recurrent checkpoints start each inference episode with fresh carry."""

    def test_load_policy_ignores_training_carry_and_matches_greedy_action(self, mocker: MockerFixture) -> None:
        config = _config(CHECKPOINT_KIND_PPO_GRU)
        env = navix_full_jit_env_factory(config)
        observation = _sample_observation(env)
        policy, _process_action = create_gru_policy_network(config.algorithm, env)
        initial_carry = policy.initialize_carry(1)
        params = policy.init(
            jax.random.PRNGKey(7),
            observation,
            initial_carry,
            method=policy.apply_one_step,
        )
        checkpoint = _ppo_gru_checkpoint(params, jnp.ones_like(initial_carry))
        workspace = _workspace(mocker, checkpoint)

        loaded = load_ppo_gru_inference_policy(workspace, config, 21, env)
        policy_state = loaded.runtime.initial_state(1)
        action, next_policy_state = loaded.runtime.greedy_step(
            loaded.params,
            observation,
            policy_state,
        )

        logits, expected_next_state = policy.apply(
            params,
            observation,
            initial_carry,
            method=policy.apply_one_step,
        )
        expected_action = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        np.testing.assert_array_equal(policy_state, initial_carry)
        np.testing.assert_array_equal(action, expected_action)
        np.testing.assert_allclose(next_policy_state, expected_next_state)
        assert loaded.params is params
        assert loaded.checkpoint_kind == CHECKPOINT_KIND_PPO_GRU
        workspace.load_training_checkpoint.assert_called_once_with(PPOGrUCheckpoint, step=21)


def _config(algorithm_name: str) -> RLRunConfig:
    return RLRunConfig(
        environment=EnvironmentConfig(
            env_id="Navix-Empty-5x5-v0",
            nr_envs=1,
            max_episode_steps=8,
        ),
        algorithm=AlgorithmConfig(
            name=algorithm_name,
            total_timesteps=1,
            nr_steps=1,
            minibatch_size=1,
        ),
    )


def _sample_observation(env: object) -> jax.Array:
    raw_shape = env.raw_observation_shape
    raw = jnp.zeros((1, *raw_shape), dtype=jnp.uint8)
    return env.preprocess_observation(raw)


def _workspace(
    mocker: MockerFixture,
    checkpoint: PPOCheckpoint | PPOGrUCheckpoint,
) -> NodeWorkspace:
    workspace = mocker.Mock(spec=NodeWorkspace)
    workspace.id = "node_a"
    workspace.load_training_checkpoint.return_value = checkpoint
    return workspace


def _ppo_checkpoint(params: object) -> PPOCheckpoint:
    return PPOCheckpoint(
        version=1,
        kind=CHECKPOINT_KIND_PPO,
        algorithm_name=CHECKPOINT_KIND_PPO,
        global_step=64,
        optimizer_updates=9,
        rng_key=(0, 1),
        policy=TrainStateLeaf(step=9, params=params, opt_state={"unused": True}),
        critic=TrainStateLeaf(step=9, params={}, opt_state={"unused": True}),
    )


def _ppo_gru_checkpoint(
    params: object,
    policy_carry: jax.Array,
) -> PPOGrUCheckpoint:
    return PPOGrUCheckpoint(
        version=1,
        kind=CHECKPOINT_KIND_PPO_GRU,
        algorithm_name=CHECKPOINT_KIND_PPO_GRU,
        global_step=96,
        optimizer_updates=11,
        rng_key=(0, 2),
        policy=TrainStateLeaf(step=11, params=params, opt_state={"unused": True}),
        critic=TrainStateLeaf(step=11, params={}, opt_state={"unused": True}),
        policy_carry=policy_carry,
    )
