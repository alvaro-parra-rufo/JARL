"""Tests for PPO Flax networks and action helpers."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest
from gymnasium import spaces

from jarl.agents.ppo.action import (
    continuous_log_prob,
    discrete_log_prob,
    make_processed_action_fn,
    sample_continuous_action,
    sample_discrete_action,
)
from jarl.agents.ppo.networks import (
    ContinuousPolicy,
    Critic,
    DiscretePolicy,
    create_critic_network,
    create_policy_network,
)
from jarl.envs.navix import NavixFullJITEnv, make_navix_full_jit_env
from jarl.envs.types import ActionSpaceType, DataInterfaceType, ObservationSpaceType
from jarl.training.config import AlgorithmConfig


@dataclass
class _GeneralProperties:
    action_space_type: ActionSpaceType
    observation_space_type: ObservationSpaceType
    data_interface_type: DataInterfaceType = DataInterfaceType.JAX


@dataclass
class _ContinuousEnvStub:
    general_properties: _GeneralProperties
    single_action_space: spaces.Box
    single_observation_space: spaces.Box


@pytest.fixture()
def discrete_navix_env() -> NavixFullJITEnv:
    """Small Navix env for discrete-policy network tests."""
    return make_navix_full_jit_env(env_id="Navix-Empty-5x5-v0")


@pytest.fixture()
def continuous_env_stub() -> _ContinuousEnvStub:
    """Minimal continuous-control stub for policy/critic factory tests."""
    return _ContinuousEnvStub(
        general_properties=_GeneralProperties(
            action_space_type=ActionSpaceType.CONTINUOUS,
            observation_space_type=ObservationSpaceType.FLAT_VALUES,
        ),
        single_action_space=spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=jnp.float32),
        single_observation_space=spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=jnp.float32),
    )


class TestPPONetworkFactories:
    """Tests for environment-driven network construction."""

    @pytest.mark.slow
    def test_create_discrete_policy_and_critic_shapes(self, discrete_navix_env: NavixFullJITEnv) -> None:
        algorithm = AlgorithmConfig()
        batch_size = 4
        observations = jnp.zeros((batch_size, *discrete_navix_env.single_observation_space.shape))

        policy, process_action = create_policy_network(algorithm, discrete_navix_env)
        critic = create_critic_network(discrete_navix_env)

        policy_key, critic_key = jax.random.split(jax.random.PRNGKey(0))
        policy_params = policy.init(policy_key, observations)
        critic_params = critic.init(critic_key, observations)

        logits = policy.apply(policy_params, observations)
        values = critic.apply(critic_params, observations)
        actions = process_action(jnp.zeros((batch_size,), dtype=jnp.int32))

        assert logits.shape == (batch_size, int(discrete_navix_env.single_action_space.n))
        assert values.shape == (batch_size, 1)
        assert actions.shape == (batch_size,)

    def test_create_continuous_policy_and_critic_shapes(self, continuous_env_stub: _ContinuousEnvStub) -> None:
        algorithm = AlgorithmConfig(std_dev=0.5, action_clipping_and_rescaling=True)
        batch_size = 3
        observations = jnp.zeros((batch_size, *continuous_env_stub.single_observation_space.shape))

        policy, process_action = create_policy_network(algorithm, continuous_env_stub)
        critic = create_critic_network(continuous_env_stub)

        policy_key, critic_key = jax.random.split(jax.random.PRNGKey(1))
        policy_params = policy.init(policy_key, observations)
        critic_params = critic.init(critic_key, observations)

        mean, logstd = policy.apply(policy_params, observations)
        values = critic.apply(critic_params, observations)
        processed = process_action(jnp.full((batch_size, 2), 2.0))

        assert mean.shape == (batch_size, 2)
        assert logstd.shape == (1, 2)
        assert values.shape == (batch_size, 1)
        assert float(jnp.min(processed)) >= -1.0
        assert float(jnp.max(processed)) <= 1.0


class TestPPOActionHelpers:
    """Tests for action sampling and post-processing."""

    def test_sample_discrete_action_returns_valid_log_prob(self) -> None:
        key = jax.random.PRNGKey(2)
        logits = jnp.array([[1.0, 0.0, -1.0], [0.0, 0.0, 0.0]])

        action, log_prob = sample_discrete_action(key, logits)

        assert action.shape == (2,)
        assert log_prob.shape == (2,)
        assert jnp.allclose(log_prob, discrete_log_prob(logits, action))

    def test_sample_continuous_action_matches_gaussian_log_prob(self) -> None:
        key = jax.random.PRNGKey(3)
        mean = jnp.zeros((2, 3))
        logstd = jnp.full((1, 3), -0.5)

        action, log_prob = sample_continuous_action(key, mean, logstd)

        assert action.shape == (2, 3)
        assert log_prob.shape == (2,)
        assert jnp.allclose(log_prob, continuous_log_prob(action, mean, logstd))

    def test_make_processed_action_fn_clips_and_scales(self) -> None:
        processor = make_processed_action_fn(
            action_clipping_and_rescaling=True,
            action_low=jnp.array([-2.0, 0.0]),
            action_high=jnp.array([2.0, 4.0]),
        )

        scaled = processor(jnp.array([[2.0, -2.0], [0.0, 0.0]]))

        assert jnp.allclose(scaled[0], jnp.array([2.0, 0.0]))
        assert jnp.allclose(scaled[1], jnp.array([0.0, 2.0]))

    @pytest.mark.parametrize(
        ("policy_cls", "expected_output_rank"),
        [
            pytest.param(DiscretePolicy, 2, id="discrete"),
            pytest.param(ContinuousPolicy, 2, id="continuous-mean"),
        ],
    )
    def test_policy_modules_initialize_under_jit(
        self,
        policy_cls: type,
        expected_output_rank: int,
    ) -> None:
        if policy_cls is DiscretePolicy:
            policy = DiscretePolicy(n_actions=4, observation_indices=jnp.arange(6))
            observations = jnp.zeros((5, 6))
        else:
            policy = ContinuousPolicy(action_shape=(2,), std_dev=1.0, observation_indices=jnp.arange(6))
            observations = jnp.zeros((5, 6))

        @jax.jit
        def forward(params: object, obs: jax.Array) -> jax.Array:
            output = policy.apply(params, obs)
            if isinstance(output, tuple):
                return output[0]
            return output

        params = policy.init(jax.random.PRNGKey(4), observations)
        output = forward(params, observations)

        assert output.ndim == expected_output_rank

    def test_critic_module_initializes_under_jit(self) -> None:
        critic = Critic(observation_indices=jnp.arange(6))
        observations = jnp.zeros((5, 6))

        @jax.jit
        def forward(params: object, obs: jax.Array) -> jax.Array:
            return critic.apply(params, obs)

        params = critic.init(jax.random.PRNGKey(5), observations)
        values = forward(params, observations)

        assert values.shape == (5, 1)
