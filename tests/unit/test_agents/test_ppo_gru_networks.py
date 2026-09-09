"""Tests for PPO-GRU Flax networks and factories."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jarl.agents.ppo.networks import (
    ContinuousGrUPolicy,
    DiscreteGrUPolicy,
    GrUCritic,
    create_gru_critic_network,
    create_gru_policy_network,
)
from jarl.envs.navix import NavixFullJITEnv, make_navix_full_jit_env
from jarl.training.config import AlgorithmConfig

# Compiles Navix XLA; `make test` runs slow markers serially.
pytestmark = pytest.mark.slow


@pytest.fixture()
def discrete_navix_env() -> NavixFullJITEnv:
    """Small Navix env for GRU policy network tests."""
    return make_navix_full_jit_env(env_id="Navix-Empty-5x5-v0")


class TestPPOGrUNetworkFactories:
    """Tests for environment-driven GRU network construction."""

    def test_create_discrete_gru_policy_shapes(self, discrete_navix_env: NavixFullJITEnv) -> None:
        algorithm = AlgorithmConfig(
            obs_encoding_dim=32,
            gru_hidden_dim=16,
            gru_obs_combine_method="concat",
            share_gru_obs_encoder=False,
        )
        batch_size = 4
        observations = jnp.zeros((batch_size, *discrete_navix_env.single_observation_space.shape))

        policy, process_action = create_gru_policy_network(algorithm, discrete_navix_env)
        carry = policy.initialize_carry(batch_size)

        policy_key = jax.random.PRNGKey(0)
        policy_params = policy.init(
            policy_key,
            observations,
            carry,
            method=policy.apply_one_step,
        )

        logits, next_carry = policy.apply(
            policy_params,
            observations,
            carry,
            method=policy.apply_one_step,
        )
        actions = process_action(jnp.zeros((batch_size,), dtype=jnp.int32))

        assert isinstance(policy, DiscreteGrUPolicy)
        assert logits.shape == (batch_size, int(discrete_navix_env.single_action_space.n))
        assert next_carry.shape == (batch_size, algorithm.gru_hidden_dim)
        assert actions.shape == (batch_size,)

    def test_discrete_gru_forward_sequence_under_jit(self, discrete_navix_env: NavixFullJITEnv) -> None:
        algorithm = AlgorithmConfig(obs_encoding_dim=16, gru_hidden_dim=8)
        policy, _ = create_gru_policy_network(algorithm, discrete_navix_env)
        nr_steps = 5
        obs_dim = int(discrete_navix_env.single_observation_space.shape[0])
        obs_seq = jnp.zeros((nr_steps, obs_dim))
        done_seq = jnp.zeros((nr_steps,), dtype=jnp.bool_)
        init_carry = policy.initialize_carry(1)[0]

        @jax.jit
        def forward(params: object) -> jax.Array:
            return policy.apply(
                params,
                obs_seq,
                done_seq,
                init_carry,
                method=policy.forward_sequence,
            )

        params = policy.init(
            jax.random.PRNGKey(1),
            obs_seq[0],
            init_carry,
            method=policy.apply_one_step,
        )
        logits_seq = forward(params)

        assert logits_seq.shape == (nr_steps, int(discrete_navix_env.single_action_space.n))

    @pytest.mark.parametrize("combine_method", ["concat", "film"])
    def test_gru_combine_methods_initialize(self, discrete_navix_env: NavixFullJITEnv, combine_method: str) -> None:
        algorithm = AlgorithmConfig(
            obs_encoding_dim=16,
            gru_hidden_dim=8,
            gru_obs_combine_method=combine_method,  # type: ignore[arg-type]
        )
        policy, _ = create_gru_policy_network(algorithm, discrete_navix_env)
        observations = jnp.zeros((2, *discrete_navix_env.single_observation_space.shape))
        carry = policy.initialize_carry(2)
        params = policy.init(
            jax.random.PRNGKey(2),
            observations,
            carry,
            method=policy.apply_one_step,
        )
        logits, _ = policy.apply(params, observations, carry, method=policy.apply_one_step)
        assert logits.shape[0] == 2

    def test_continuous_gru_policy_apply_one_step(self) -> None:
        policy = ContinuousGrUPolicy(
            action_shape=(2,),
            std_dev=0.5,
            obs_encoding_dim=16,
            gru_hidden_dim=8,
            gru_obs_combine_method="concat",
            share_gru_obs_encoder=True,
            observation_indices=jnp.arange(6),
        )
        observations = jnp.zeros((3, 6))
        carry = policy.initialize_carry(3)
        params = policy.init(jax.random.PRNGKey(3), observations, carry, method=policy.apply_one_step)
        mean, logstd, next_carry = policy.apply(params, observations, carry, method=policy.apply_one_step)

        assert mean.shape == (3, 2)
        assert logstd.shape == (1, 2)
        assert next_carry.shape == (3, 8)


class TestPPOGrUCritic:
    """Tests for the recurrent GRU critic."""

    def test_create_gru_critic_shapes(self, discrete_navix_env: NavixFullJITEnv) -> None:
        algorithm = AlgorithmConfig(
            obs_encoding_dim=32,
            gru_hidden_dim=16,
            gru_obs_combine_method="concat",
            share_gru_obs_encoder=False,
        )
        batch_size = 4
        observations = jnp.zeros((batch_size, *discrete_navix_env.single_observation_space.shape))
        critic = create_gru_critic_network(algorithm, discrete_navix_env)
        carry = critic.initialize_carry(batch_size)
        params = critic.init(
            jax.random.PRNGKey(4),
            observations,
            carry,
            method=critic.apply_one_step,
        )
        values, next_carry = critic.apply(
            params,
            observations,
            carry,
            method=critic.apply_one_step,
        )

        assert isinstance(critic, GrUCritic)
        assert values.shape == (batch_size, 1)
        assert next_carry.shape == (batch_size, algorithm.gru_hidden_dim)

    def test_gru_critic_forward_sequence_under_jit(self, discrete_navix_env: NavixFullJITEnv) -> None:
        algorithm = AlgorithmConfig(obs_encoding_dim=16, gru_hidden_dim=8)
        critic = create_gru_critic_network(algorithm, discrete_navix_env)
        nr_steps = 5
        obs_dim = int(discrete_navix_env.single_observation_space.shape[0])
        obs_seq = jnp.zeros((nr_steps, obs_dim))
        done_seq = jnp.zeros((nr_steps,), dtype=jnp.bool_)
        init_carry = critic.initialize_carry(1)[0]

        @jax.jit
        def forward(params: object) -> jax.Array:
            return critic.apply(
                params,
                obs_seq,
                done_seq,
                init_carry,
                method=critic.forward_sequence,
            )

        params = critic.init(
            jax.random.PRNGKey(5),
            obs_seq[0],
            init_carry,
            method=critic.apply_one_step,
        )
        value_seq = forward(params)

        assert value_seq.shape == (nr_steps, 1)
