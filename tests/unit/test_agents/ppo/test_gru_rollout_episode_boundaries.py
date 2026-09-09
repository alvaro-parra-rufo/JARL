"""GRU rollout integration tests for carry reset and GAE truncation masks."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jarl.agents.ppo.core.gae import compute_gae_advantages
from jarl.agents.ppo.networks import create_gru_policy_network
from jarl.envs.navix import NavixFullJITEnv, make_navix_full_jit_env
from jarl.training.config import AlgorithmConfig
from tests.helpers.ppo_gae_reference import (
    reference_gae_leaky_terminations_only,
    reference_gae_with_episode_masks,
)
from tests.helpers.ppo_rollout_helpers import collect_navix_rollout_masks


def _run_gru_rollout_carry_trace(
    *,
    env: NavixFullJITEnv,
    policy: object,
    policy_params: object,
    nr_envs: int,
    nr_steps: int,
    seed: int,
) -> tuple[jax.Array, jax.Array]:
    """Mirror ``gru_loop`` carry reset and return per-step carries plus ``done`` flags."""
    reset_keys = jax.random.split(jax.random.PRNGKey(seed), nr_envs)
    env_state = env.reset(reset_keys, eval_mode=False)
    policy_carry = policy.initialize_carry(nr_envs)

    def rollout_step(
        carry: tuple[object, jax.Array],
        _unused: None,
    ) -> tuple[tuple[object, jax.Array], tuple[jax.Array, jax.Array]]:
        env_state, policy_carry = carry
        observation = env_state.next_observation
        _, next_policy_carry = policy.apply(
            policy_params,
            observation,
            policy_carry,
            method=policy.apply_one_step,
        )
        env_state = env.step(env_state, jnp.zeros((nr_envs,), dtype=jnp.int32))
        done = env_state.terminated | env_state.truncated
        next_policy_carry = next_policy_carry * (1.0 - done[:, None])
        return (env_state, next_policy_carry), (done, next_policy_carry)

    _, batch = jax.lax.scan(
        rollout_step,
        (env_state, policy_carry),
        None,
        nr_steps,
    )
    dones, carries = batch
    return dones, carries


class TestGrURolloutEpisodeBoundaries:
    """GRU rollouts must reset carry and pass truncation masks into GAE."""

    @pytest.mark.slow
    def test_navix_rollout_records_truncations_without_terminal_flag(self) -> None:
        terminations, truncations, dones = collect_navix_rollout_masks(
            max_episode_steps=2,
            nr_envs=4,
            nr_steps=8,
            seed=0,
        )

        assert jnp.any(truncations)
        assert jnp.any(dones)
        assert jnp.any(truncations & jnp.logical_not(terminations))

    def test_gru_rollout_gae_uses_truncation_mask_in_time_major_batch(self) -> None:
        """Rollout batch layout ``[T, nr_envs]`` must call GAE with ``truncations``."""
        rewards = jnp.array([[0.0, 0.0], [0.0, 0.0], [100.0, 1.0]])
        values = jnp.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        next_values = jnp.array([[0.0, 0.0], [5.0, 0.0], [0.0, 0.0]])
        terminations = jnp.array([[False, False], [False, False], [False, False]])
        truncations = jnp.array([[False, False], [True, False], [False, False]])

        advantages, _ = compute_gae_advantages(
            rewards,
            values,
            next_values,
            terminations,
            truncations=truncations,
            gamma=1.0,
            gae_lambda=1.0,
        )
        expected_advantages, _ = reference_gae_with_episode_masks(
            rewards,
            values,
            next_values,
            terminations,
            truncations,
            gamma=1.0,
            gae_lambda=1.0,
        )
        leaky_advantages = reference_gae_leaky_terminations_only(
            rewards,
            values,
            next_values,
            terminations,
            gamma=1.0,
            gae_lambda=1.0,
        )

        leaky_truncated_advantage = 105.0

        assert jnp.allclose(advantages, expected_advantages, atol=1e-6)
        assert float(advantages[1, 0]) == pytest.approx(5.0, rel=1e-5)
        assert float(leaky_advantages[1, 0]) == pytest.approx(leaky_truncated_advantage, rel=1e-5)
        assert float(advantages[1, 0]) != pytest.approx(leaky_truncated_advantage, rel=1e-3)

    @pytest.mark.slow
    def test_gru_rollout_carry_zeroed_after_done(self) -> None:
        env = make_navix_full_jit_env(
            env_id="Navix-Empty-5x5-v0",
            max_episode_steps=2,
        )
        algorithm = AlgorithmConfig(obs_encoding_dim=16, gru_hidden_dim=8)
        policy, _ = create_gru_policy_network(algorithm, env)
        observations = jnp.zeros((2, *env.single_observation_space.shape))
        init_carry = policy.initialize_carry(2)
        policy_params = policy.init(
            jax.random.PRNGKey(4),
            observations,
            init_carry,
            method=policy.apply_one_step,
        )

        dones, carries = _run_gru_rollout_carry_trace(
            env=env,
            policy=policy,
            policy_params=policy_params,
            nr_envs=2,
            nr_steps=6,
            seed=1,
        )

        assert jnp.any(dones)
        done_positions = jnp.argwhere(dones)
        for time_index, env_index in done_positions:
            carry_row = carries[time_index, env_index]
            assert jnp.allclose(carry_row, 0.0, atol=1e-6)

    @pytest.mark.slow
    def test_gru_forward_sequence_matches_rollout_carry_reset(self) -> None:
        env = make_navix_full_jit_env(
            env_id="Navix-Empty-5x5-v0",
            max_episode_steps=1,
        )
        algorithm = AlgorithmConfig(obs_encoding_dim=16, gru_hidden_dim=8)
        policy, _ = create_gru_policy_network(algorithm, env)
        nr_envs = 1
        nr_steps = 3
        observations = jnp.zeros((nr_envs, *env.single_observation_space.shape))
        init_carry = policy.initialize_carry(nr_envs)
        policy_params = policy.init(
            jax.random.PRNGKey(5),
            observations,
            init_carry,
            method=policy.apply_one_step,
        )

        dones, _ = _run_gru_rollout_carry_trace(
            env=env,
            policy=policy,
            policy_params=policy_params,
            nr_envs=nr_envs,
            nr_steps=nr_steps,
            seed=2,
        )
        assert bool(dones[0, 0])

        obs_seq = jnp.zeros((nr_steps, int(env.single_observation_space.shape[0])))
        done_seq = dones[:, 0]
        rollout_logits = policy.apply(
            policy_params,
            obs_seq,
            done_seq,
            init_carry[0],
            method=policy.forward_sequence,
        )
        fresh_logits, _ = policy.apply(
            policy_params,
            obs_seq[2],
            jnp.zeros((algorithm.gru_hidden_dim,)),
            method=policy.apply_one_step,
        )

        assert jnp.allclose(rollout_logits[2], fresh_logits[0], atol=1e-5)
