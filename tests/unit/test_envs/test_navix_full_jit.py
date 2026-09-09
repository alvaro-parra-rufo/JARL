"""Tests for jarl Navix full-JIT environment adapter."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jarl.envs.navix import NavixFullJITEnv, make_navix_full_jit_env
from jarl.envs.navix.reward_config import RewardWeightsConfig
from jarl.envs.types import ActionSpaceType, DataInterfaceType, ObservationSpaceType
from jarl.training.config import EnvironmentConfig, RLRunConfig
from jarl.training.env_factory import navix_full_jit_env_factory

# Compiles Navix XLA; `make test` runs slow markers serially.
pytestmark = pytest.mark.slow


@pytest.fixture()
def navix_env() -> NavixFullJITEnv:
    """Small Navix env with flattened symbolic observations."""
    return make_navix_full_jit_env(env_id="Navix-Empty-5x5-v0")


class TestNavixFullJITEnv:
    """Tests for reset, step, and preprocessing."""

    def test_reset_and_step_shapes(self) -> None:
        env = make_navix_full_jit_env(env_id="Navix-Empty-5x5-v0")
        keys = jax.random.split(jax.random.PRNGKey(0), 3)

        state = env.reset(keys, eval_mode=False)
        next_state = env.step(state, jnp.zeros((3,), dtype=jnp.int32))

        assert state.next_observation.shape[0] == 3
        assert next_state.next_observation.ndim == 2
        assert next_state.actual_next_observation.shape == next_state.next_observation.shape
        assert next_state.reward.shape == (3,)
        assert env.general_properties.observation_space_type is ObservationSpaceType.FLAT_VALUES
        assert env.general_properties.action_space_type is ActionSpaceType.DISCRETE
        assert env.general_properties.data_interface_type is DataInterfaceType.JAX

    def test_preprocess_observation_is_normalized_flat_vector(self, navix_env: NavixFullJITEnv) -> None:
        raw = jnp.full(navix_env.raw_observation_shape, 255, dtype=jnp.uint8)
        processed = navix_env.preprocess_observation(raw)

        assert processed.ndim == 1
        assert processed.dtype == jnp.float32
        assert float(jnp.max(processed)) == pytest.approx(1.0)

    def test_autoreset_clears_episode_store_after_done(self) -> None:
        env = make_navix_full_jit_env(
            env_id="Navix-Empty-5x5-v0",
            max_episode_steps=1,
        )
        keys = jax.random.split(jax.random.PRNGKey(1), 2)

        state = env.reset(keys, eval_mode=False)
        next_state = env.step(state, jnp.zeros((2,), dtype=jnp.int32))

        assert jnp.all(next_state.truncated)
        assert jnp.all(next_state.info["rollout/episode_length"] == 1)
        assert jnp.all(next_state.info_episode_store["episode_length"] == 0)

    def test_max_episode_steps_is_configurable(self) -> None:
        env = make_navix_full_jit_env(
            env_id="Navix-Empty-5x5-v0",
            max_episode_steps=1,
        )

        assert env.horizon == 1

    def test_reset_step_inside_jit_wrapper(self, navix_env: NavixFullJITEnv) -> None:
        keys = jax.random.split(jax.random.PRNGKey(2), 4)

        @jax.jit
        def run_batch(batch_keys: jax.Array) -> jax.Array:
            state = navix_env.reset(batch_keys, eval_mode=False)
            next_state = navix_env.step(state, jnp.zeros((batch_keys.shape[0],), dtype=jnp.int32))
            return next_state.next_observation

        output = run_batch(keys)

        assert output.shape[0] == 4


class TestNavixEnvFactory:
    """Tests for config-driven Navix environment construction."""

    def test_navix_full_jit_env_factory_uses_environment_block(self) -> None:
        config = RLRunConfig(
            environment=EnvironmentConfig(
                env_id="Navix-Empty-5x5-v0",
                max_episode_steps=10,
            )
        )

        env = navix_full_jit_env_factory(config)

        assert env.env_id == "Navix-Empty-5x5-v0"
        assert env.horizon == 10
        assert env.general_properties.observation_space_type is ObservationSpaceType.FLAT_VALUES

    def test_factory_applies_custom_mix_on_empty(self) -> None:
        config = RLRunConfig(
            environment=EnvironmentConfig(
                env_id="Navix-Empty-5x5-v0",
                reward=RewardWeightsConfig(),
            )
        )
        env = navix_full_jit_env_factory(config)
        keys = jax.random.split(jax.random.PRNGKey(0), 1)
        state = env.reset(keys, eval_mode=True)
        # rotate cw, forward, forward, rotate ccw, forward, forward -> goal
        actions = [1, 2, 2, 0, 2, 2]
        for action in actions:
            state = env.step(state, jnp.asarray([action], dtype=jnp.int32))

        assert float(state.reward[0]) == pytest.approx(1.0)

    def test_make_without_reward_does_not_pass_custom_fn(self) -> None:
        native = make_navix_full_jit_env(env_id="Navix-Empty-5x5-v0")
        mixed = make_navix_full_jit_env(
            env_id="Navix-Empty-5x5-v0",
            reward=RewardWeightsConfig(),
        )
        keys = jax.random.split(jax.random.PRNGKey(0), 1)
        native_state = native.reset(keys, eval_mode=True)
        mixed_state = mixed.reset(keys, eval_mode=True)
        for action in (1, 2, 2, 0, 2, 2):
            native_state = native.step(native_state, jnp.asarray([action], dtype=jnp.int32))
            mixed_state = mixed.step(mixed_state, jnp.asarray([action], dtype=jnp.int32))

        assert float(native_state.reward[0]) == pytest.approx(float(mixed_state.reward[0]))
