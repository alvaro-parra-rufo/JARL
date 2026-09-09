"""Tests for typed PPO training checkpoint payloads."""

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
    checkpoint_from_pytree,
)
from jarl.envs.navix.reward_config import RewardWeightsConfig
from jarl.training.config import EnvironmentConfig
from jarl.training.env_runtime import env_io_contract_unchanged
from tests.helpers.checkpoint_restore import orbax_like_opt_state, train_state_with_lr_schedule


def _train_state(params: dict[str, jnp.ndarray], *, step: int = 0) -> TrainState:
    optimizer = optax.adam(1e-3)
    return TrainState.create(apply_fn=lambda *_args, **_kwargs: None, params=params, tx=optimizer).replace(
        step=jnp.asarray(step, dtype=jnp.int32),
    )


class TestPPOCheckpoint:
    """Round-trip tests for feedforward PPO checkpoints."""

    def test_round_trip_preserves_train_state_leaves(self) -> None:
        key = jax.random.PRNGKey(7)
        policy_state = _train_state({"policy": jnp.array([1.0, 2.0])}, step=3)
        critic_state = _train_state({"critic": jnp.array([0.5])}, step=4)
        checkpoint = PPOCheckpoint.build(
            algorithm_name=CHECKPOINT_KIND_PPO,
            global_step=128,
            optimizer_updates=16,
            rng_key=key,
            policy_state=policy_state,
            critic_state=critic_state,
        )

        restored_tree = checkpoint.to_pytree()
        loaded = PPOCheckpoint.from_pytree(restored_tree)
        policy_template = _train_state({"policy": jnp.array([0.0, 0.0])})
        critic_template = _train_state({"critic": jnp.array([0.0])})
        restored_key, restored_policy, restored_critic = loaded.restore_train_states(
            policy_template=policy_template,
            critic_template=critic_template,
        )

        assert loaded.kind == CHECKPOINT_KIND_PPO
        assert loaded.global_step == 128
        assert loaded.optimizer_updates == 16
        assert tuple(int(value) for value in jax.device_get(jax.random.key_data(restored_key)).reshape(-1)) == (0, 7)
        assert jax.tree_util.tree_all(
            jax.tree.map(lambda a, b: jnp.allclose(a, b), restored_policy.params, policy_state.params)
        )
        assert jax.tree_util.tree_all(
            jax.tree.map(lambda a, b: jnp.allclose(a, b), restored_critic.params, critic_state.params)
        )
        assert int(restored_policy.step) == 3
        assert int(restored_critic.step) == 4

    def test_dispatch_from_pytree(self) -> None:
        checkpoint = PPOCheckpoint.build(
            algorithm_name=CHECKPOINT_KIND_PPO,
            global_step=1,
            optimizer_updates=1,
            rng_key=jax.random.PRNGKey(0),
            policy_state=_train_state({"policy": jnp.array([1.0])}),
            critic_state=_train_state({"critic": jnp.array([2.0])}),
        )
        loaded = checkpoint_from_pytree(checkpoint.to_pytree())
        assert isinstance(loaded, PPOCheckpoint)


class TestOrbaxOptStateRestore:
    """Orbax dict opt_state must rebuild into optax structures before training resumes."""

    def test_restore_train_states_rebuilds_inject_hyperparams_after_orbax(self) -> None:
        policy_state = train_state_with_lr_schedule({"policy": jnp.array([1.0, 2.0])}, step=3)
        critic_state = train_state_with_lr_schedule({"critic": jnp.array([0.5])}, step=4)
        checkpoint = PPOCheckpoint.build(
            algorithm_name=CHECKPOINT_KIND_PPO,
            global_step=128,
            optimizer_updates=16,
            rng_key=jax.random.PRNGKey(7),
            policy_state=policy_state,
            critic_state=critic_state,
        )
        policy_template = train_state_with_lr_schedule({"policy": jnp.array([0.0, 0.0])})
        critic_template = train_state_with_lr_schedule({"critic": jnp.array([0.0])})
        orbax_tree = checkpoint.to_pytree()
        orbax_tree["policy_opt_state"] = orbax_like_opt_state(checkpoint.policy.opt_state)
        orbax_tree["critic_opt_state"] = orbax_like_opt_state(checkpoint.critic.opt_state)
        loaded = PPOCheckpoint.from_pytree(orbax_tree)
        restored_policy, restored_critic = loaded.restore_train_states(
            policy_template=policy_template,
            critic_template=critic_template,
        )[1:]

        grads = jax.tree.map(jnp.zeros_like, restored_policy.params)
        restored_policy.apply_gradients(grads=grads)
        critic_grads = jax.tree.map(jnp.zeros_like, restored_critic.params)
        restored_critic.apply_gradients(grads=critic_grads)

        assert hasattr(restored_policy.opt_state[1], "hyperparams")
        assert hasattr(restored_critic.opt_state[1], "hyperparams")


class TestPPOGrUCheckpoint:
    """GRU checkpoints restore weights and always reset episodic hidden state."""

    def test_restore_discards_stored_carry_and_keeps_params(self) -> None:
        policy_carry = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        critic_carry = jnp.array([[5.0, 6.0], [7.0, 8.0]])
        policy_state = _train_state({"policy": jnp.array([1.0])}, step=3)
        critic_state = _train_state({"critic": jnp.array([2.0])}, step=4)
        checkpoint = PPOGrUCheckpoint.build(
            algorithm_name=CHECKPOINT_KIND_PPO_GRU,
            global_step=64,
            optimizer_updates=8,
            rng_key=jax.random.PRNGKey(3),
            policy_state=policy_state,
            critic_state=critic_state,
            policy_carry=policy_carry,
            critic_carry=critic_carry,
        )

        loaded = PPOGrUCheckpoint.from_pytree(checkpoint.to_pytree())
        zero_carry = jnp.zeros_like(policy_carry)
        _, restored_policy, restored_critic, restored_policy_carry, restored_critic_carry = loaded.restore_train_states(
            policy_template=_train_state({"policy": jnp.array([0.0])}),
            critic_template=_train_state({"critic": jnp.array([0.0])}),
            zero_carry=zero_carry,
        )

        assert jnp.allclose(restored_policy_carry, zero_carry)
        assert jnp.allclose(restored_critic_carry, zero_carry)
        assert jax.tree_util.tree_all(
            jax.tree.map(lambda a, b: jnp.allclose(a, b), restored_policy.params, policy_state.params)
        )
        assert jax.tree_util.tree_all(
            jax.tree.map(lambda a, b: jnp.allclose(a, b), restored_critic.params, critic_state.params)
        )
        assert int(restored_policy.step) == 3
        assert int(restored_critic.step) == 4

    def test_restore_rebuilds_optimizer_and_zeros_carry(self) -> None:
        policy_state = train_state_with_lr_schedule({"gru": jnp.array([1.0, 2.0])}, step=3)
        critic_state = train_state_with_lr_schedule({"gru": jnp.array([0.5])}, step=4)
        live_carry = jnp.array([[9.0, 8.0]])
        checkpoint = PPOGrUCheckpoint.build(
            algorithm_name=CHECKPOINT_KIND_PPO_GRU,
            global_step=128,
            optimizer_updates=16,
            rng_key=jax.random.PRNGKey(7),
            policy_state=policy_state,
            critic_state=critic_state,
            policy_carry=live_carry,
            critic_carry=live_carry,
        )
        policy_template = train_state_with_lr_schedule({"gru": jnp.array([0.0, 0.0])})
        critic_template = train_state_with_lr_schedule({"gru": jnp.array([0.0])})
        orbax_tree = checkpoint.to_pytree()
        orbax_tree["policy_opt_state"] = orbax_like_opt_state(checkpoint.policy.opt_state)
        orbax_tree["critic_opt_state"] = orbax_like_opt_state(checkpoint.critic.opt_state)
        loaded = PPOGrUCheckpoint.from_pytree(orbax_tree)
        zero_carry = jnp.zeros_like(live_carry)
        _, restored_policy, restored_critic, restored_policy_carry, restored_critic_carry = loaded.restore_train_states(
            policy_template=policy_template,
            critic_template=critic_template,
            zero_carry=zero_carry,
        )

        grads = jax.tree.map(jnp.zeros_like, restored_policy.params)
        restored_policy.apply_gradients(grads=grads)
        critic_grads = jax.tree.map(jnp.zeros_like, restored_critic.params)
        restored_critic.apply_gradients(grads=critic_grads)

        assert hasattr(restored_policy.opt_state[1], "hyperparams")
        assert hasattr(restored_critic.opt_state[1], "hyperparams")
        assert jax.tree_util.tree_all(
            jax.tree.map(lambda a, b: jnp.allclose(a, b), restored_policy.params, policy_state.params)
        )
        assert jnp.allclose(restored_policy_carry, zero_carry)
        assert jnp.allclose(restored_critic_carry, zero_carry)


class TestEnvIoContractUnchanged:
    """Tests for environment I/O fingerprint comparison."""

    def test_same_runtime_fields_are_unchanged(self) -> None:
        parent = EnvironmentConfig(env_id="Navix-Empty-5x5-v0", seed=1, nr_envs=4, max_episode_steps=32)
        child = EnvironmentConfig(env_id="Navix-Empty-5x5-v0", seed=1, nr_envs=4, max_episode_steps=32)
        assert env_io_contract_unchanged(parent, child)

    def test_map_change_is_not_unchanged(self) -> None:
        parent = EnvironmentConfig(env_id="Navix-Empty-5x5-v0", seed=1, nr_envs=4)
        child = EnvironmentConfig(env_id="Navix-DoorKey-6x6-v0", seed=1, nr_envs=4)
        assert not env_io_contract_unchanged(parent, child)

    def test_reward_or_overlay_change_keeps_io_contract(self) -> None:
        parent = EnvironmentConfig(env_id="Navix-Empty-5x5-v0", seed=1, nr_envs=4)
        child = EnvironmentConfig(
            env_id="Navix-Empty-5x5-v0",
            seed=1,
            nr_envs=4,
            reward=RewardWeightsConfig(goal_reached=4.0),
            scenario_reward_id="navix.floor_cell",
            scenario_reward_version=1,
        )
        assert env_io_contract_unchanged(parent, child)
