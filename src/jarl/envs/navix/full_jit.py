"""jarl full-JIT adapter for native Navix environments."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import navix as nx
from flax import struct
from gymnasium import spaces

from jarl.envs.navix.aliases import resolve_navix_registry_id
from jarl.envs.navix.reward_config import RewardWeightsConfig
from jarl.envs.navix.scenario_rewards import ScenarioRewardSpec, effective_reward_fn
from jarl.envs.types import ActionSpaceType, DataInterfaceType, ObservationSpaceType


@struct.dataclass
class GeneralProperties:
    """Environment properties consumed by jarl full-JIT PPO agents."""

    observation_space_type: ObservationSpaceType = ObservationSpaceType.FLAT_VALUES
    action_space_type: ActionSpaceType = ActionSpaceType.DISCRETE
    data_interface_type: DataInterfaceType = DataInterfaceType.JAX


@struct.dataclass
class NavixFullJITState:
    """Batched state returned by the jarl full-JIT Navix adapter."""

    timestep: nx.Timestep
    next_observation: jax.Array
    actual_next_observation: jax.Array
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    info: dict[str, Any]
    info_episode_store: dict[str, Any]
    key: jax.Array


class NavixFullJITEnv:
    """Navix environment with the state API expected by jarl full-JIT PPO."""

    def __init__(
        self,
        env_id: str,
        max_episode_steps: int | None = None,
        *,
        reward: RewardWeightsConfig | None = None,
        scenario_spec: ScenarioRewardSpec | None = None,
    ) -> None:
        """Create a full-JIT Navix adapter for PPO training."""
        registry_env_id = resolve_navix_registry_id(env_id)
        self.env_id = env_id
        make_kwargs: dict[str, Any] = {"observation_fn": nx.observations.symbolic_first_person}
        if max_episode_steps is not None:
            make_kwargs["max_steps"] = max_episode_steps
        if reward is not None:
            make_kwargs["reward_fn"] = effective_reward_fn(reward, scenario_spec)
        self._env = nx.make(registry_env_id, **make_kwargs)

        self._reset_batch = jax.jit(jax.vmap(self._env.reset))
        self._step_batch = jax.jit(jax.vmap(self._env.step))
        self.raw_observation_shape = tuple(int(dim) for dim in self._env.observation_space.shape)
        self.horizon = int(self._env.max_steps)
        self.general_properties = GeneralProperties()
        self.single_action_space = spaces.Discrete(int(self._env.action_space.n))
        self.single_observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=self.get_processed_observation_shape(),
            dtype=jnp.float32,
        )

    def reset(self, key: jax.Array, eval_mode: bool) -> NavixFullJITState:
        """Reset a batch of Navix environments."""
        del eval_mode
        timestep = self._reset_batch(key)
        next_observation = self.preprocess_observation(timestep.observation)
        batch_size = key.shape[0]
        reward = jnp.zeros((batch_size,), dtype=jnp.float32)
        terminated = jnp.zeros((batch_size,), dtype=jnp.bool_)
        truncated = jnp.zeros((batch_size,), dtype=jnp.bool_)
        info = {
            "rollout/episode_return": reward,
            "rollout/episode_length": jnp.zeros((batch_size,), dtype=jnp.int32),
        }
        info_episode_store = {
            "episode_return": reward,
            "episode_length": jnp.zeros((batch_size,), dtype=jnp.int32),
        }
        return NavixFullJITState(
            timestep=timestep,
            next_observation=next_observation,
            actual_next_observation=next_observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            info_episode_store=info_episode_store,
            key=key,
        )

    def step(self, state: NavixFullJITState, action: jax.Array) -> NavixFullJITState:
        """Step a batch and autoreset finished episodes."""
        action = jnp.asarray(action, dtype=jnp.int32)
        stepped_timestep = self._step_batch(state.timestep, action)
        actual_next_observation = self.preprocess_observation(stepped_timestep.observation)
        reward = jnp.asarray(stepped_timestep.reward, dtype=jnp.float32)
        terminated = jnp.asarray(stepped_timestep.is_termination(), dtype=jnp.bool_)
        truncated = jnp.asarray(stepped_timestep.is_truncation(), dtype=jnp.bool_)
        done = terminated | truncated | jnp.asarray(stepped_timestep.is_done(), dtype=jnp.bool_)

        split_keys = jax.vmap(lambda key: jax.random.split(key, 2))(state.key)
        next_key = split_keys[:, 0]
        reset_key = split_keys[:, 1]
        reset_timestep = self._reset_batch(reset_key)
        timestep = jax.tree_util.tree_map(
            lambda reset_leaf, step_leaf: jnp.where(
                self._expand_done(done, step_leaf),
                reset_leaf,
                step_leaf,
            ),
            reset_timestep,
            stepped_timestep,
        )
        next_observation = self.preprocess_observation(timestep.observation)

        episode_return = state.info_episode_store["episode_return"] + reward
        episode_length = state.info_episode_store["episode_length"] + 1
        info = {
            "rollout/episode_return": jnp.where(done, episode_return, state.info["rollout/episode_return"]),
            "rollout/episode_length": jnp.where(done, episode_length, state.info["rollout/episode_length"]),
        }
        info_episode_store = {
            "episode_return": jnp.where(done, 0.0, episode_return),
            "episode_length": jnp.where(done, 0, episode_length),
        }

        return state.replace(
            timestep=timestep,
            next_observation=next_observation,
            actual_next_observation=actual_next_observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            info_episode_store=info_episode_store,
            key=next_key,
        )

    def close(self) -> None:
        """Release resources held by the environment."""

    def render(self, state: NavixFullJITState) -> NavixFullJITState:
        """Return `state`; rendering is intentionally not implemented for JIT training."""
        return state

    def preprocess_observation(self, observation: jax.Array) -> jax.Array:
        """Flatten symbolic Navix observations to normalized ``float32`` vectors."""
        trailing_dims = len(self.raw_observation_shape)
        batch_shape = observation.shape[:-trailing_dims]
        return jnp.asarray(observation, dtype=jnp.float32).reshape((*batch_shape, -1)) / 255.0

    def get_processed_observation_shape(self) -> tuple[int, ...]:
        """Return the shape produced by ``preprocess_observation`` for one observation."""
        observation = jnp.zeros(self.raw_observation_shape, dtype=self._env.observation_space.dtype)
        return tuple(int(dim) for dim in self.preprocess_observation(observation).shape)

    def discrete_action_names(self) -> tuple[str, ...]:
        """Return ordered Navix action function names."""
        return tuple(action.__name__ for action in self._env.action_set)

    @staticmethod
    def _expand_done(done: jax.Array, leaf: jax.Array) -> jax.Array:
        shape = (done.shape[0],) + (1,) * (jnp.ndim(leaf) - 1)
        return done.reshape(shape)


def make_navix_full_jit_env(
    env_id: str,
    max_episode_steps: int | None = None,
    *,
    reward: RewardWeightsConfig | None = None,
    scenario_spec: ScenarioRewardSpec | None = None,
) -> NavixFullJITEnv:
    """Create a Navix adapter for jarl full-JIT PPO agents."""
    return NavixFullJITEnv(
        env_id=env_id,
        max_episode_steps=max_episode_steps,
        reward=reward,
        scenario_spec=scenario_spec,
    )


__all__ = [
    "GeneralProperties",
    "NavixFullJITEnv",
    "NavixFullJITState",
    "make_navix_full_jit_env",
]
