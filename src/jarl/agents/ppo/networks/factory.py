"""Environment-driven factories for PPO Flax networks."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp

from jarl.agents.ppo.action import ProcessedActionFn, make_processed_action_fn
from jarl.agents.ppo.networks.critic import Critic
from jarl.agents.ppo.networks.gru_critic import GrUCritic
from jarl.agents.ppo.networks.gru_policy import ContinuousGrUPolicy, DiscreteGrUPolicy
from jarl.agents.ppo.networks.policy import ContinuousPolicy, DiscretePolicy
from jarl.envs.types import ActionSpaceType, ObservationSpaceType
from jarl.training.config import AlgorithmConfig

__all__ = [
    "PPOEnvLike",
    "create_action_processor",
    "create_critic_network",
    "create_gru_critic_network",
    "create_gru_policy_network",
    "create_policy_network",
]


@runtime_checkable
class PPOEnvLike(Protocol):
    """Minimal environment contract required to build PPO networks."""

    general_properties: Any
    single_action_space: Any
    single_observation_space: Any


def _observation_indices(env: PPOEnvLike, attribute_name: str) -> jax.Array:
    observation_dim = int(env.single_observation_space.shape[0])
    indices = getattr(env, attribute_name, jnp.arange(observation_dim))
    return jnp.asarray(indices)


def create_policy_network(
    algorithm: AlgorithmConfig,
    env: PPOEnvLike,
) -> tuple[ContinuousPolicy | DiscretePolicy, ProcessedActionFn]:
    """Build a policy network and action post-processor for ``env``.

    Args:
        algorithm: Algorithm hyperparameters.
        env: Environment exposing action/observation spaces and general properties.

    Returns:
        Policy module and JIT action post-processor.

    Raises:
        ValueError: If the environment combination is unsupported.
    """
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type
    policy_indices = _observation_indices(env, "policy_observation_indices")

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        policy = ContinuousPolicy(
            action_shape=env.single_action_space.shape,
            std_dev=algorithm.std_dev,
            observation_indices=policy_indices,
        )
        processor = make_processed_action_fn(
            action_clipping_and_rescaling=algorithm.action_clipping_and_rescaling,
            action_low=jnp.asarray(env.single_action_space.low),
            action_high=jnp.asarray(env.single_action_space.high),
        )
        return policy, processor

    if action_space_type == ActionSpaceType.DISCRETE and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        policy = DiscretePolicy(
            n_actions=int(env.single_action_space.n),
            observation_indices=policy_indices,
        )
        return policy, jax.jit(lambda action: action)

    msg = f"Unsupported PPO environment: action={action_space_type!r}, observation={observation_space_type!r}."
    raise ValueError(msg)


def create_gru_policy_network(
    algorithm: AlgorithmConfig,
    env: PPOEnvLike,
) -> tuple[ContinuousGrUPolicy | DiscreteGrUPolicy, ProcessedActionFn]:
    """Build a GRU policy network and action post-processor for ``env``.

    Args:
        algorithm: Algorithm hyperparameters including GRU fields.
        env: Environment exposing action/observation spaces and general properties.

    Returns:
        GRU policy module and JIT action post-processor.

    Raises:
        ValueError: If the environment combination is unsupported.
    """
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type
    policy_indices = _observation_indices(env, "policy_observation_indices")
    gru_kwargs = _gru_torso_kwargs(algorithm, policy_indices)

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        policy = ContinuousGrUPolicy(
            action_shape=env.single_action_space.shape,
            std_dev=algorithm.std_dev,
            **gru_kwargs,
        )
        processor = make_processed_action_fn(
            action_clipping_and_rescaling=algorithm.action_clipping_and_rescaling,
            action_low=jnp.asarray(env.single_action_space.low),
            action_high=jnp.asarray(env.single_action_space.high),
        )
        return policy, processor

    if action_space_type == ActionSpaceType.DISCRETE and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        policy = DiscreteGrUPolicy(
            n_actions=int(env.single_action_space.n),
            **gru_kwargs,
        )
        return policy, jax.jit(lambda action: action)

    msg = f"Unsupported PPO-GRU environment: action={action_space_type!r}, observation={observation_space_type!r}."
    raise ValueError(msg)


def create_critic_network(
    env: PPOEnvLike,
) -> Critic:
    """Build a critic network for ``env``.

    Args:
        env: Environment exposing observation spaces and general properties.

    Returns:
        Critic module.

    Raises:
        ValueError: If the observation space type is unsupported.
    """
    observation_space_type = env.general_properties.observation_space_type
    if observation_space_type != ObservationSpaceType.FLAT_VALUES:
        msg = f"Unsupported critic observation space: {observation_space_type!r}."
        raise ValueError(msg)

    critic_indices = _observation_indices(env, "critic_observation_indices")
    return Critic(observation_indices=critic_indices)


def create_gru_critic_network(
    algorithm: AlgorithmConfig,
    env: PPOEnvLike,
) -> GrUCritic:
    """Build a recurrent GRU critic for ``env``.

    Args:
        algorithm: Algorithm hyperparameters including GRU fields.
        env: Environment exposing observation spaces and general properties.

    Returns:
        Recurrent critic module with the same GRU history contract as the policy.

    Raises:
        ValueError: If the observation space type is unsupported.
    """
    observation_space_type = env.general_properties.observation_space_type
    if observation_space_type != ObservationSpaceType.FLAT_VALUES:
        msg = f"Unsupported GRU critic observation space: {observation_space_type!r}."
        raise ValueError(msg)

    critic_indices = _observation_indices(env, "critic_observation_indices")
    return GrUCritic(**_gru_torso_kwargs(algorithm, critic_indices))


def _gru_torso_kwargs(algorithm: AlgorithmConfig, observation_indices: jax.Array) -> dict[str, object]:
    """Return shared GRU torso constructor kwargs for policy and critic."""
    return {
        "obs_encoding_dim": algorithm.obs_encoding_dim,
        "gru_hidden_dim": algorithm.gru_hidden_dim,
        "gru_obs_combine_method": algorithm.gru_obs_combine_method,
        "share_gru_obs_encoder": algorithm.share_gru_obs_encoder,
        "observation_indices": observation_indices,
    }


def create_action_processor(
    algorithm: AlgorithmConfig,
    env: PPOEnvLike,
) -> ProcessedActionFn:
    """Return only the action post-processor for ``env`` (identity for discrete spaces)."""
    _, processor = create_policy_network(algorithm, env)
    return processor
