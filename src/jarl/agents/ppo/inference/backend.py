"""PPO policy loaders for the algorithm-independent inference runtime."""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp

from jarl.agents.ppo.checkpoint_state import (
    CHECKPOINT_KIND_PPO,
    CHECKPOINT_KIND_PPO_GRU,
    PPOCheckpoint,
    PPOGrUCheckpoint,
)
from jarl.agents.ppo.networks import (
    PPOEnvLike,
    create_gru_policy_network,
    create_policy_network,
)
from jarl.envs.types import ActionSpaceType
from jarl.experiments.node import NodeWorkspace
from jarl.inference.backend import InferenceEnvironment
from jarl.inference.policy import InferencePolicyRuntime, LoadedInferencePolicy, PolicyState
from jarl.training.config import RLRunConfig

__all__ = [
    "load_ppo_gru_inference_policy",
    "load_ppo_inference_policy",
]


def load_ppo_inference_policy(
    workspace: NodeWorkspace,
    config: RLRunConfig,
    checkpoint_step: int,
    env: InferenceEnvironment,
) -> LoadedInferencePolicy:
    """Load a feed-forward PPO checkpoint as an inference policy."""
    _validate_algorithm(config, CHECKPOINT_KIND_PPO)
    ppo_env = cast(PPOEnvLike, env)
    policy, process_action = create_policy_network(config.algorithm, ppo_env)
    checkpoint = workspace.load_training_checkpoint(PPOCheckpoint, step=checkpoint_step)
    is_discrete = ppo_env.general_properties.action_space_type == ActionSpaceType.DISCRETE

    def initial_state(batch_size: int) -> tuple[()]:
        del batch_size
        return ()

    def greedy_step(
        params: object,
        observation: jax.Array,
        state: PolicyState,
    ) -> tuple[jax.Array, PolicyState]:
        policy_output = policy.apply(params, observation)
        if is_discrete:
            action = jnp.argmax(policy_output, axis=-1).astype(jnp.int32)
        else:
            action, _logstd = policy_output
        return process_action(action), state

    runtime = InferencePolicyRuntime(
        preprocess_observation=env.preprocess_observation,
        initial_state=initial_state,
        greedy_step=greedy_step,
    )
    return _loaded_policy(
        workspace=workspace,
        checkpoint_step=checkpoint_step,
        config=config,
        runtime=runtime,
        params=checkpoint.policy.params,
        checkpoint=checkpoint,
    )


def load_ppo_gru_inference_policy(
    workspace: NodeWorkspace,
    config: RLRunConfig,
    checkpoint_step: int,
    env: InferenceEnvironment,
) -> LoadedInferencePolicy:
    """Load a PPO-GRU checkpoint as a recurrent inference policy."""
    _validate_algorithm(config, CHECKPOINT_KIND_PPO_GRU)
    ppo_env = cast(PPOEnvLike, env)
    policy, process_action = create_gru_policy_network(config.algorithm, ppo_env)
    checkpoint = workspace.load_training_checkpoint(PPOGrUCheckpoint, step=checkpoint_step)
    is_discrete = ppo_env.general_properties.action_space_type == ActionSpaceType.DISCRETE

    def greedy_step(
        params: object,
        observation: jax.Array,
        state: PolicyState,
    ) -> tuple[jax.Array, PolicyState]:
        carry = cast(jax.Array, state)
        policy_output = policy.apply(
            params,
            observation,
            carry,
            method=policy.apply_one_step,
        )
        if is_discrete:
            logits, next_carry = policy_output
            action = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        else:
            action, _logstd, next_carry = policy_output
        return process_action(action), next_carry

    runtime = InferencePolicyRuntime(
        preprocess_observation=env.preprocess_observation,
        initial_state=policy.initialize_carry,
        greedy_step=greedy_step,
    )
    return _loaded_policy(
        workspace=workspace,
        checkpoint_step=checkpoint_step,
        config=config,
        runtime=runtime,
        params=checkpoint.policy.params,
        checkpoint=checkpoint,
    )


def _loaded_policy(
    *,
    workspace: NodeWorkspace,
    checkpoint_step: int,
    config: RLRunConfig,
    runtime: InferencePolicyRuntime,
    params: object,
    checkpoint: PPOCheckpoint | PPOGrUCheckpoint,
) -> LoadedInferencePolicy:
    return LoadedInferencePolicy(
        runtime=runtime,
        params=params,
        source_node_id=workspace.id,
        checkpoint_step=int(checkpoint_step),
        checkpoint_kind=checkpoint.kind,
        checkpoint_version=int(checkpoint.version),
        algorithm_name=config.algorithm.name,
        global_step=int(checkpoint.global_step),
        optimizer_updates=int(checkpoint.optimizer_updates),
    )


def _validate_algorithm(config: RLRunConfig, expected_name: str) -> None:
    if config.algorithm.name == expected_name:
        return
    msg = f"Inference backend {expected_name!r} cannot load algorithm {config.algorithm.name!r}."
    raise ValueError(msg)
