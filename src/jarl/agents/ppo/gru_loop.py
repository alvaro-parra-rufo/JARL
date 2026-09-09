"""Compiled PPO-GRU recurrent training loop."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from jarl.agents.ppo.action import sample_continuous_action, sample_discrete_action
from jarl.agents.ppo.core.gae import compute_gae_advantages
from jarl.agents.ppo.core.gru_loss import make_ppo_gru_loss_fn
from jarl.agents.ppo.core.gru_minibatch import make_gru_minibatch_env_indices
from jarl.agents.ppo.core.hyperparameters import PPOHyperparameters
from jarl.agents.ppo.core.loss import mean_loss_metrics
from jarl.agents.ppo.core.metrics import (
    aggregate_optimization_metrics,
    compute_explained_variance,
    compute_policy_std_dev,
    compute_rollout_target_stats,
)
from jarl.agents.ppo.core.minibatch import normalize_advantages
from jarl.agents.ppo.loop_config import PPOGrULoopConfig
from jarl.agents.ppo.metric_schema import PPO_EVAL_METRIC_SCHEMA, PPO_TRAIN_METRIC_SCHEMA, pack_device_metrics

__all__ = ["build_ppo_gru_train_fn"]

_GrUTrainCarry = tuple[TrainState, TrainState, object, jax.Array, jax.Array, jax.Array]


def build_ppo_gru_train_fn(
    *,
    train_env: object,
    eval_env: object,
    policy: object,
    critic: object,
    process_action: Callable[[jax.Array], jax.Array],
    loop_config: PPOGrULoopConfig,
    hyperparameters: PPOHyperparameters,
    is_discrete: bool,
    log_train_metrics: Callable[..., None],
    log_eval_metrics: Callable[..., None],
    save_model: Callable[..., None],
    save_checkpoint: Callable[..., None] | None = None,
) -> Callable[
    [jax.Array, TrainState, TrainState],
    tuple[TrainState, TrainState],
]:
    """Return a JIT-compiled PPO-GRU training function."""

    def policy_forward_sequence(
        params: object,
        obs_seq: jax.Array,
        done_seq: jax.Array,
        init_carry: jax.Array,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        return policy.apply(
            params,
            obs_seq,
            done_seq,
            init_carry,
            method=policy.forward_sequence,
        )

    def policy_apply_one_step(
        params: object,
        obs: jax.Array,
        carry: jax.Array,
    ) -> tuple[jax.Array, ...]:
        return policy.apply(
            params,
            obs,
            carry,
            method=policy.apply_one_step,
        )

    def critic_forward_sequence(
        params: object,
        obs_seq: jax.Array,
        done_seq: jax.Array,
        init_carry: jax.Array,
    ) -> jax.Array:
        return critic.apply(
            params,
            obs_seq,
            done_seq,
            init_carry,
            method=critic.forward_sequence,
        )

    def critic_apply_one_step(
        params: object,
        obs: jax.Array,
        carry: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return critic.apply(
            params,
            obs,
            carry,
            method=critic.apply_one_step,
        )

    loss_fn = make_ppo_gru_loss_fn(
        policy_forward_sequence=policy_forward_sequence,
        critic_forward_sequence=critic_forward_sequence,
        is_discrete=is_discrete,
        hyperparameters=hyperparameters,
    )
    vmap_loss_fn = jax.vmap(
        loss_fn,
        in_axes=(None, None, 1, 1, 1, 1, 1, 1, 1, 0, 0),
        out_axes=0,
    )

    def mean_vmapped_loss_fn(
        policy_params: object,
        critic_params: object,
        states: jax.Array,
        actions: jax.Array,
        log_probs: jax.Array,
        returns: jax.Array,
        advantages: jax.Array,
        old_values: jax.Array,
        dones: jax.Array,
        init_policy_carries: jax.Array,
        init_critic_carries: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        losses, metrics = vmap_loss_fn(
            policy_params,
            critic_params,
            states,
            actions,
            log_probs,
            returns,
            advantages,
            old_values,
            dones,
            init_policy_carries,
            init_critic_carries,
        )
        return jnp.mean(losses), mean_loss_metrics(metrics)

    grad_loss_fn = jax.value_and_grad(mean_vmapped_loss_fn, argnums=(0, 1), has_aux=True)

    def train(
        key: jax.Array,
        policy_state: TrainState,
        critic_state: TrainState,
    ) -> tuple[TrainState, TrainState]:
        key, reset_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, loop_config.nr_envs)
        env_state = train_env.reset(reset_keys, False)
        policy_carry = policy.initialize_carry(loop_config.nr_envs)
        critic_carry = critic.initialize_carry(loop_config.nr_envs)

        def multi_eval_iteration(  # noqa: ANN202
            carry: _GrUTrainCarry,
            multi_step: jax.Array,
        ):
            policy_state, critic_state, env_state, policy_carry, critic_carry, key = carry

            def learning_iteration(  # noqa: ANN202
                learning_carry: _GrUTrainCarry,
                learning_step: jax.Array,
            ):
                policy_state, critic_state, env_state, policy_carry, critic_carry, key = learning_carry
                rollout_init_policy_carry = policy_carry
                rollout_init_critic_carry = critic_carry

                def single_rollout(  # noqa: ANN202
                    rollout_carry: _GrUTrainCarry,
                    _unused: None,
                ):
                    policy_state, critic_state, env_state, policy_carry, critic_carry, key = rollout_carry
                    key, subkey = jax.random.split(key)
                    observation = env_state.next_observation
                    if is_discrete:
                        logits, next_policy_carry = policy_apply_one_step(
                            policy_state.params, observation, policy_carry
                        )
                        action, log_prob = sample_discrete_action(subkey, logits)
                    else:
                        action_mean, action_logstd, next_policy_carry = policy_apply_one_step(
                            policy_state.params,
                            observation,
                            policy_carry,
                        )
                        action, log_prob = sample_continuous_action(subkey, action_mean, action_logstd)
                    processed_action = process_action(action)
                    value, next_critic_carry = critic_apply_one_step(
                        critic_state.params,
                        observation,
                        critic_carry,
                    )
                    value = value.squeeze(-1)
                    env_state = train_env.step(env_state, processed_action)
                    done = env_state.terminated | env_state.truncated
                    next_value, _unused_bootstrap_carry = critic_apply_one_step(
                        critic_state.params,
                        env_state.actual_next_observation,
                        next_critic_carry,
                    )
                    next_value = next_value.squeeze(-1)
                    next_policy_carry = next_policy_carry * (1.0 - done[:, None])
                    next_critic_carry = next_critic_carry * (1.0 - done[:, None])
                    transition = (
                        observation,
                        env_state.actual_next_observation,
                        action,
                        env_state.reward,
                        value,
                        next_value,
                        env_state.terminated,
                        env_state.truncated,
                        done,
                        log_prob,
                        env_state.info,
                    )
                    return (
                        policy_state,
                        critic_state,
                        env_state,
                        next_policy_carry,
                        next_critic_carry,
                        key,
                    ), transition

                rollout_carry, batch = jax.lax.scan(
                    single_rollout,
                    (policy_state, critic_state, env_state, policy_carry, critic_carry, key),
                    None,
                    loop_config.nr_steps,
                )
                policy_state, critic_state, env_state, policy_carry, critic_carry, key = rollout_carry
                (
                    states,
                    _next_states,
                    actions,
                    rewards,
                    values,
                    next_values,
                    terminations,
                    truncations,
                    dones,
                    log_probs,
                    infos,
                ) = batch

                advantages, returns = compute_gae_advantages(
                    rewards,
                    values,
                    next_values,
                    terminations,
                    truncations=truncations,
                    gamma=hyperparameters.gamma,
                    gae_lambda=hyperparameters.gae_lambda,
                )

                key, subkey = jax.random.split(key)
                batch_env_indices = make_gru_minibatch_env_indices(
                    subkey,
                    nr_envs=loop_config.nr_envs,
                    nr_epochs=loop_config.nr_epochs,
                    nr_minibatches=loop_config.nr_minibatches,
                    nr_minibatch_envs=loop_config.nr_minibatch_envs,
                )

                def minibatch_update(  # noqa: ANN202
                    update_carry: tuple[TrainState, TrainState],
                    minibatch_env_indices: jax.Array,
                ):
                    policy_state, critic_state = update_carry
                    minibatch_advantages = advantages[:, minibatch_env_indices]
                    minibatch_advantages = normalize_advantages(minibatch_advantages)
                    (_, metrics), (policy_gradients, critic_gradients) = grad_loss_fn(
                        policy_state.params,
                        critic_state.params,
                        states[:, minibatch_env_indices],
                        actions[:, minibatch_env_indices],
                        log_probs[:, minibatch_env_indices],
                        returns[:, minibatch_env_indices],
                        minibatch_advantages,
                        values[:, minibatch_env_indices],
                        dones[:, minibatch_env_indices],
                        rollout_init_policy_carry[minibatch_env_indices],
                        rollout_init_critic_carry[minibatch_env_indices],
                    )
                    policy_state = policy_state.apply_gradients(grads=policy_gradients)
                    critic_state = critic_state.apply_gradients(grads=critic_gradients)
                    metrics = dict(metrics)
                    metrics["policy_ratio/max"] = jnp.max(metrics["policy_ratio/ratio"])
                    metrics = {key: value for key, value in metrics.items() if key != "policy_ratio/ratio"}
                    metrics["gradients/policy_grad_norm"] = optax.global_norm(policy_gradients)
                    metrics["gradients/critic_grad_norm"] = optax.global_norm(critic_gradients)
                    return (policy_state, critic_state), metrics

                (policy_state, critic_state), optimization_metrics = jax.lax.scan(
                    minibatch_update,
                    (policy_state, critic_state),
                    batch_env_indices,
                )
                optimization_metrics = aggregate_optimization_metrics(optimization_metrics)
                optimization_metrics.update(compute_rollout_target_stats(values, returns, advantages))
                optimization_metrics["lr/learning_rate"] = policy_state.opt_state[1].hyperparams["learning_rate"]
                optimization_metrics["v_value/explained_variance"] = compute_explained_variance(
                    returns.reshape(-1),
                    values.reshape(-1),
                )
                policy_logstd = None if is_discrete else policy_state.params["params"]["policy_logstd"]
                optimization_metrics["policy/std_dev"] = compute_policy_std_dev(
                    is_discrete=is_discrete,
                    policy_logstd=policy_logstd,
                )

                combined_rollout_step = multi_step * loop_config.nr_updates_per_eval + learning_step + 1
                metric_body = pack_device_metrics(
                    {
                        "rollout/episode_return": jnp.mean(infos["rollout/episode_return"]),
                        "rollout/episode_length": jnp.mean(infos["rollout/episode_length"]),
                        **optimization_metrics,
                    },
                    PPO_TRAIN_METRIC_SCHEMA,
                )
                jax.debug.callback(log_train_metrics, combined_rollout_step, metric_body, policy_state)
                return (policy_state, critic_state, env_state, policy_carry, critic_carry, key), None

            learning_carry, _ = jax.lax.scan(
                learning_iteration,
                (policy_state, critic_state, env_state, policy_carry, critic_carry, key),
                jnp.arange(loop_config.nr_updates_per_eval),
            )
            policy_state, critic_state, env_state, policy_carry, critic_carry, key = learning_carry

            if loop_config.evaluation_active:

                def eval_rollout(  # noqa: ANN202
                    eval_carry: tuple[TrainState, object, jax.Array],
                    _unused: None,
                ):
                    policy_state, eval_env_state, eval_policy_carry = eval_carry
                    if is_discrete:
                        eval_logits, eval_policy_carry = policy_apply_one_step(
                            policy_state.params,
                            eval_env_state.next_observation,
                            eval_policy_carry,
                        )
                        eval_action = jnp.argmax(eval_logits, axis=-1)
                    else:
                        eval_action_mean, _, eval_policy_carry = policy_apply_one_step(
                            policy_state.params,
                            eval_env_state.next_observation,
                            eval_policy_carry,
                        )
                        eval_action = eval_action_mean
                    eval_env_state = eval_env.step(eval_env_state, process_action(eval_action))
                    eval_done = eval_env_state.terminated | eval_env_state.truncated
                    eval_policy_carry = eval_policy_carry * (1.0 - eval_done[:, None])
                    return (policy_state, eval_env_state, eval_policy_carry), None

                key, reset_key = jax.random.split(key)
                reset_keys = jax.random.split(reset_key, loop_config.nr_envs)
                eval_env_state = eval_env.reset(reset_keys, True)
                eval_policy_carry = policy.initialize_carry(loop_config.nr_envs)
                eval_carry, _ = jax.lax.scan(
                    eval_rollout,
                    (policy_state, eval_env_state, eval_policy_carry),
                    None,
                    loop_config.horizon,
                )
                _, eval_env_state, _ = eval_carry
                eval_return = jnp.mean(eval_env_state.info["rollout/episode_return"])
                eval_length = jnp.mean(eval_env_state.info["rollout/episode_length"])
                eval_global_step = (
                    (multi_step + 1) * loop_config.nr_updates_per_eval * loop_config.nr_steps * loop_config.nr_envs
                )
                eval_values = pack_device_metrics(
                    {
                        "eval/episode_return": eval_return,
                        "eval/episode_length": eval_length,
                    },
                    PPO_EVAL_METRIC_SCHEMA,
                )
                jax.debug.callback(log_eval_metrics, eval_global_step, eval_values)

            if loop_config.save_model:
                save_step = (
                    (multi_step + 1) * loop_config.nr_updates_per_eval * loop_config.nr_steps * loop_config.nr_envs
                )
                jax.debug.callback(save_model, save_step, "latest", policy_state, critic_state)

            if loop_config.save_checkpoint and save_checkpoint is not None:
                checkpoint_step = (
                    (multi_step + 1) * loop_config.nr_updates_per_eval * loop_config.nr_steps * loop_config.nr_envs
                )
                rollout_step = (multi_step + 1) * loop_config.nr_updates_per_eval
                optimizer_updates = rollout_step * loop_config.nr_epochs * loop_config.nr_minibatches
                jax.debug.callback(
                    save_checkpoint,
                    checkpoint_step,
                    checkpoint_step,
                    optimizer_updates,
                    key,
                    policy_state,
                    critic_state,
                    policy_carry,
                    critic_carry,
                )

            return (policy_state, critic_state, env_state, policy_carry, critic_carry, key), None

        final_carry, _ = jax.lax.scan(
            multi_eval_iteration,
            (policy_state, critic_state, env_state, policy_carry, critic_carry, key),
            jnp.arange(loop_config.nr_multi_eval_iterations),
        )
        policy_state, critic_state, _env_state, _policy_carry, _critic_carry, _key = final_carry
        return policy_state, critic_state

    return train
