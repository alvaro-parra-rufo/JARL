"""Compiled PPO feedforward training loop."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from jarl.agents.ppo.action import sample_continuous_action, sample_discrete_action
from jarl.agents.ppo.core.batch import flatten_rollout_tensor
from jarl.agents.ppo.core.gae import compute_gae_advantages
from jarl.agents.ppo.core.hyperparameters import PPOHyperparameters
from jarl.agents.ppo.core.loss import make_ppo_loss_fn, mean_loss_metrics
from jarl.agents.ppo.core.metrics import (
    aggregate_optimization_metrics,
    compute_explained_variance,
    compute_policy_std_dev,
    compute_rollout_target_stats,
)
from jarl.agents.ppo.core.minibatch import make_minibatch_indices, normalize_advantages
from jarl.agents.ppo.loop_config import PPOFeedforwardLoopConfig
from jarl.agents.ppo.metric_schema import PPO_EVAL_METRIC_SCHEMA, PPO_TRAIN_METRIC_SCHEMA, pack_device_metrics

__all__ = ["build_ppo_feedforward_train_fn"]


def build_ppo_feedforward_train_fn(
    *,
    train_env: object,
    eval_env: object,
    policy: object,
    critic: object,
    process_action: Callable[[jax.Array], jax.Array],
    loop_config: PPOFeedforwardLoopConfig,
    hyperparameters: PPOHyperparameters,
    is_discrete: bool,
    observation_shape: tuple[int, ...],
    action_shape: tuple[int, ...],
    log_train_metrics: Callable[..., None],
    log_eval_metrics: Callable[..., None],
    save_model: Callable[..., None],
    save_checkpoint: Callable[..., None] | None = None,
) -> Callable[[jax.Array, TrainState, TrainState], tuple[TrainState, TrainState]]:
    """Return a JIT-compiled PPO feedforward training function."""
    loss_fn = make_ppo_loss_fn(
        policy_apply=policy.apply,
        critic_apply=critic.apply,
        is_discrete=is_discrete,
        hyperparameters=hyperparameters,
    )
    vmap_loss_fn = jax.vmap(loss_fn, in_axes=(None, None, 0, 0, 0, 0, 0, 0), out_axes=0)

    def mean_vmapped_loss_fn(
        policy_params: object,
        critic_params: object,
        states: jax.Array,
        actions: jax.Array,
        log_probs: jax.Array,
        returns: jax.Array,
        advantages: jax.Array,
        old_values: jax.Array,
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
        )
        return jnp.mean(losses), mean_loss_metrics(metrics)

    grad_loss_fn = jax.value_and_grad(mean_vmapped_loss_fn, argnums=(0, 1), has_aux=True)

    def train(key: jax.Array, policy_state: TrainState, critic_state: TrainState) -> tuple[TrainState, TrainState]:
        key, reset_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, loop_config.nr_envs)
        env_state = train_env.reset(reset_keys, False)

        def multi_eval_iteration(  # noqa: ANN202
            carry: tuple[TrainState, TrainState, object, jax.Array],
            multi_step: jax.Array,
        ):
            policy_state, critic_state, env_state, key = carry

            def learning_iteration(  # noqa: ANN202
                learning_carry: tuple[TrainState, TrainState, object, jax.Array],
                learning_step: jax.Array,
            ):
                policy_state, critic_state, env_state, key = learning_carry

                def single_rollout(  # noqa: ANN202
                    rollout_carry: tuple[TrainState, TrainState, object, jax.Array],
                    _unused: None,
                ):
                    policy_state, critic_state, env_state, key = rollout_carry
                    key, subkey = jax.random.split(key)
                    observation = env_state.next_observation
                    if is_discrete:
                        logits = policy.apply(policy_state.params, observation)
                        action, log_prob = sample_discrete_action(subkey, logits)
                    else:
                        action_mean, action_logstd = policy.apply(policy_state.params, observation)
                        action, log_prob = sample_continuous_action(subkey, action_mean, action_logstd)
                    processed_action = process_action(action)
                    value = critic.apply(critic_state.params, observation).squeeze(-1)
                    env_state = train_env.step(env_state, processed_action)
                    transition = (
                        observation,
                        env_state.actual_next_observation,
                        action,
                        env_state.reward,
                        value,
                        env_state.terminated,
                        env_state.truncated,
                        log_prob,
                        env_state.info,
                    )
                    return (policy_state, critic_state, env_state, key), transition

                rollout_carry, batch = jax.lax.scan(
                    single_rollout,
                    (policy_state, critic_state, env_state, key),
                    None,
                    loop_config.nr_steps,
                )
                policy_state, critic_state, env_state, key = rollout_carry
                states, next_states, actions, rewards, values, terminations, truncations, log_probs, infos = batch

                next_values = critic.apply(critic_state.params, next_states).squeeze(-1)
                advantages, returns = compute_gae_advantages(
                    rewards,
                    values,
                    next_values,
                    terminations,
                    truncations=truncations,
                    gamma=hyperparameters.gamma,
                    gae_lambda=hyperparameters.gae_lambda,
                )

                batch_states = flatten_rollout_tensor(states, observation_shape)
                batch_actions = actions.reshape(-1) if is_discrete else flatten_rollout_tensor(actions, action_shape)
                batch_advantages = advantages.reshape(-1)
                batch_returns = returns.reshape(-1)
                batch_log_probs = log_probs.reshape(-1)
                batch_old_values = values.reshape(-1)

                key, subkey = jax.random.split(key)
                batch_indices = make_minibatch_indices(
                    subkey,
                    batch_size=loop_config.batch_size,
                    nr_epochs=loop_config.nr_epochs,
                    nr_minibatches=loop_config.nr_minibatches,
                    minibatch_size=loop_config.minibatch_size,
                )

                def minibatch_update(  # noqa: ANN202
                    update_carry: tuple[TrainState, TrainState],
                    minibatch_index_row: jax.Array,
                ):
                    policy_state, critic_state = update_carry
                    minibatch_advantages = normalize_advantages(batch_advantages[minibatch_index_row])
                    (_, metrics), (policy_gradients, critic_gradients) = grad_loss_fn(
                        policy_state.params,
                        critic_state.params,
                        batch_states[minibatch_index_row],
                        batch_actions[minibatch_index_row],
                        batch_log_probs[minibatch_index_row],
                        batch_returns[minibatch_index_row],
                        minibatch_advantages,
                        batch_old_values[minibatch_index_row],
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
                    batch_indices,
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
                return (policy_state, critic_state, env_state, key), None

            learning_carry, _ = jax.lax.scan(
                learning_iteration,
                (policy_state, critic_state, env_state, key),
                jnp.arange(loop_config.nr_updates_per_eval),
            )
            policy_state, critic_state, env_state, key = learning_carry

            if loop_config.evaluation_active:

                def eval_rollout(  # noqa: ANN202
                    eval_carry: tuple[TrainState, object],
                    _unused: None,
                ):
                    policy_state, eval_env_state = eval_carry
                    if is_discrete:
                        eval_logits = policy.apply(policy_state.params, eval_env_state.next_observation)
                        eval_action = jnp.argmax(eval_logits, axis=-1)
                    else:
                        eval_action_mean, _ = policy.apply(policy_state.params, eval_env_state.next_observation)
                        eval_action = eval_action_mean
                    eval_env_state = eval_env.step(eval_env_state, process_action(eval_action))
                    return (policy_state, eval_env_state), None

                key, reset_key = jax.random.split(key)
                reset_keys = jax.random.split(reset_key, loop_config.nr_envs)
                eval_env_state = eval_env.reset(reset_keys, True)
                eval_carry, _ = jax.lax.scan(
                    eval_rollout,
                    (policy_state, eval_env_state),
                    None,
                    loop_config.horizon,
                )
                _, eval_env_state = eval_carry
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
                )

            return (policy_state, critic_state, env_state, key), None

        final_carry, _ = jax.lax.scan(
            multi_eval_iteration,
            (policy_state, critic_state, env_state, key),
            jnp.arange(loop_config.nr_multi_eval_iterations),
        )
        policy_state, critic_state, _env_state, _key = final_carry
        return policy_state, critic_state

    return train
