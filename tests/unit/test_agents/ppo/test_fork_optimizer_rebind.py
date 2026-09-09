"""Tests for fork/extend optimizer rebind after parent checkpoint restore (P11)."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
import pytest
from flax.training.train_state import TrainState
from pytest_mock import MockerFixture

from jarl.agents.ppo import training_restore
from jarl.agents.ppo.checkpoint_state import CHECKPOINT_KIND_PPO, PPOCheckpoint
from jarl.agents.ppo.core.learning_rate import make_linear_annealed_learning_rate_schedule
from jarl.agents.ppo.optimizer_restore import (
    read_optimizer_learning_rate,
    rebind_optimizer_after_fork_restore,
)
from jarl.agents.ppo.training_restore import prepare_training_restore
from jarl.experiments.node import NodeWorkspace
from jarl.training.config import AlgorithmConfig, EnvironmentConfig, RLRunConfig
from jarl.training.schedule import compute_training_schedule

_BASELINE_NR_UPDATES = 244
_OPTIMIZER_STEPS_PER_ROLLOUT = 16
_PARENT_OPTIMIZER_STEPS = _BASELINE_NR_UPDATES * _OPTIMIZER_STEPS_PER_ROLLOUT


def _make_ppo_train_state(
    *,
    learning_rate: float | Callable[[jax.Array], jax.Array],
    params: dict[str, jax.Array] | None = None,
) -> TrainState:
    resolved_params = params or {"policy": jnp.array([1.0])}
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate),
    )
    return TrainState.create(
        apply_fn=lambda *_args, **_kwargs: None,
        params=resolved_params,
        tx=optimizer,
    )


def _advance_optimizer(train_state: TrainState, steps: int) -> TrainState:
    state = train_state
    zero_grads = jax.tree.map(jnp.zeros_like, state.params)
    for _ in range(steps):
        state = state.apply_gradients(grads=zero_grads)
    return state


def _exhausted_parent_state() -> TrainState:
    schedule = make_linear_annealed_learning_rate_schedule(
        base_learning_rate=0.00025,
        nr_updates=_BASELINE_NR_UPDATES,
        nr_minibatches=4,
        nr_epochs=4,
    )
    parent = _make_ppo_train_state(learning_rate=schedule)
    parent = _advance_optimizer(parent, _PARENT_OPTIMIZER_STEPS)
    assert float(read_optimizer_learning_rate(parent)) < 1e-5
    return parent


class TestRebindOptimizerAfterForkRestore:
    """Fork restore must start the child LR schedule from a fresh optimizer."""

    def test_child_constant_learning_rate_after_exhausted_parent(self) -> None:
        parent = _exhausted_parent_state()
        child_template = _make_ppo_train_state(learning_rate=3e-5)

        rebound = rebind_optimizer_after_fork_restore(parent, child_template)

        assert float(read_optimizer_learning_rate(rebound)) == pytest.approx(3e-5, rel=1e-6)
        assert rebound.params["policy"] == pytest.approx(parent.params["policy"])
        assert int(rebound.step) == int(parent.step)

    def test_child_annealed_learning_rate_starts_at_base_after_exhausted_parent(self) -> None:
        parent = _exhausted_parent_state()
        child_schedule = make_linear_annealed_learning_rate_schedule(
            base_learning_rate=0.00025,
            nr_updates=120,
            nr_minibatches=4,
            nr_epochs=4,
        )
        child_template = _make_ppo_train_state(learning_rate=child_schedule)

        rebound = rebind_optimizer_after_fork_restore(parent, child_template)

        assert float(read_optimizer_learning_rate(rebound)) == pytest.approx(0.00025, rel=1e-6)


class TestPrepareTrainingRestoreOptimizerRebind:
    """``prepare_training_restore`` rebinding policy for fork vs resume."""

    @pytest.fixture()
    def config(self) -> RLRunConfig:
        return RLRunConfig(
            environment=EnvironmentConfig(nr_envs=2, seed=3),
            algorithm=AlgorithmConfig(
                total_timesteps=128,
                nr_steps=8,
                nr_epochs=4,
                minibatch_size=16,
                learning_rate=3e-5,
                anneal_learning_rate=False,
                evaluation_and_save_frequency=-1,
                evaluation_active=False,
            ),
        )

    def test_fork_restore_rebinds_policy_and_critic_optimizers(
        self,
        config: RLRunConfig,
        mocker: MockerFixture,
    ) -> None:
        schedule = compute_training_schedule(config.environment, config.algorithm)
        policy_template = _make_ppo_train_state(learning_rate=3e-5, params={"policy": jnp.array([0.0])})
        critic_template = _make_ppo_train_state(learning_rate=3e-5, params={"critic": jnp.array([0.0])})
        exhausted_parent = _exhausted_parent_state()
        restored_critic = exhausted_parent.replace(params={"critic": jnp.array([9.0])})
        workspace = mocker.create_autospec(NodeWorkspace, instance=True)
        workspace.latest_execution_attempt.return_value = None
        mocker.patch(
            "jarl.agents.ppo.training_restore.restore_fork_parent_checkpoint",
            return_value=(
                jax.random.PRNGKey(0),
                exhausted_parent,
                restored_critic,
                None,
                None,
            ),
        )
        rebind_spy = mocker.spy(training_restore, "rebind_optimizer_after_fork_restore")

        result = prepare_training_restore(
            workspace=workspace,
            config=config,
            schedule=schedule,
            policy_template=policy_template,
            critic_template=critic_template,
            key=jax.random.PRNGKey(1),
        )

        assert rebind_spy.call_count == 2
        assert float(read_optimizer_learning_rate(result.policy_state)) == pytest.approx(3e-5, rel=1e-6)
        assert float(read_optimizer_learning_rate(result.critic_state)) == pytest.approx(3e-5, rel=1e-6)
        assert result.policy_state.params["policy"] == pytest.approx(exhausted_parent.params["policy"])
        assert result.critic_state.params["critic"] == pytest.approx(jnp.array([9.0]))

    def test_resume_restore_does_not_rebind_optimizer(
        self,
        config: RLRunConfig,
        mocker: MockerFixture,
    ) -> None:
        schedule = compute_training_schedule(config.environment, config.algorithm)
        policy_template = _make_ppo_train_state(learning_rate=3e-5)
        critic_template = _make_ppo_train_state(learning_rate=3e-5, params={"critic": jnp.array([0.0])})
        checkpoint = PPOCheckpoint.build(
            algorithm_name=CHECKPOINT_KIND_PPO,
            global_step=64,
            optimizer_updates=4,
            rng_key=jax.random.PRNGKey(2),
            policy_state=policy_template,
            critic_state=critic_template,
        )
        workspace = mocker.create_autospec(NodeWorkspace, instance=True)
        workspace.latest_execution_attempt.return_value = mocker.Mock(resume_of_attempt_id=1)
        workspace.load_resume_checkpoint.return_value = checkpoint.to_pytree()
        rebind_spy = mocker.spy(training_restore, "rebind_optimizer_after_fork_restore")

        prepare_training_restore(
            workspace=workspace,
            config=config,
            schedule=schedule,
            policy_template=policy_template,
            critic_template=critic_template,
            key=jax.random.PRNGKey(3),
        )

        rebind_spy.assert_not_called()
