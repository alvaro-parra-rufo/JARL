"""Tests for trainer intra-node resume contracts."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
import pytest
from flax.training.train_state import TrainState
from pytest_mock import MockerFixture

from jarl.agents.ppo.checkpoint_state import (
    CHECKPOINT_KIND_PPO,
    CHECKPOINT_KIND_PPO_GRU,
    PPOCheckpoint,
    PPOGrUCheckpoint,
)
from jarl.agents.ppo.gru_trainer import ppo_gru_full_jax_trainer
from jarl.agents.ppo.trainer import ppo_full_jax_trainer
from jarl.envs.types import ActionSpaceType
from jarl.experiments.node import NodeWorkspace
from jarl.training.config import (
    AlgorithmConfig,
    EnvironmentConfig,
    RLRunConfig,
    VideoConfig,
)
from jarl.training.schedule import compute_training_schedule
from jarl.training.trainer import Trainer


@dataclass(frozen=True, slots=True)
class TrainerCase:
    """Patch targets needed to run one PPO trainer without a compiled loop."""

    trainer: Trainer
    module: str
    policy_factory: str
    critic_factory: str
    state_factory: str
    loop_factory: str
    checkpoint_kind: str


TRAINER_CASES = [
    pytest.param(
        TrainerCase(
            trainer=ppo_full_jax_trainer,
            module="jarl.agents.ppo.trainer",
            policy_factory="create_policy_network",
            critic_factory="create_critic_network",
            state_factory="_create_train_states",
            loop_factory="build_ppo_feedforward_train_fn",
            checkpoint_kind=CHECKPOINT_KIND_PPO,
        ),
        id="ppo",
    ),
    pytest.param(
        TrainerCase(
            trainer=ppo_gru_full_jax_trainer,
            module="jarl.agents.ppo.gru_trainer",
            policy_factory="create_gru_policy_network",
            critic_factory="create_gru_critic_network",
            state_factory="_create_gru_train_states",
            loop_factory="build_ppo_gru_train_fn",
            checkpoint_kind=CHECKPOINT_KIND_PPO_GRU,
        ),
        id="ppo_gru",
    ),
]


@pytest.fixture()
def minimal_resume_trainer_config() -> RLRunConfig:
    """Minimal RL config for mocked trainer resume contract tests."""
    return RLRunConfig(
        environment=EnvironmentConfig(nr_envs=1, seed=7),
        algorithm=AlgorithmConfig(
            total_timesteps=128,
            nr_steps=1,
            nr_epochs=1,
            minibatch_size=1,
            evaluation_and_save_frequency=-1,
            evaluation_active=False,
        ),
        video=VideoConfig(record_video=False, record_final_video=False),
    )


def _train_state(params: dict[str, jnp.ndarray]) -> TrainState:
    return TrainState.create(apply_fn=lambda *_args, **_kwargs: None, params=params, tx=optax.adam(1e-3))


def _mock_trainer_dependencies(
    mocker: MockerFixture,
    case: TrainerCase,
) -> tuple[NodeWorkspace, object, object]:
    """Replace environment, network, and JAX loop dependencies with probes."""
    train_env = mocker.Mock()
    train_env.general_properties.action_space_type = ActionSpaceType.DISCRETE
    train_env.horizon = 1
    train_env.single_observation_space.shape = (1,)
    train_env.single_action_space.shape = ()
    train_env.reset.return_value.next_observation = mocker.sentinel.observation
    env_factory = mocker.Mock(return_value=train_env)

    policy = mocker.Mock()
    policy.initialize_carry.return_value = jnp.zeros((1, 2))
    critic = mocker.Mock()
    critic.initialize_carry.return_value = jnp.zeros((1, 2))
    process_action = mocker.Mock()
    mocker.patch(f"{case.module}.{case.policy_factory}", return_value=(policy, process_action))
    mocker.patch(f"{case.module}.{case.critic_factory}", return_value=critic)
    mocker.patch(
        f"{case.module}.{case.state_factory}",
        return_value=(_train_state({"policy": jnp.array([0.0])}), _train_state({"critic": jnp.array([0.0])})),
    )
    train_fn = mocker.Mock(
        return_value=(_train_state({"policy": jnp.array([1.0])}), _train_state({"critic": jnp.array([1.0])})),
    )
    mocker.patch(f"{case.module}.{case.loop_factory}", return_value=train_fn)
    mocker.patch(f"{case.module}.jax.jit", side_effect=lambda function: function)
    mocker.patch(f"{case.module}.jax.block_until_ready", side_effect=lambda value: value)
    mocker.patch(f"{case.module}.log_jax_training_devices")
    mocker.patch(f"{case.module}.LOGGER.info")

    workspace = mocker.create_autospec(NodeWorkspace, instance=True)
    workspace.wandb_active = False
    workspace.parent_checkpoint_step = None
    workspace.init_checkpoint_manager.return_value = mocker.Mock()
    workspace.make_model_archive_save_callback.return_value = mocker.Mock()
    workspace.save_training_checkpoint.return_value = True
    return workspace, env_factory, train_fn


def _resume_checkpoint_tree(case: TrainerCase) -> dict[str, object]:
    policy_state = _train_state({"policy": jnp.array([2.0])})
    critic_state = _train_state({"critic": jnp.array([3.0])})
    if case.checkpoint_kind == CHECKPOINT_KIND_PPO_GRU:
        checkpoint = PPOGrUCheckpoint.build(
            algorithm_name=CHECKPOINT_KIND_PPO_GRU,
            global_step=64,
            optimizer_updates=4,
            rng_key=jax.random.PRNGKey(1),
            policy_state=policy_state,
            critic_state=critic_state,
            policy_carry=jnp.array([[1.0, 2.0]]),
            critic_carry=jnp.array([[3.0, 4.0]]),
        )
    else:
        checkpoint = PPOCheckpoint.build(
            algorithm_name=CHECKPOINT_KIND_PPO,
            global_step=64,
            optimizer_updates=4,
            rng_key=jax.random.PRNGKey(1),
            policy_state=policy_state,
            critic_state=critic_state,
        )
    return checkpoint.to_pytree()


def _config_for_case(base: RLRunConfig, case: TrainerCase) -> RLRunConfig:
    if case.checkpoint_kind == CHECKPOINT_KIND_PPO_GRU:
        return base.apply_overrides({"algorithm.name": CHECKPOINT_KIND_PPO_GRU})
    return base


class TestTrainerResumeContract:
    """Trainer must restore intra-node checkpoints on resume attempts."""

    @pytest.mark.parametrize("case", TRAINER_CASES)
    def test_available_resume_checkpoint_is_loaded(
        self,
        case: TrainerCase,
        minimal_resume_trainer_config: RLRunConfig,
        mocker: MockerFixture,
    ) -> None:
        workspace, env_factory, _train_fn = _mock_trainer_dependencies(mocker, case)
        config = _config_for_case(minimal_resume_trainer_config, case)
        run_schedule = compute_training_schedule(config.environment, config.algorithm)
        workspace.latest_execution_attempt.return_value = mocker.Mock(resume_of_attempt_id=1)
        workspace.load_resume_checkpoint.return_value = _resume_checkpoint_tree(case)

        case.trainer(workspace, config, run_schedule, env_factory=env_factory)

        workspace.load_resume_checkpoint.assert_called_once_with()

    @pytest.mark.parametrize("case", TRAINER_CASES)
    def test_resume_uses_remaining_timestep_budget(
        self,
        case: TrainerCase,
        minimal_resume_trainer_config: RLRunConfig,
        mocker: MockerFixture,
    ) -> None:
        from jarl.agents.ppo import training_restore

        workspace, env_factory, _train_fn = _mock_trainer_dependencies(mocker, case)
        config = _config_for_case(minimal_resume_trainer_config, case)
        run_schedule = compute_training_schedule(config.environment, config.algorithm)
        workspace.latest_execution_attempt.return_value = mocker.Mock(resume_of_attempt_id=1)
        workspace.load_resume_checkpoint.return_value = _resume_checkpoint_tree(case)
        spy = mocker.spy(training_restore, "apply_remaining_timestep_budget")

        case.trainer(workspace, config, run_schedule, env_factory=env_factory)

        spy.assert_called_once_with(config, 64)


def _gru_trainer_case() -> TrainerCase:
    """Return the PPO-GRU trainer case used by restore-semantics tests."""
    return TrainerCase(
        trainer=ppo_gru_full_jax_trainer,
        module="jarl.agents.ppo.gru_trainer",
        policy_factory="create_gru_policy_network",
        critic_factory="create_gru_critic_network",
        state_factory="_create_gru_train_states",
        loop_factory="build_ppo_gru_train_fn",
        checkpoint_kind=CHECKPOINT_KIND_PPO_GRU,
    )


class TestPPOGrUTrainStartsWithZeroCarry:
    """PPO-GRU ``train()`` always starts from ``env.reset()`` with zero GRU carries."""

    def test_fresh_train_fn_receives_no_episode_carry(
        self,
        minimal_resume_trainer_config: RLRunConfig,
        mocker: MockerFixture,
    ) -> None:
        case = _gru_trainer_case()
        workspace, env_factory, train_fn = _mock_trainer_dependencies(mocker, case)
        config = _config_for_case(minimal_resume_trainer_config, case)
        run_schedule = compute_training_schedule(config.environment, config.algorithm)
        workspace.latest_execution_attempt.return_value = None

        case.trainer(workspace, config, run_schedule, env_factory=env_factory)

        args = train_fn.call_args.args
        assert len(args) == 3
        assert jnp.allclose(args[1].params["policy"], jnp.array([0.0]))
        assert jnp.allclose(args[2].params["critic"], jnp.array([0.0]))

    def test_resume_from_nonzero_carry_starts_without_carry_and_restores_params(
        self,
        minimal_resume_trainer_config: RLRunConfig,
        mocker: MockerFixture,
    ) -> None:
        case = _gru_trainer_case()
        workspace, env_factory, train_fn = _mock_trainer_dependencies(mocker, case)
        config = _config_for_case(minimal_resume_trainer_config, case)
        run_schedule = compute_training_schedule(config.environment, config.algorithm)
        workspace.latest_execution_attempt.return_value = mocker.Mock(resume_of_attempt_id=1)
        workspace.load_resume_checkpoint.return_value = _resume_checkpoint_tree(case)

        case.trainer(workspace, config, run_schedule, env_factory=env_factory)

        args = train_fn.call_args.args
        assert len(args) == 3
        assert jnp.allclose(args[1].params["policy"], jnp.array([2.0]))
        assert jnp.allclose(args[2].params["critic"], jnp.array([3.0]))

    def test_fork_from_nonzero_carry_starts_without_carry_and_restores_params(
        self,
        minimal_resume_trainer_config: RLRunConfig,
        mocker: MockerFixture,
    ) -> None:
        case = _gru_trainer_case()
        workspace, env_factory, train_fn = _mock_trainer_dependencies(mocker, case)
        config = _config_for_case(minimal_resume_trainer_config, case)
        run_schedule = compute_training_schedule(config.environment, config.algorithm)
        workspace.latest_execution_attempt.return_value = None
        workspace.parent_checkpoint_step = 2
        workspace.load_parent_checkpoint.return_value = _resume_checkpoint_tree(case)

        case.trainer(workspace, config, run_schedule, env_factory=env_factory)

        args = train_fn.call_args.args
        assert len(args) == 3
        assert jnp.allclose(args[1].params["policy"], jnp.array([2.0]))
        assert jnp.allclose(args[2].params["critic"], jnp.array([3.0]))
        workspace.load_parent_checkpoint.assert_called_once_with()
