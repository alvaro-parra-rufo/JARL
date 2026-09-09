"""Tests for the EmptyVariant map geometry, overlay payment, and compact summaries."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import navix as nx
import pytest

from jarl.envs.navix.custom.empty_variant import CELL_ENTRY_POSITION, EMPTY_VARIANT_ENV_ID
from jarl.envs.navix.full_jit import NavixFullJITEnv, make_navix_full_jit_env
from jarl.envs.navix.reward_config import RewardWeightsConfig
from jarl.envs.navix.rewards import occupancy_entries, occupancy_mask, occupancy_steps
from jarl.envs.navix.scenario_rewards import (
    FLOOR_CELL_SCENARIO_ID,
    FLOOR_CELL_SCENARIO_VERSION,
    resolve_scenario_reward_spec,
)
from jarl.envs.navix.telemetry import (
    NavixCaptureProfile,
    NavixRolloutIdentity,
    NavixTelemetryRollout,
    derive_events,
    summarize_navix_trace,
)
from jarl.inference.policy import InferencePolicyRuntime, PolicyState
from jarl.training.config import EnvironmentConfig, RLRunConfig
from jarl.training.env_factory import navix_full_jit_env_factory

# Compiles Navix rollouts; `make test` runs slow markers serially.
pytestmark = pytest.mark.slow

_ROT_CCW = 0
_ROT_CW = 1
_FORWARD = 2

_REENTRY_ACTIONS = (_FORWARD, _ROT_CW, _FORWARD, _ROT_CW, _ROT_CW, _FORWARD)
_PATH_THROUGH_CELL_TO_GOAL = (_FORWARD, _ROT_CW, _FORWARD, _FORWARD, _ROT_CCW, _FORWARD)
_COMPACT_FORBIDDEN = (
    "cell_entry",
    "occupancy",
    "dopamina",
    "dopamine",
    "bonus",
    "trap",
    "floor_cell",
    "hidden_cell",
)


def _reset(seed: int = 0) -> tuple[nx.Environment, nx.Timestep]:
    env = nx.make(EMPTY_VARIANT_ENV_ID, observation_fn=nx.observations.symbolic_first_person)
    return env, env.reset(jax.random.PRNGKey(seed))


def _factory_env(*, overlay: bool, mix: bool) -> NavixFullJITEnv:
    if not mix:
        environment = EnvironmentConfig(env_id=EMPTY_VARIANT_ENV_ID)
    else:
        environment = EnvironmentConfig(
            env_id=EMPTY_VARIANT_ENV_ID,
            reward=RewardWeightsConfig(),
            scenario_reward_id=FLOOR_CELL_SCENARIO_ID if overlay else None,
            scenario_reward_version=FLOOR_CELL_SCENARIO_VERSION if overlay else None,
        )
    return navix_full_jit_env_factory(RLRunConfig(environment=environment))


def _step_rewards(env: NavixFullJITEnv, actions: tuple[int, ...]) -> list[float]:
    keys = jax.random.split(jax.random.PRNGKey(0), 1)
    state = env.reset(keys, eval_mode=True)
    values: list[float] = []
    for action in actions:
        state = env.step(state, jnp.asarray([action], dtype=jnp.int32))
        values.append(float(state.reward[0]))
    return values


def _sequence_runtime() -> InferencePolicyRuntime:
    def preprocess_observation(observation: jax.Array) -> jax.Array:
        return jnp.asarray(observation, dtype=jnp.float32).reshape((observation.shape[0], -1)) / 255.0

    def initial_state(batch_size: int) -> jax.Array:
        return jnp.zeros((batch_size,), dtype=jnp.int32)

    def greedy_step(
        params: object,
        _observation: jax.Array,
        state: PolicyState,
    ) -> tuple[jax.Array, PolicyState]:
        action_sequence = jnp.asarray(params, dtype=jnp.int32)
        indices = jnp.minimum(jnp.asarray(state, dtype=jnp.int32), action_sequence.shape[0] - 1)
        return action_sequence[indices], jnp.asarray(state, dtype=jnp.int32) + 1

    return InferencePolicyRuntime(
        preprocess_observation=preprocess_observation,
        initial_state=initial_state,
        greedy_step=greedy_step,
    )


class TestEmptyVariantFactory:
    """The variant id loads with the Empty-5x5 observation contract."""

    def test_factory_accepts_id_and_obs_shape(self) -> None:
        env = make_navix_full_jit_env(EMPTY_VARIANT_ENV_ID)

        assert env.env_id == EMPTY_VARIANT_ENV_ID
        assert env.get_processed_observation_shape() == (147,)
        assert int(env.single_action_space.n) == 7


class TestEmptyVariantGeometry:
    """The overlay cell is walkable floor, not a second Goal."""

    def test_floor_cell_is_walkable_and_not_the_goal(self) -> None:
        _env, timestep = _reset()
        row, col = CELL_ENTRY_POSITION
        player = timestep.state.get_player().position
        goal = timestep.state.get_goals().position[0]

        assert int(timestep.state.grid[row, col]) == 0
        assert tuple(int(v) for v in player) == (1, 1)
        assert tuple(int(v) for v in goal) == (3, 3)
        assert tuple(int(v) for v in goal) != CELL_ENTRY_POSITION


class TestEmptyVariantRewards:
    """Geometry alone does not pay; overlay scale is applied only when configured."""

    def test_native_reward_does_not_pay_on_entry(self) -> None:
        values = _step_rewards(_factory_env(overlay=False, mix=False), (_FORWARD,))

        assert values[0] == pytest.approx(0.0)

    def test_mix_without_overlay_does_not_pay_on_entry(self) -> None:
        values = _step_rewards(_factory_env(overlay=False, mix=True), (_FORWARD,))

        assert values[0] == pytest.approx(0.0)

    def test_mix_with_overlay_pays_scale_on_entry_and_goal(self) -> None:
        values = _step_rewards(_factory_env(overlay=True, mix=True), _PATH_THROUGH_CELL_TO_GOAL)

        assert values[0] == pytest.approx(0.5)
        assert values[1:-1] == pytest.approx([0.0] * (len(values) - 2))
        assert values[-1] == pytest.approx(1.0)

    def test_native_path_through_cell_pays_only_at_goal(self) -> None:
        values = _step_rewards(_factory_env(overlay=False, mix=False), _PATH_THROUGH_CELL_TO_GOAL)

        assert values[:-1] == pytest.approx([0.0] * (len(values) - 1))
        assert values[-1] == pytest.approx(1.0)

    def test_occupancy_entries_match_reentry_trace(self) -> None:
        env, timestep = _reset()
        positions = [timestep.state.get_player().position]
        for action in _REENTRY_ACTIONS:
            timestep = env.step(timestep, jnp.asarray(action, dtype=jnp.int32))
            positions.append(timestep.state.get_player().position)
        stacked = jnp.stack(positions)
        mask = occupancy_mask(stacked, CELL_ENTRY_POSITION)

        assert occupancy_entries(mask) == 2
        assert occupancy_steps(mask) >= occupancy_entries(mask)


class TestEmptyVariantCompact:
    """Compact rollout text must not name the overlay or occupancy oracle."""

    def test_compact_summary_omits_overlay_tokens(self) -> None:
        spec = resolve_scenario_reward_spec(
            EMPTY_VARIANT_ENV_ID,
            FLOOR_CELL_SCENARIO_ID,
            FLOOR_CELL_SCENARIO_VERSION,
        )
        rollout = NavixTelemetryRollout(
            EMPTY_VARIANT_ENV_ID,
            _sequence_runtime(),
            len(_REENTRY_ACTIONS),
            NavixCaptureProfile.analysis(),
            reward=RewardWeightsConfig(),
            scenario_spec=spec,
        )
        keys = jax.random.split(jax.random.PRNGKey(0), 1)

        trace = rollout.rollout(jnp.asarray(_REENTRY_ACTIONS, dtype=jnp.int32), keys)
        events = derive_events(trace, action_names=rollout.action_names)
        summary = summarize_navix_trace(
            trace,
            identity=NavixRolloutIdentity(env_id=EMPTY_VARIANT_ENV_ID, seed=0),
            action_names=rollout.action_names,
        )
        text = summary.as_text(compact=True).casefold()
        kinds = {event.kind for event in events}

        assert float(trace.rewards[0, 0]) == pytest.approx(0.5)
        for token in _COMPACT_FORBIDDEN:
            assert token not in text
            assert token not in kinds
