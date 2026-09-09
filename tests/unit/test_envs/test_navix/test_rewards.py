"""Tests for configurable Navix reward primitives."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import navix as nx
import pytest

from jarl.envs.navix.reward_config import RewardWeightsConfig
from jarl.envs.navix.rewards import (
    CHANNEL_SPECS,
    RewardFn,
    compose_reward_fn,
    occupancy_enter_reward,
    occupancy_entries,
    occupancy_mask,
    occupancy_reward,
    occupancy_steps,
)
from jarl.envs.navix.scenario_rewards import effective_reward_fn
from jarl.training.config import EnvironmentConfig, RLRunConfig
from jarl.training.env_factory import navix_full_jit_env_factory

# Steps live Navix maps; `make test` runs slow markers serially.
pytestmark = pytest.mark.slow

_ROT_CCW = 0
_ROT_CW = 1
_FORWARD = 2
_PICKUP = 3
_TOGGLE = 5
_DONE = 6

_EMPTY = "Navix-Empty-5x5-v0"
_DOORKEY = "Navix-DoorKey-5x5-v0"
_LAVA = "Navix-LavaGapS5-v0"
_GOTODOOR = "Navix-GoToDoor-5x5-v0"
_DYNAMIC = "Navix-Dynamic-Obstacles-6x6-v0"
_FLOOR_CELL = (1, 2)

_EMPTY_TO_GOAL = (_ROT_CW, _FORWARD, _FORWARD, _ROT_CCW, _FORWARD, _FORWARD)
_DOORKEY_PICKUP = (_ROT_CW, _PICKUP)
_DOORKEY_OPEN = (_ROT_CW, _PICKUP, _FORWARD, _FORWARD, _ROT_CCW, _FORWARD, _TOGGLE)
_GOTODOOR_FACE_MISSION = (_ROT_CW, _ROT_CW, _FORWARD, _FORWARD, _ROT_CW, _ROT_CW, _ROT_CW, _FORWARD, _ROT_CW)


def _reset(env_id: str, seed: int = 0) -> tuple[nx.Environment, nx.Timestep]:
    env = nx.make(env_id, observation_fn=nx.observations.symbolic_first_person)
    return env, env.reset(jax.random.PRNGKey(seed))


def _step_reward(
    fn: RewardFn,
    env: nx.Environment,
    timestep: nx.Timestep,
    action: int,
) -> tuple[nx.Timestep, float]:
    prev = timestep
    next_timestep = env.step(timestep, jnp.asarray(action, dtype=jnp.int32))
    value = float(fn(prev.state, jnp.asarray(action, dtype=jnp.int32), next_timestep.state))
    return next_timestep, value


def _channel(name: str) -> RewardFn:
    return next(spec.primitive for spec in CHANNEL_SPECS if spec.name == name)


def _sequence_rewards(env_id: str, actions: tuple[int, ...], fn: RewardFn, seed: int = 0) -> list[float]:
    env, timestep = _reset(env_id, seed)
    values: list[float] = []
    for action in actions:
        timestep, value = _step_reward(fn, env, timestep, action)
        values.append(value)
    return values


class TestComposeAndBounds:
    """Weighted mix stays in the documented numeric range."""

    def test_each_primitive_stays_in_unit_interval(self) -> None:
        env, timestep = _reset(_EMPTY)
        next_timestep = env.step(timestep, jnp.asarray(_FORWARD, dtype=jnp.int32))
        action = jnp.asarray(_FORWARD, dtype=jnp.int32)
        for spec in CHANNEL_SPECS:
            value = float(spec.primitive(timestep.state, action, next_timestep.state))
            assert 0.0 <= value <= 1.0

    def test_goal_reached_weight_four_scales_pulse(self) -> None:
        fn = compose_reward_fn(RewardWeightsConfig(goal_reached=4.0))
        values = _sequence_rewards(_EMPTY, _EMPTY_TO_GOAL, fn)

        assert values[-1] == pytest.approx(4.0)
        assert all(value == pytest.approx(0.0) for value in values[:-1])

    def test_two_firing_channels_sum_weights(self) -> None:
        fn = compose_reward_fn(RewardWeightsConfig(goal_reached=4.0, goal_approach=2.0))
        values = _sequence_rewards(_EMPTY, _EMPTY_TO_GOAL, fn)

        assert values[-1] == pytest.approx(6.0)

    def test_holding_key_and_open_door_sum(self) -> None:
        fn = compose_reward_fn(RewardWeightsConfig(goal_reached=0.0, holding_key=1.0, door_is_open=1.0))
        values = _sequence_rewards(_DOORKEY, _DOORKEY_OPEN, fn)

        # Pickup pays holding_key. Unlocking consumes the key, so the toggle
        # step is only door_is_open.
        assert values[1] == pytest.approx(1.0)
        assert values[-1] == pytest.approx(1.0)


class TestVanillaIdentity:
    """Native Navix reward is unchanged when the mix is absent."""

    @pytest.mark.parametrize(
        ("env_id", "actions", "expected_last"),
        [
            pytest.param(_EMPTY, _EMPTY_TO_GOAL, 1.0, id="empty"),
            pytest.param(_DOORKEY, _EMPTY_TO_GOAL, 0.0, id="doorkey-no-goal-yet"),
            pytest.param(_LAVA, (_FORWARD,), 0.0, id="lava-forward-blocked"),
        ],
    )
    def test_adapter_without_mix_matches_native_scalar(
        self,
        env_id: str,
        actions: tuple[int, ...],
        expected_last: float,
    ) -> None:
        env, timestep = _reset(env_id)
        last_reward = 0.0
        for action in actions:
            timestep = env.step(timestep, jnp.asarray(action, dtype=jnp.int32))
            last_reward = float(timestep.reward)

        assert last_reward == pytest.approx(expected_last)

    def test_empty_mix_goal_reached_matches_native(self) -> None:
        native = _sequence_rewards(_EMPTY, _EMPTY_TO_GOAL, nx.rewards.on_goal_reached)
        custom = _sequence_rewards(_EMPTY, _EMPTY_TO_GOAL, compose_reward_fn(RewardWeightsConfig()))

        assert custom == native
        assert custom[-1] == pytest.approx(1.0)

    def test_gotodoor_native_is_door_done_not_goal_reached(self) -> None:
        native = _sequence_rewards(_GOTODOOR, _GOTODOOR_FACE_MISSION, nx.rewards.on_door_done)
        goal_mix = _sequence_rewards(_GOTODOOR, _GOTODOOR_FACE_MISSION, compose_reward_fn(RewardWeightsConfig()))
        door_mix = _sequence_rewards(
            _GOTODOOR,
            _GOTODOOR_FACE_MISSION,
            compose_reward_fn(RewardWeightsConfig(goal_reached=0.0, door_done=1.0)),
        )

        assert native[-1] == pytest.approx(1.0)
        assert goal_mix[-1] == pytest.approx(0.0)
        assert door_mix == native


class TestObservationHasNoText:
    """PPO input is the symbolic crop; mission text never enters the tensor."""

    def test_fetch_observation_is_the_same_numeric_grid_as_empty(self) -> None:
        _empty_env, empty_ts = _reset(_EMPTY)
        _fetch_env, fetch_ts = _reset("Navix-Fetch-5x5-N2-v0")

        assert empty_ts.observation.shape == (7, 7, 3)
        assert fetch_ts.observation.shape == empty_ts.observation.shape
        assert empty_ts.observation.dtype == fetch_ts.observation.dtype
        assert fetch_ts.observation.dtype == jnp.uint8
        assert len(fetch_ts.state.mission) > 0


class TestEventPulses:
    """Event channels pay once when the Navix slot flips to happened."""

    def test_goal_reached_pulse_on_empty(self) -> None:
        values = _sequence_rewards(_EMPTY, _EMPTY_TO_GOAL, _channel("goal_reached"))

        assert values[-1] == pytest.approx(1.0)
        assert all(value == pytest.approx(0.0) for value in values[:-1])

    def test_key_pickup_is_a_pulse(self) -> None:
        values = _sequence_rewards(_DOORKEY, (*_DOORKEY_PICKUP, _FORWARD), _channel("key_pickup"))

        assert values[1] == pytest.approx(1.0)
        assert values[2] == pytest.approx(0.0)

    def test_door_opening_pulse_and_door_stays_open(self) -> None:
        opening = _sequence_rewards(_DOORKEY, _DOORKEY_OPEN, _channel("door_opening"))
        is_open = _sequence_rewards(_DOORKEY, _DOORKEY_OPEN, _channel("door_is_open"))
        to_door = _sequence_rewards(_DOORKEY, _DOORKEY_OPEN, _channel("distance_to_door"))

        assert opening[-1] == pytest.approx(1.0)
        assert is_open[-1] == pytest.approx(1.0)
        assert to_door[-1] == pytest.approx(0.0)

    def test_empty_foreign_channels_are_zero(self) -> None:
        names = (
            "key_pickup",
            "holding_key",
            "distance_to_key",
            "door_opening",
            "door_unlock",
            "door_is_open",
            "distance_to_door",
            "lava_clearance",
            "distance_to_ball",
            "distance_to_box",
            "door_done",
            "ball_pickup",
            "box_pickup",
        )
        env, timestep = _reset(_EMPTY)
        next_timestep = env.step(timestep, jnp.asarray(_FORWARD, dtype=jnp.int32))
        action = jnp.asarray(_FORWARD, dtype=jnp.int32)
        for name in names:
            value = float(_channel(name)(timestep.state, action, next_timestep.state))
            assert value == pytest.approx(0.0)

        weighted = compose_reward_fn(RewardWeightsConfig(goal_reached=0.0, key_pickup=4.0, lava_clearance=2.0))
        assert float(weighted(timestep.state, action, next_timestep.state)) == pytest.approx(0.0)


class TestProximityAndPerception:
    """Geometry channels follow on-map entities."""

    def test_distance_to_goal_increases_when_approaching(self) -> None:
        values = _sequence_rewards(_EMPTY, _EMPTY_TO_GOAL, _channel("distance_to_goal"))
        approach = _sequence_rewards(_EMPTY, _EMPTY_TO_GOAL, _channel("goal_approach"))

        assert values[1] > values[0]
        assert values[-1] == pytest.approx(1.0)
        assert approach[1] == pytest.approx(1.0)
        assert approach[0] == pytest.approx(0.0)

    def test_goal_approach_idle_is_zero(self) -> None:
        values = _sequence_rewards(_EMPTY, (_DONE,), _channel("goal_approach"))

        assert values[0] == pytest.approx(0.0)

    def test_doorkey_key_distance_and_holding(self) -> None:
        to_key = _sequence_rewards(_DOORKEY, _DOORKEY_PICKUP, _channel("distance_to_key"))
        holding = _sequence_rewards(_DOORKEY, _DOORKEY_PICKUP, _channel("holding_key"))

        assert to_key[0] > 0.0
        assert to_key[-1] == pytest.approx(0.0)
        assert holding[-1] == pytest.approx(1.0)

    def test_lava_clearance_present_and_no_lava_fall_channel(self) -> None:
        names = {spec.name for spec in CHANNEL_SPECS}
        assert "lava_fall" not in names
        env, timestep = _reset(_LAVA)
        next_timestep = env.step(timestep, jnp.asarray(_ROT_CW, dtype=jnp.int32))
        value = float(_channel("lava_clearance")(timestep.state, jnp.asarray(_ROT_CW), next_timestep.state))
        assert 0.0 < value <= 1.0

    def test_dynamic_obstacles_has_ball_distance_empty_does_not(self) -> None:
        empty_env, empty_ts = _reset(_EMPTY)
        empty_next = empty_env.step(empty_ts, jnp.asarray(_FORWARD, dtype=jnp.int32))
        dyn_env, dyn_ts = _reset(_DYNAMIC)
        dyn_next = dyn_env.step(dyn_ts, jnp.asarray(_ROT_CW, dtype=jnp.int32))

        empty_value = float(_channel("distance_to_ball")(empty_ts.state, jnp.asarray(_FORWARD), empty_next.state))
        dyn_value = float(_channel("distance_to_ball")(dyn_ts.state, jnp.asarray(_ROT_CW), dyn_next.state))

        assert empty_value == pytest.approx(0.0)
        assert dyn_value > 0.0

    def test_obstructed_maze_has_box_distance_empty_does_not(self) -> None:
        empty_env, empty_ts = _reset(_EMPTY)
        empty_next = empty_env.step(empty_ts, jnp.asarray(_FORWARD, dtype=jnp.int32))
        maze_env, maze_ts = _reset("Navix-ObstructedMaze-1Dlh-v0")
        maze_next = maze_env.step(maze_ts, jnp.asarray(_ROT_CW, dtype=jnp.int32))

        empty_value = float(_channel("distance_to_box")(empty_ts.state, jnp.asarray(_FORWARD), empty_next.state))
        maze_value = float(_channel("distance_to_box")(maze_ts.state, jnp.asarray(_ROT_CW), maze_next.state))
        empty_pickup = float(_channel("box_pickup")(empty_ts.state, jnp.asarray(_FORWARD), empty_next.state))
        maze_idle_pickup = float(_channel("box_pickup")(maze_ts.state, jnp.asarray(_ROT_CW), maze_next.state))

        assert empty_value == pytest.approx(0.0)
        assert maze_value > 0.0
        assert empty_pickup == pytest.approx(0.0)
        assert maze_idle_pickup == pytest.approx(0.0)

    def test_goal_visible_facing_and_done_at_goal(self) -> None:
        visible = _sequence_rewards(_EMPTY, (_ROT_CW,), _channel("goal_visible"))
        facing = _sequence_rewards(_EMPTY, _EMPTY_TO_GOAL, _channel("facing_goal"))
        done_at = _sequence_rewards(_EMPTY, (*_EMPTY_TO_GOAL[:-1], _DONE), _channel("done_at_goal"))

        assert visible[0] == pytest.approx(1.0)
        assert facing[-1] == pytest.approx(0.0)
        assert facing[-2] == pytest.approx(1.0)
        assert done_at[-1] == pytest.approx(1.0)

    def test_door_done_on_empty_is_zero(self) -> None:
        values = _sequence_rewards(_EMPTY, (_DONE,), _channel("door_done"))

        assert values[0] == pytest.approx(0.0)


class TestOccupancy:
    """Entry pulses ignore dwell; occupancy helpers count rising edges."""

    def test_enter_pulse_versus_dwell(self) -> None:
        enter = occupancy_enter_reward(_FLOOR_CELL)
        occupy = occupancy_reward(_FLOOR_CELL)
        values_enter = _sequence_rewards(_EMPTY, (_FORWARD, _DONE, _DONE), enter)
        values_occupy = _sequence_rewards(_EMPTY, (_FORWARD, _DONE, _DONE), occupy)

        assert values_enter[0] == pytest.approx(1.0)
        assert values_enter[1] == pytest.approx(0.0)
        assert values_occupy[0] == pytest.approx(1.0)
        assert values_occupy[1] == pytest.approx(1.0)

    def test_occupancy_entries_match_enter_pulses(self) -> None:
        env, timestep = _reset(_EMPTY)
        positions = [timestep.state.get_player().position]
        for action in (_FORWARD, _ROT_CW, _FORWARD, _ROT_CW, _ROT_CW, _FORWARD):
            timestep = env.step(timestep, jnp.asarray(action, dtype=jnp.int32))
            positions.append(timestep.state.get_player().position)
        stacked = jnp.stack(positions)
        mask = occupancy_mask(stacked, _FLOOR_CELL)

        assert occupancy_entries(mask) == 2
        assert occupancy_steps(mask) >= occupancy_entries(mask)

    def test_effective_fn_without_overlay_does_not_pay_floor_cell(self) -> None:
        fn = effective_reward_fn(RewardWeightsConfig(), spec=None)
        values = _sequence_rewards(_EMPTY, (_FORWARD,), fn)

        assert values[0] == pytest.approx(0.0)

    def test_effective_fn_with_overlay_pays_scale_on_entry(self) -> None:
        from jarl.envs.navix.scenario_rewards import ScenarioRewardSpec

        spec = ScenarioRewardSpec(
            id="test.overlay",
            version=1,
            scale=0.5,
            compatible_env_ids=(_EMPTY,),
            position=_FLOOR_CELL,
        )
        mix = compose_reward_fn(RewardWeightsConfig())
        total = effective_reward_fn(RewardWeightsConfig(), spec)
        enter_values = _sequence_rewards(_EMPTY, (_FORWARD, _DONE), total)
        mix_values = _sequence_rewards(_EMPTY, (_FORWARD, _DONE), mix)

        assert enter_values[0] == pytest.approx(0.5)
        assert enter_values[1] == pytest.approx(0.0)
        assert mix_values[0] == pytest.approx(0.0)


class TestFactoryInjection:
    """Train factory applies the mix and rejects incompatible overlays before ``nx.make``."""

    def test_factory_without_mix_uses_native_empty_reward(self) -> None:
        config = RLRunConfig(environment=EnvironmentConfig(env_id=_EMPTY))
        env = navix_full_jit_env_factory(config)
        keys = jax.random.split(jax.random.PRNGKey(0), 1)
        state = env.reset(keys, eval_mode=True)
        next_state = env.step(state, jnp.asarray([_FORWARD], dtype=jnp.int32))

        assert float(next_state.reward[0]) == pytest.approx(0.0)

    def test_factory_rejects_incompatible_overlay_before_make(self) -> None:
        from jarl.envs.navix.scenario_rewards import ScenarioRewardError

        config = RLRunConfig(
            environment=EnvironmentConfig(
                env_id=_EMPTY,
                reward=RewardWeightsConfig(),
                scenario_reward_id="navix.floor_cell",
                scenario_reward_version=1,
            )
        )

        with pytest.raises(ScenarioRewardError, match="incompatible with env_id"):
            navix_full_jit_env_factory(config)
