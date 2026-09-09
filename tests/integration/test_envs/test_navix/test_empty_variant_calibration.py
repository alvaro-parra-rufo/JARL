"""Offline calibration gate for the EmptyVariant overlay case.

This module is the empirical gate for the experiment case: prefix hook, a
tool-reachable mix that makes extra loops suboptimal, overlay still paying,
and no ``le`` cap on weights. PPO multi-seed notes live under
``.agents/local/notes/`` and are not an LLM surface.

Tabular thresholds belong here, not in prompts, ``CaseSpec``, or tools.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from jarl.envs.navix.custom.empty_variant import CELL_ENTRY_POSITION, EMPTY_VARIANT_ENV_ID
from jarl.envs.navix.full_jit import NavixFullJITEnv, NavixFullJITState
from jarl.envs.navix.reward_config import RewardWeightsConfig
from jarl.envs.navix.rewards import occupancy_entries, occupancy_mask
from jarl.envs.navix.scenario_rewards import FLOOR_CELL_SCENARIO_ID, FLOOR_CELL_SCENARIO_VERSION
from jarl.experiments.cases.builtin.navix_empty_variant import CASE
from jarl.experiments.cases.metadata import ExperimentCaseMetadata
from jarl.operations.graph.set_reward import SetRewardRequest, set_reward
from jarl.training.env_factory import navix_full_jit_env_factory

pytestmark = pytest.mark.slow

_ROT_CCW = 0
_ROT_CW = 1
_FORWARD = 2
# Closed (1,1)↔(1,2) shuttle: enter, 180°, leave, 180°. Repeating the
# two-entry geometry path does not restore pose and wanders into the Goal;
# FullJIT then autoresets and the farm return is no longer overlay-only.
_FARM_PERIOD = (_FORWARD, _ROT_CW, _ROT_CW, _FORWARD, _ROT_CW, _ROT_CW)
_PATH_THROUGH_CELL_TO_GOAL = (_FORWARD, _ROT_CW, _FORWARD, _FORWARD, _ROT_CCW, _FORWARD)
_SOLVABLE_GOAL_WEIGHT = 26.0
_FARM_CYCLES = 15
_GAMMA = 0.99


def _player_xy(state: NavixFullJITState) -> jax.Array:
    position = jnp.asarray(state.timestep.state.get_player().position, dtype=jnp.int32)
    return position.reshape(-1, 2)[0]


def _step_trace(env: NavixFullJITEnv, actions: tuple[int, ...]) -> tuple[list[float], jax.Array]:
    keys = jax.random.split(jax.random.PRNGKey(0), 1)
    state = env.reset(keys, eval_mode=True)
    rewards: list[float] = []
    positions = [_player_xy(state)]
    for action in actions:
        state = env.step(state, jnp.asarray([action], dtype=jnp.int32))
        rewards.append(float(state.reward[0]))
        positions.append(_player_xy(state))
    return rewards, jnp.stack(positions)


def _discounted_return(rewards: list[float], *, gamma: float = _GAMMA) -> float:
    total = 0.0
    discount = 1.0
    for reward in rewards:
        total += discount * reward
        discount *= gamma
    return total


def _farm_actions() -> tuple[int, ...]:
    return _FARM_PERIOD * _FARM_CYCLES


class TestEmptyVariantCalibrationGate:
    """Scripted/tabular solvency of the materialized case. No LLM surfaces."""

    def test_prefix_farm_hooks_and_entry_pays_scale(self, tmp_path: Path) -> None:
        context = CASE.materialize(tmp_path / "prefix")
        graph = context.reload_graph()
        env = navix_full_jit_env_factory(graph.resolve_config(graph.current_node))

        farm_rewards, farm_positions = _step_trace(env, _farm_actions())
        direct_rewards, _direct_positions = _step_trace(env, _PATH_THROUGH_CELL_TO_GOAL)
        entries = occupancy_entries(occupancy_mask(farm_positions, CELL_ENTRY_POSITION))
        snapshot = ExperimentCaseMetadata.load(graph.layout.case_metadata_path)

        assert float(farm_rewards[0]) == pytest.approx(0.5)
        assert entries >= 8
        assert _discounted_return(farm_rewards) > _discounted_return(direct_rewards)
        assert snapshot.scenario_reward_id == FLOOR_CELL_SCENARIO_ID
        assert snapshot.scenario_reward_version == FLOOR_CELL_SCENARIO_VERSION

    def test_tool_reachable_mix_makes_direct_path_better_with_overlay_intact(
        self,
        tmp_path: Path,
    ) -> None:
        context = CASE.materialize(tmp_path / "solvable")
        graph = context.reload_graph()
        set_reward(graph, SetRewardRequest(weights={"goal_reached": _SOLVABLE_GOAL_WEIGHT}))
        resolved = graph.resolve_config(graph.current_node)
        env = navix_full_jit_env_factory(resolved)

        farm_rewards, farm_positions = _step_trace(env, _farm_actions())
        direct_rewards, direct_positions = _step_trace(env, _PATH_THROUGH_CELL_TO_GOAL)
        farm_entries = occupancy_entries(occupancy_mask(farm_positions, CELL_ENTRY_POSITION))
        direct_entries = occupancy_entries(occupancy_mask(direct_positions, CELL_ENTRY_POSITION))

        assert resolved.environment.env_id == EMPTY_VARIANT_ENV_ID
        assert resolved.environment.scenario_reward_id == FLOOR_CELL_SCENARIO_ID
        assert resolved.environment.reward is not None
        assert resolved.environment.reward.goal_reached == _SOLVABLE_GOAL_WEIGHT
        assert float(direct_rewards[0]) == pytest.approx(0.5)
        assert direct_entries <= 1
        assert farm_entries > direct_entries
        assert _discounted_return(direct_rewards) > _discounted_return(farm_rewards)

    def test_weight_schema_has_no_operational_cap(self) -> None:
        properties = RewardWeightsConfig.model_json_schema()["properties"]

        for name in RewardWeightsConfig.model_fields:
            schema = properties[name]
            assert "maximum" not in schema
            assert schema.get("minimum") == 0.0
