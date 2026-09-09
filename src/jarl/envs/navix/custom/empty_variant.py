"""Empty-like 5x5 Navix map used by the floor-cell overlay scenario.

The map contributes geometry only. Native ``reward_fn`` / ``termination_fn`` are
``on_goal_reached``. Scenario overlay payment is applied by the registered spec,
not by this constructor.
"""

from __future__ import annotations

from typing import Any

from navix.environments.empty import Room
from navix.environments.registry import register_env
from navix.observations import symbolic
from navix.rewards import on_goal_reached
from navix.terminations import on_goal_reached as terminate_on_goal_reached

from jarl.envs.navix.catalog import NAVIX_DISCRETE_ACTION_NAMES, NavixMapContract, register_map

__all__ = [
    "CELL_ENTRY_POSITION",
    "EMPTY_VARIANT_ENV_ID",
    "EMPTY_VARIANT_HEIGHT",
    "EMPTY_VARIANT_WIDTH",
    "EmptyVariant",
]

EMPTY_VARIANT_ENV_ID = "Navix-EmptyVariant-5x5-v0"
"""Gymnasium id for the 5x5 empty-like variant map."""

EMPTY_VARIANT_HEIGHT = 5
"""Grid height of ``Navix-EmptyVariant-5x5-v0``, including walls."""

EMPTY_VARIANT_WIDTH = 5
"""Grid width of ``Navix-EmptyVariant-5x5-v0``, including walls."""

CELL_ENTRY_POSITION = (1, 2)
"""Walkable floor cell used by the registered floor-cell overlay geometry."""


class EmptyVariant(Room):
    """5x5 empty room with a fixed spawn, a real Goal, and a documented floor cell."""


def _make_empty_variant(**kwargs: Any) -> EmptyVariant:
    """Build the 5x5 variant with native ``on_goal_reached`` reward and termination.

    Overlay payment is applied later by ``effective_reward_fn``, not here.
    """
    observation_fn = kwargs.pop("observation_fn", symbolic)
    reward_fn = kwargs.pop("reward_fn", on_goal_reached)
    termination_fn = kwargs.pop("termination_fn", terminate_on_goal_reached)
    return EmptyVariant.create(
        height=EMPTY_VARIANT_HEIGHT,
        width=EMPTY_VARIANT_WIDTH,
        random_start=False,
        observation_fn=observation_fn,
        reward_fn=reward_fn,
        termination_fn=termination_fn,
        **kwargs,
    )


def _register() -> None:
    """Register the Gymnasium id and the Empty-5x5 transfer contract."""
    register_env(EMPTY_VARIANT_ENV_ID, _make_empty_variant)
    # Same I/O as Empty-5x5. Do not call get_contract here: catalog import probes
    # full_jit → scenario_rewards → this module before builtins are registered.
    register_map(
        NavixMapContract(
            env_id=EMPTY_VARIANT_ENV_ID,
            processed_obs_shape=(147,),
            action_names=NAVIX_DISCRETE_ACTION_NAMES,
            transfer_group="navix_symbolic_fp_147_7",
        )
    )


_register()
