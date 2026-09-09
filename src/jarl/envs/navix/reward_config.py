"""Configurable Navix reward weights.

This module owns the Pydantic schema only. Runtime primitives live in
``jarl.envs.navix.rewards``.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from jarl.config import BaseConfig

__all__ = ["RewardWeightsConfig"]


def _weight_field(description: str) -> Field:
    """Return a non-negative finite weight field."""
    return Field(ge=0.0, allow_inf_nan=False, description=description)


class RewardWeightsConfig(BaseConfig):
    """Named non-negative weights for the global Navix reward catalog.

    Each weight multiplies a channel in ``[0, 1]``. Defaults instantiate a
    task-only mix: ``goal_reached=1.0`` and every other channel at ``0.0``.
    """

    goal_reached: Annotated[
        float,
        _weight_field("Pulse when the goal-reached event fires on this step."),
    ] = 1.0
    key_pickup: Annotated[
        float,
        _weight_field("Pulse when a key is picked up on this step."),
    ] = 0.0
    door_opening: Annotated[
        float,
        _weight_field("Pulse when a door is opened on this step."),
    ] = 0.0
    door_unlock: Annotated[
        float,
        _weight_field("Pulse when a door is unlocked on this step."),
    ] = 0.0
    ball_pickup: Annotated[
        float,
        _weight_field("Pulse when a ball is picked up on this step."),
    ] = 0.0
    box_pickup: Annotated[
        float,
        _weight_field("Pulse when a box is picked up on this step."),
    ] = 0.0
    door_done: Annotated[
        float,
        _weight_field("Indicator that the agent faces the mission door. Zero when the environment has no mission."),
    ] = 0.0
    holding_key: Annotated[
        float,
        _weight_field("Indicator that the agent pocket currently holds a key."),
    ] = 0.0
    door_is_open: Annotated[
        float,
        _weight_field("Indicator that at least one door is currently open."),
    ] = 0.0
    distance_to_goal: Annotated[
        float,
        _weight_field(
            "Proximity to the nearest on-map goal: ``1 - min_d / d_max``. "
            "Zero when no goal is on the map. ``d_max = (H - 1) + (W - 1)``."
        ),
    ] = 0.0
    distance_to_key: Annotated[
        float,
        _weight_field("Proximity to the nearest on-map key. Zero when no key is on the map."),
    ] = 0.0
    distance_to_door: Annotated[
        float,
        _weight_field("Proximity to the nearest closed door. Zero when every door is open or the map has no doors."),
    ] = 0.0
    lava_clearance: Annotated[
        float,
        _weight_field("Normalized Manhattan distance to the nearest on-map lava. Zero when the map has no lava."),
    ] = 0.0
    distance_to_ball: Annotated[
        float,
        _weight_field("Proximity to the nearest on-map ball. Zero when no ball is on the map."),
    ] = 0.0
    distance_to_box: Annotated[
        float,
        _weight_field("Proximity to the nearest on-map box. Zero when no box is on the map."),
    ] = 0.0
    goal_visible: Annotated[
        float,
        _weight_field("Indicator that an on-map goal is inside the first-person crop (radius 3)."),
    ] = 0.0
    key_visible: Annotated[
        float,
        _weight_field("Indicator that an on-map key is inside the first-person crop (radius 3)."),
    ] = 0.0
    facing_goal: Annotated[
        float,
        _weight_field("Indicator that the cell in front of the agent is an on-map goal."),
    ] = 0.0
    facing_key: Annotated[
        float,
        _weight_field("Indicator that the cell in front of the agent is an on-map key."),
    ] = 0.0
    done_at_goal: Annotated[
        float,
        _weight_field("Pulse when the done action is taken while occupying or facing a goal."),
    ] = 0.0
    goal_approach: Annotated[
        float,
        _weight_field(
            "Clipped reduction in Manhattan distance to the nearest on-map goal "
            "on this step: ``clip(d_prev - d_curr, 0, 1)``. Idle or moving away "
            "yields zero."
        ),
    ] = 0.0
