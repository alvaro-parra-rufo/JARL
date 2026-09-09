"""Registered scenario reward overlays applied on top of the configurable mix."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import navix as nx

from jarl.envs.navix.custom.empty_variant import CELL_ENTRY_POSITION, EMPTY_VARIANT_ENV_ID
from jarl.envs.navix.reward_config import RewardWeightsConfig
from jarl.envs.navix.rewards import RewardFn, compose_reward_fn, occupancy_enter_reward, scale_reward

__all__ = [
    "EMPTY_VARIANT_ENV_ID",
    "FLOOR_CELL_SCENARIO_ID",
    "FLOOR_CELL_SCENARIO_VERSION",
    "SCENARIO_REWARD_REGISTRY",
    "ScenarioRewardError",
    "ScenarioRewardSpec",
    "effective_reward_fn",
    "resolve_scenario_reward_spec",
    "scenario_reward_fn",
]

FLOOR_CELL_SCENARIO_ID = "navix.floor_cell"
"""Stable identifier of the v1 floor-cell overlay."""

FLOOR_CELL_SCENARIO_VERSION = 1
"""Version of the v1 floor-cell overlay."""

_FLOOR_CELL_SCALE = 0.5


class ScenarioRewardError(ValueError):
    """Raised when a scenario overlay key is unknown or incompatible with ``env_id``."""


@dataclass(frozen=True, slots=True)
class ScenarioRewardSpec:
    """Registered overlay applied on top of the configurable reward mix.

    Args:
        id: Stable overlay identifier.
        version: Overlay version.
        scale: Multiplier applied to the overlay pulse.
        compatible_env_ids: Environment ids that may receive this overlay.
        position: Grid coordinates of the overlay pulse.
    """

    id: str
    version: int
    scale: float
    compatible_env_ids: tuple[str, ...]
    position: tuple[int, int]


SCENARIO_REWARD_REGISTRY: dict[tuple[str, int], ScenarioRewardSpec] = {
    (FLOOR_CELL_SCENARIO_ID, FLOOR_CELL_SCENARIO_VERSION): ScenarioRewardSpec(
        id=FLOOR_CELL_SCENARIO_ID,
        version=FLOOR_CELL_SCENARIO_VERSION,
        scale=_FLOOR_CELL_SCALE,
        compatible_env_ids=(EMPTY_VARIANT_ENV_ID,),
        position=CELL_ENTRY_POSITION,
    ),
}
"""Explicit overlay catalog keyed by ``(id, version)``."""


def resolve_scenario_reward_spec(
    env_id: str,
    scenario_reward_id: str | None,
    scenario_reward_version: int | None,
) -> ScenarioRewardSpec | None:
    """Resolve a registered overlay for ``env_id``.

    Args:
        env_id: Gymnasium environment identifier about to be constructed.
        scenario_reward_id: Overlay id, or ``None`` when no overlay is configured.
        scenario_reward_version: Overlay version, or ``None`` when no overlay is configured.

    Returns:
        The registered spec, or ``None`` when both id and version are ``None``.

    Raises:
        ScenarioRewardError: If the key is unknown or ``env_id`` is incompatible.
    """
    if scenario_reward_id is None and scenario_reward_version is None:
        return None
    if scenario_reward_id is None or scenario_reward_version is None:
        msg = "scenario_reward_id and scenario_reward_version must both be set or both be None."
        raise ScenarioRewardError(msg)
    spec = SCENARIO_REWARD_REGISTRY.get((scenario_reward_id, scenario_reward_version))
    if spec is None:
        msg = f"Unknown ScenarioRewardSpec {scenario_reward_id} v{scenario_reward_version}."
        raise ScenarioRewardError(msg)
    if env_id not in spec.compatible_env_ids:
        msg = f"ScenarioRewardSpec {scenario_reward_id} v{scenario_reward_version} incompatible with env_id {env_id!r}."
        raise ScenarioRewardError(msg)
    return spec


def scenario_reward_fn(spec: ScenarioRewardSpec) -> RewardFn:
    """Return the overlay pulse scaled by ``spec.scale``."""
    return scale_reward(occupancy_enter_reward(spec.position), spec.scale)


def effective_reward_fn(
    weights: RewardWeightsConfig,
    spec: ScenarioRewardSpec | None,
) -> RewardFn:
    """Return ``r_configurable + r_scenario`` for one Navix timestep."""
    configurable = compose_reward_fn(weights)
    if spec is None:
        return configurable
    return nx.rewards.compose(configurable, scenario_reward_fn(spec), operator=jnp.sum)
