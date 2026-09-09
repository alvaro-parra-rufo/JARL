"""Canonical Navix env ids and registry alias resolution."""

from __future__ import annotations

import navix as nx

__all__ = [
    "canonical_navix_env_id",
    "canonical_navix_env_ids",
    "is_navix_map_installed",
    "resolve_navix_registry_id",
]

# Canonical MiniGrid-style maps (7 actions). Memory stays out: different action set.
# Fetch / GoToObject / PutNear / GoToDoor stay out: the goal is named in the mission
# and PPO observations have no instruction channel.
_CANONICAL_NAVIX_ENV_IDS: tuple[str, ...] = (
    "Navix-DistShift1-v0",
    "Navix-DistShift2-v0",
    "Navix-DoorKey-5x5-v0",
    "Navix-DoorKey-6x6-v0",
    "Navix-DoorKey-8x8-v0",
    "Navix-DoorKey-16x16-v0",
    "Navix-DoorKey-5x5-Random-v0",
    "Navix-DoorKey-6x6-Random-v0",
    "Navix-DoorKey-8x8-Random-v0",
    "Navix-DoorKey-16x16-Random-v0",
    "Navix-Dynamic-Obstacles-5x5-v0",
    "Navix-Dynamic-Obstacles-6x6-v0",
    "Navix-Dynamic-Obstacles-8x8-v0",
    "Navix-Dynamic-Obstacles-16x16-v0",
    "Navix-Dynamic-Obstacles-Random-5x5-v0",
    "Navix-Dynamic-Obstacles-Random-6x6-v0",
    "Navix-Dynamic-Obstacles-Random-8x8-v0",
    "Navix-Dynamic-Obstacles-Random-16x16-v0",
    "Navix-Empty-5x5-v0",
    "Navix-Empty-6x6-v0",
    "Navix-Empty-8x8-v0",
    "Navix-Empty-16x16-v0",
    "Navix-Empty-Random-5x5-v0",
    "Navix-Empty-Random-6x6-v0",
    "Navix-Empty-Random-8x8-v0",
    "Navix-Empty-Random-16x16-v0",
    "Navix-FourRooms-v0",
    "Navix-FourRooms-7x7-v0",
    "Navix-FourRooms-9x9-v0",
    "Navix-FourRooms-11x11-v0",
    "Navix-FourRooms-13x13-v0",
    "Navix-FourRooms-15x15-v0",
    "Navix-FourRooms-17x17-v0",
    # Instruction-mission maps stay out: PPO never sees the mission text.
    # "Navix-GoToDoor-5x5-v0",
    # "Navix-GoToDoor-6x6-v0",
    # "Navix-GoToDoor-8x8-v0",
    # "Navix-GoToObject-6x6-N2-v0",
    # "Navix-GoToObject-8x8-N2-v0",
    "Navix-KeyCorridorS3R1-v0",
    "Navix-KeyCorridorS3R2-v0",
    "Navix-KeyCorridorS3R3-v0",
    "Navix-KeyCorridorS4R3-v0",
    "Navix-KeyCorridorS5R3-v0",
    "Navix-KeyCorridorS6R3-v0",
    "Navix-LavaGap-S5-v0",
    "Navix-LavaGap-S6-v0",
    "Navix-LavaGap-S7-v0",
    "Navix-Crossings-S9N1-v0",
    "Navix-Crossings-S9N2-v0",
    "Navix-Crossings-S9N3-v0",
    "Navix-Crossings-S11N5-v0",
    "Navix-LavaCrossing-S9N1-v0",
    "Navix-LavaCrossing-S9N2-v0",
    "Navix-LavaCrossing-S9N3-v0",
    "Navix-LavaCrossing-S11N5-v0",
    # Instruction-mission maps stay out: PPO never sees the mission text.
    # "Navix-Fetch-5x5-N2-v0",
    # "Navix-Fetch-6x6-N2-v0",
    # "Navix-Fetch-8x8-N3-v0",
    # "Navix-PutNear-6x6-N2-v0",
    # "Navix-PutNear-8x8-N3-v0",
    "Navix-RedBlueDoors-6x6-v0",
    "Navix-RedBlueDoors-8x8-v0",
    "Navix-Unlock-v0",
    "Navix-UnlockPickup-v0",
    "Navix-BlockedUnlockPickup-v0",
    "Navix-LockedRoom-v0",
    "Navix-MultiRoom-N2-S4-v0",
    "Navix-MultiRoom-N4-S5-v0",
    "Navix-MultiRoom-N6-v0",
    "Navix-ObstructedMaze-1Dl-v0",
    "Navix-ObstructedMaze-1Dlh-v0",
    "Navix-ObstructedMaze-1Dlhb-v0",
    "Navix-ObstructedMaze-2Dl-v0",
    "Navix-ObstructedMaze-2Dlh-v0",
    "Navix-ObstructedMaze-2Dlhb-v0",
    "Navix-ObstructedMaze-1Q-v0",
    "Navix-ObstructedMaze-2Q-v0",
    "Navix-ObstructedMaze-Full-v0",
    "Navix-Playground-v0",
)

_REGISTRY_ALIASES: dict[str, str] = {
    "Navix-DoorKey-5x5-Random-v0": "Navix-DoorKey-Random-5x5-v0",
    "Navix-DoorKey-6x6-Random-v0": "Navix-DoorKey-Random-6x6-v0",
    "Navix-DoorKey-8x8-Random-v0": "Navix-DoorKey-Random-8x8-v0",
    "Navix-DoorKey-16x16-Random-v0": "Navix-DoorKey-Random-16x16-v0",
    "Navix-Dynamic-Obstacles-Random-5x5-v0": "Navix-Dynamic-Obstacles-5x5-Random-v0",
    "Navix-Dynamic-Obstacles-Random-6x6-v0": "Navix-Dynamic-Obstacles-6x6-Random-v0",
    "Navix-LavaGap-S5-v0": "Navix-LavaGapS5-v0",
    "Navix-LavaGap-S6-v0": "Navix-LavaGapS6-v0",
    "Navix-LavaGap-S7-v0": "Navix-LavaGapS7-v0",
    "Navix-Crossings-S9N1-v0": "Navix-SimpleCrossingS9N1-v0",
    "Navix-Crossings-S9N2-v0": "Navix-SimpleCrossingS9N2-v0",
    "Navix-Crossings-S9N3-v0": "Navix-SimpleCrossingS9N3-v0",
    "Navix-Crossings-S11N5-v0": "Navix-SimpleCrossingS11N5-v0",
    "Navix-LavaCrossing-S9N1-v0": "Navix-LavaCrossingS9N1-v0",
    "Navix-LavaCrossing-S9N2-v0": "Navix-LavaCrossingS9N2-v0",
    "Navix-LavaCrossing-S9N3-v0": "Navix-LavaCrossingS9N3-v0",
    "Navix-LavaCrossing-S11N5-v0": "Navix-LavaCrossingS11N5-v0",
}

_REGISTRY_TO_CANONICAL: dict[str, str] = {
    registry_id: canonical_id for canonical_id, registry_id in _REGISTRY_ALIASES.items()
}


def canonical_navix_env_ids() -> tuple[str, ...]:
    """Return all canonical Navix environment ids."""
    return _CANONICAL_NAVIX_ENV_IDS


def canonical_navix_env_id(env_id: str) -> str:
    """Normalize a Navix env id to the canonical catalog name."""
    return _REGISTRY_TO_CANONICAL.get(env_id, env_id)


def resolve_navix_registry_id(env_id: str) -> str:
    """Resolve a canonical or legacy env id to the local Navix registry id.

    Args:
        env_id: Canonical or registry Navix environment identifier.

    Returns:
        Environment id accepted by ``navix.make``.

    Raises:
        KeyError: If the map is not installed in the local Navix package.
    """
    canonical_id = canonical_navix_env_id(env_id)
    registry_id = _REGISTRY_ALIASES.get(canonical_id, canonical_id)
    if registry_id in nx.registry():
        return registry_id
    msg = f"Navix map is not installed locally: {env_id!r} (registry id {registry_id!r})."
    raise KeyError(msg)


def is_navix_map_installed(env_id: str) -> bool:
    """Return whether ``env_id`` is available in the local Navix package."""
    try:
        resolve_navix_registry_id(env_id)
    except KeyError:
        return False
    return True
