"""Navix environment catalog for discovery and agentic tooling."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from jarl.envs.navix.aliases import (
    canonical_navix_env_id,
    canonical_navix_env_ids,
    is_navix_map_installed,
    resolve_navix_registry_id,
)
from jarl.envs.navix.catalog import list_registered_maps

__all__ = [
    "NavixMapCategorySlug",
    "NavixMapInfo",
    "canonical_navix_env_id",
    "canonical_navix_env_ids",
    "infer_navix_map_category",
    "is_navix_map_installed",
    "list_navix_map_infos",
    "navix_map_categories",
    "navix_map_category_slugs",
    "resolve_navix_registry_id",
]

NavixMapCategorySlug = Literal[
    "dist_shift",
    "door_key",
    "dynamic_obstacles",
    "empty",
    "four_rooms",
    "key_corridor",
    "lava_gap",
    "crossings",
    "lava_crossing",
    "red_blue_doors",
    "unlock",
    "locked_room",
    "multi_room",
    "obstructed_maze",
    "playground",
]

_CATEGORY_ORDER: tuple[str, ...] = (
    "dist_shift",
    "door_key",
    "dynamic_obstacles",
    "empty",
    "four_rooms",
    "key_corridor",
    "lava_gap",
    "crossings",
    "lava_crossing",
    "red_blue_doors",
    "unlock",
    "locked_room",
    "multi_room",
    "obstructed_maze",
    "playground",
)

_CATEGORY_LABELS: dict[str, str] = {
    "empty": "Empty room",
    "door_key": "Door and key",
    "four_rooms": "Four rooms",
    "key_corridor": "Key corridor",
    "crossings": "Crossings",
    "dist_shift": "Distribution shift",
    "lava_gap": "Lava gap",
    "dynamic_obstacles": "Dynamic obstacles",
    "lava_crossing": "Lava crossing",
    "red_blue_doors": "Red and blue doors",
    "unlock": "Unlock",
    "locked_room": "Locked room",
    "multi_room": "Multi room",
    "obstructed_maze": "Obstructed maze",
    "playground": "Playground",
}

_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "empty": ("empty", "vacío", "vacio", "room", "habitación", "habitacion"),
    "door_key": ("door", "key", "llave", "puerta", "doorkey"),
    "four_rooms": ("four", "rooms", "cuartos", "habitaciones"),
    "key_corridor": ("key", "llave", "corridor", "pasillo"),
    "crossings": ("crossing", "crossings", "cruce", "muros", "walls"),
    "dist_shift": ("distshift", "distribution", "shift", "distribución", "distribucion"),
    "lava_gap": ("lava", "gap", "hueco", "lava_gap"),
    "dynamic_obstacles": ("dynamic", "obstacles", "obstáculos", "obstaculos", "móvil", "movil"),
    "lava_crossing": ("lava", "crossing", "cruce", "lava_crossing"),
    "red_blue_doors": ("red", "blue", "ordered", "puertas", "red_blue"),
    "unlock": ("unlock", "desbloquear", "pickup"),
    "locked_room": ("locked", "cerrada", "locked_room"),
    "multi_room": ("multi", "rooms", "habitaciones", "multi_room"),
    "obstructed_maze": ("obstructed", "maze", "laberinto", "box", "obstructed_maze"),
    "playground": ("playground", "sandbox", "patio"),
}

# Heuristic relative difficulty in [1, 100] for curriculum / agent filtering.
_DIFFICULTIES: dict[str, int] = {
    # Empty
    "Navix-Empty-5x5-v0": 5,
    "Navix-Empty-6x6-v0": 10,
    "Navix-Empty-8x8-v0": 15,
    "Navix-Empty-16x16-v0": 25,
    "Navix-Empty-Random-5x5-v0": 10,
    "Navix-Empty-Random-6x6-v0": 15,
    "Navix-Empty-Random-8x8-v0": 20,
    "Navix-Empty-Random-16x16-v0": 30,
    # DoorKey
    "Navix-DoorKey-5x5-v0": 20,
    "Navix-DoorKey-6x6-v0": 30,
    "Navix-DoorKey-8x8-v0": 45,
    "Navix-DoorKey-16x16-v0": 70,
    "Navix-DoorKey-5x5-Random-v0": 28,
    "Navix-DoorKey-6x6-Random-v0": 38,
    "Navix-DoorKey-8x8-Random-v0": 55,
    "Navix-DoorKey-16x16-Random-v0": 80,
    # FourRooms
    "Navix-FourRooms-v0": 40,
    "Navix-FourRooms-7x7-v0": 25,
    "Navix-FourRooms-9x9-v0": 28,
    "Navix-FourRooms-11x11-v0": 32,
    "Navix-FourRooms-13x13-v0": 36,
    "Navix-FourRooms-15x15-v0": 40,
    "Navix-FourRooms-17x17-v0": 44,
    # DistShift
    "Navix-DistShift1-v0": 40,
    "Navix-DistShift2-v0": 40,
    # LavaGap
    "Navix-LavaGap-S5-v0": 30,
    "Navix-LavaGap-S6-v0": 35,
    "Navix-LavaGap-S7-v0": 40,
    # Dynamic Obstacles
    "Navix-Dynamic-Obstacles-5x5-v0": 25,
    "Navix-Dynamic-Obstacles-6x6-v0": 35,
    "Navix-Dynamic-Obstacles-8x8-v0": 50,
    "Navix-Dynamic-Obstacles-16x16-v0": 75,
    "Navix-Dynamic-Obstacles-Random-5x5-v0": 32,
    "Navix-Dynamic-Obstacles-Random-6x6-v0": 42,
    "Navix-Dynamic-Obstacles-Random-8x8-v0": 58,
    "Navix-Dynamic-Obstacles-Random-16x16-v0": 85,
    # KeyCorridor
    "Navix-KeyCorridorS3R1-v0": 35,
    "Navix-KeyCorridorS3R2-v0": 45,
    "Navix-KeyCorridorS3R3-v0": 55,
    "Navix-KeyCorridorS4R3-v0": 65,
    "Navix-KeyCorridorS5R3-v0": 75,
    "Navix-KeyCorridorS6R3-v0": 85,
    # Crossings
    "Navix-Crossings-S9N1-v0": 30,
    "Navix-Crossings-S9N2-v0": 40,
    "Navix-Crossings-S9N3-v0": 50,
    "Navix-Crossings-S11N5-v0": 65,
    # LavaCrossing
    "Navix-LavaCrossing-S9N1-v0": 40,
    "Navix-LavaCrossing-S9N2-v0": 50,
    "Navix-LavaCrossing-S9N3-v0": 60,
    "Navix-LavaCrossing-S11N5-v0": 75,
    # RedBlueDoors
    "Navix-RedBlueDoors-6x6-v0": 35,
    "Navix-RedBlueDoors-8x8-v0": 48,
    # Unlock
    "Navix-Unlock-v0": 40,
    "Navix-UnlockPickup-v0": 55,
    "Navix-BlockedUnlockPickup-v0": 70,
    # LockedRoom
    "Navix-LockedRoom-v0": 75,
    # MultiRoom
    "Navix-MultiRoom-N2-S4-v0": 45,
    "Navix-MultiRoom-N4-S5-v0": 65,
    "Navix-MultiRoom-N6-v0": 80,
    # ObstructedMaze
    "Navix-ObstructedMaze-1Dl-v0": 60,
    "Navix-ObstructedMaze-1Dlh-v0": 68,
    "Navix-ObstructedMaze-1Dlhb-v0": 75,
    "Navix-ObstructedMaze-2Dl-v0": 78,
    "Navix-ObstructedMaze-2Dlh-v0": 85,
    "Navix-ObstructedMaze-2Dlhb-v0": 90,
    "Navix-ObstructedMaze-1Q-v0": 88,
    "Navix-ObstructedMaze-2Q-v0": 94,
    "Navix-ObstructedMaze-Full-v0": 100,
    # Playground
    "Navix-Playground-v0": 95,
}


# Canonical Navix env ids from https://epignatelli.com/navix/home/environments.html
_CANONICAL_NAVIX_ENV_IDS = canonical_navix_env_ids()

_DESCRIPTIONS: dict[str, str] = {
    "Navix-DistShift1-v0": "Distribution shift with 1 goal.",
    "Navix-DistShift2-v0": "Distribution shift with 2 goals.",
    "Navix-DoorKey-5x5-v0": "5x5 grid with a key and a door.",
    "Navix-DoorKey-6x6-v0": "6x6 grid with a key and a door.",
    "Navix-DoorKey-8x8-v0": "8x8 grid with a key and a door.",
    "Navix-DoorKey-16x16-v0": "16x16 grid with a key and a door.",
    "Navix-DoorKey-5x5-Random-v0": "5x5 door-key grid with random starts.",
    "Navix-DoorKey-6x6-Random-v0": "6x6 door-key grid with random starts.",
    "Navix-DoorKey-8x8-Random-v0": "8x8 door-key grid with random starts.",
    "Navix-DoorKey-16x16-Random-v0": "16x16 door-key grid with random starts.",
    "Navix-Dynamic-Obstacles-5x5-v0": "5x5 grid with dynamic obstacles.",
    "Navix-Dynamic-Obstacles-6x6-v0": "6x6 grid with dynamic obstacles.",
    "Navix-Dynamic-Obstacles-8x8-v0": "8x8 grid with dynamic obstacles.",
    "Navix-Dynamic-Obstacles-16x16-v0": "16x16 grid with dynamic obstacles.",
    "Navix-Dynamic-Obstacles-Random-5x5-v0": "5x5 dynamic obstacles with random starts.",
    "Navix-Dynamic-Obstacles-Random-6x6-v0": "6x6 dynamic obstacles with random starts.",
    "Navix-Dynamic-Obstacles-Random-8x8-v0": "8x8 dynamic obstacles with random starts.",
    "Navix-Dynamic-Obstacles-Random-16x16-v0": "16x16 dynamic obstacles with random starts.",
    "Navix-Empty-5x5-v0": "Empty 5x5 grid.",
    "Navix-Empty-6x6-v0": "Empty 6x6 grid.",
    "Navix-Empty-8x8-v0": "Empty 8x8 grid.",
    "Navix-Empty-16x16-v0": "Empty 16x16 grid.",
    "Navix-Empty-Random-5x5-v0": "Empty 5x5 grid with random starts.",
    "Navix-Empty-Random-6x6-v0": "Empty 6x6 grid with random starts.",
    "Navix-Empty-Random-8x8-v0": "Empty 8x8 grid with random starts.",
    "Navix-Empty-Random-16x16-v0": "Empty 16x16 grid with random starts.",
    "Navix-FourRooms-v0": "Four rooms connected by doors.",
    "Navix-FourRooms-7x7-v0": "Four rooms on a 7x7 grid.",
    "Navix-FourRooms-9x9-v0": "Four rooms on a 9x9 grid.",
    "Navix-FourRooms-11x11-v0": "Four rooms on an 11x11 grid.",
    "Navix-FourRooms-13x13-v0": "Four rooms on a 13x13 grid.",
    "Navix-FourRooms-15x15-v0": "Four rooms on a 15x15 grid.",
    "Navix-FourRooms-17x17-v0": "Four rooms on a 17x17 grid.",
    "Navix-KeyCorridorS3R1-v0": "Corridor with a key 3 cells away (1 room).",
    "Navix-KeyCorridorS3R2-v0": "Corridor with a key 3 cells away (2 rooms).",
    "Navix-KeyCorridorS3R3-v0": "Corridor with a key 3 cells away (3 rooms).",
    "Navix-KeyCorridorS4R3-v0": "Corridor with a key 4 cells away (3 rooms).",
    "Navix-KeyCorridorS5R3-v0": "Corridor with a key 5 cells away (3 rooms).",
    "Navix-KeyCorridorS6R3-v0": "Corridor with a key 6 cells away (3 rooms).",
    "Navix-LavaGap-S5-v0": "Lava gap in a 5x5 room.",
    "Navix-LavaGap-S6-v0": "Lava gap in a 6x6 room.",
    "Navix-LavaGap-S7-v0": "Lava gap in a 7x7 room.",
    "Navix-Crossings-S9N1-v0": "9x9 room with 1 wall crossing it.",
    "Navix-Crossings-S9N2-v0": "9x9 room with 2 walls crossing it.",
    "Navix-Crossings-S9N3-v0": "9x9 room with 3 walls crossing it.",
    "Navix-Crossings-S11N5-v0": "11x11 room with 5 walls crossing it.",
    "Navix-LavaCrossing-S9N1-v0": "9x9 room with 1 lava crossing.",
    "Navix-LavaCrossing-S9N2-v0": "9x9 room with 2 lava crossings.",
    "Navix-LavaCrossing-S9N3-v0": "9x9 room with 3 lava crossings.",
    "Navix-LavaCrossing-S11N5-v0": "11x11 room with 5 lava crossings.",
    "Navix-RedBlueDoors-6x6-v0": "6x6 grid; open the red door then the blue door.",
    "Navix-RedBlueDoors-8x8-v0": "8x8 grid; open the red door then the blue door.",
    "Navix-Unlock-v0": "Unlock a locked door.",
    "Navix-UnlockPickup-v0": "Unlock a door and pick up the object behind it.",
    "Navix-BlockedUnlockPickup-v0": "Move a blocking ball, unlock, then pick up the object.",
    "Navix-LockedRoom-v0": "Locate a locked room and open it.",
    "Navix-MultiRoom-N2-S4-v0": "Two connected rooms of size 4.",
    "Navix-MultiRoom-N4-S5-v0": "Four connected rooms of size 5.",
    "Navix-MultiRoom-N6-v0": "Six connected rooms.",
    "Navix-ObstructedMaze-1Dl-v0": "Obstructed maze with one locked door.",
    "Navix-ObstructedMaze-1Dlh-v0": "Obstructed maze with one locked door hidden by a box.",
    "Navix-ObstructedMaze-1Dlhb-v0": "Obstructed maze with one locked door, hidden key, and blocking ball.",
    "Navix-ObstructedMaze-2Dl-v0": "Obstructed maze with two locked doors.",
    "Navix-ObstructedMaze-2Dlh-v0": "Obstructed maze with two locked doors hidden by boxes.",
    "Navix-ObstructedMaze-2Dlhb-v0": "Obstructed maze with two locked doors, hidden keys, and blocking balls.",
    "Navix-ObstructedMaze-1Q-v0": "Obstructed maze covering one quadrant.",
    "Navix-ObstructedMaze-2Q-v0": "Obstructed maze covering two quadrants.",
    "Navix-ObstructedMaze-Full-v0": "Full obstructed maze.",
    "Navix-Playground-v0": "Large sandbox with many objects and rooms. No objective.",
}

_CATEGORY_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^Navix-Empty-"), "empty"),
    (re.compile(r"^Navix-DoorKey"), "door_key"),
    (re.compile(r"^Navix-FourRooms"), "four_rooms"),
    (re.compile(r"^Navix-KeyCorridor"), "key_corridor"),
    (re.compile(r"^Navix-Crossings"), "crossings"),
    (re.compile(r"^Navix-DistShift"), "dist_shift"),
    (re.compile(r"^Navix-LavaCrossing"), "lava_crossing"),
    (re.compile(r"^Navix-LavaGap"), "lava_gap"),
    (re.compile(r"^Navix-Dynamic-Obstacles"), "dynamic_obstacles"),
    (re.compile(r"^Navix-RedBlueDoors"), "red_blue_doors"),
    (re.compile(r"^Navix-BlockedUnlockPickup"), "unlock"),
    (re.compile(r"^Navix-Unlock"), "unlock"),
    (re.compile(r"^Navix-LockedRoom"), "locked_room"),
    (re.compile(r"^Navix-MultiRoom"), "multi_room"),
    (re.compile(r"^Navix-ObstructedMaze"), "obstructed_maze"),
    (re.compile(r"^Navix-Playground"), "playground"),
)


@dataclass(frozen=True, slots=True)
class NavixMapInfo:
    """Metadata for one catalog environment id."""

    env_id: str
    category: str
    category_label: str
    description: str
    difficulty: int
    jarl_transfer_ready: bool
    installed: bool


def infer_navix_map_category(env_id: str) -> str:
    """Infer the catalog category for a Navix ``env_id``.

    Args:
        env_id: Canonical or registry Gymnasium environment identifier.

    Returns:
        Category slug used for grouping maps.

    Raises:
        ValueError: If ``env_id`` does not match a known Navix family.
    """
    canonical_id = canonical_navix_env_id(env_id)
    for pattern, category in _CATEGORY_PATTERNS:
        if pattern.search(canonical_id):
            return category
    msg = f"Unknown Navix map category for env_id: {env_id!r}."
    raise ValueError(msg)


def navix_map_category_slugs() -> tuple[str, ...]:
    """Return ordered Navix map category slugs accepted by filters."""
    return _CATEGORY_ORDER


def navix_map_categories() -> tuple[str, ...]:
    """Return ordered category slugs present in the canonical catalog."""
    installed = {
        infer_navix_map_category(env_id) for env_id in _CANONICAL_NAVIX_ENV_IDS if is_navix_map_installed(env_id)
    }
    return tuple(category for category in _CATEGORY_ORDER if category in installed)


def list_navix_map_infos(
    *,
    categories: Sequence[str] | None = None,
    jarl_transfer_ready_only: bool = False,
    installed_only: bool = False,
    query: str | None = None,
    difficulty_min: int | None = None,
    difficulty_max: int | None = None,
) -> tuple[NavixMapInfo, ...]:
    """List canonical Navix maps from the project catalog.

    Args:
        categories: Optional category slug filters (OR). Omit or pass empty for all.
        jarl_transfer_ready_only: When ``True``, keep only JARL transfer-ready maps.
        installed_only: When ``True``, keep only maps available in the local Navix package.
        query: Optional case-insensitive keyword filter over id, description, category,
            labels and search keywords. All whitespace-separated tokens must match.
        difficulty_min: Inclusive lower bound on heuristic difficulty ``[1, 100]``.
        difficulty_max: Inclusive upper bound on heuristic difficulty ``[1, 100]``.

    Returns:
        Matching map metadata in canonical catalog order.

    Raises:
        ValueError: If any entry in ``categories`` is unknown, or difficulty bounds
            are invalid.
    """
    category_filter = _normalize_category_filter(categories)
    min_difficulty, max_difficulty = _normalize_difficulty_bounds(difficulty_min, difficulty_max)
    jarl_ready = {contract.env_id for contract in list_registered_maps()}
    query_tokens = _query_tokens(query)
    results: list[NavixMapInfo] = []

    for env_id in _CANONICAL_NAVIX_ENV_IDS:
        map_category = infer_navix_map_category(env_id)
        if category_filter is not None and map_category not in category_filter:
            continue
        installed = is_navix_map_installed(env_id)
        if installed_only and not installed:
            continue
        transfer_ready = env_id in jarl_ready
        if jarl_transfer_ready_only and not transfer_ready:
            continue
        difficulty = _DIFFICULTIES[env_id]
        if difficulty < min_difficulty or difficulty > max_difficulty:
            continue
        description = _DESCRIPTIONS[env_id]
        category_label = _CATEGORY_LABELS[map_category]
        if query_tokens and not _matches_query_tokens(
            _search_haystack(
                env_id=env_id,
                category=map_category,
                category_label=category_label,
                description=description,
            ),
            query_tokens,
        ):
            continue
        results.append(
            NavixMapInfo(
                env_id=env_id,
                category=map_category,
                category_label=category_label,
                description=description,
                difficulty=difficulty,
                jarl_transfer_ready=transfer_ready,
                installed=installed,
            )
        )

    return tuple(results)


def _normalize_category_filter(categories: Sequence[str] | None) -> frozenset[str] | None:
    """Return accepted category slugs, or ``None`` when no category filter applies."""
    if not categories:
        return None
    unknown = sorted({slug for slug in categories if slug not in _CATEGORY_LABELS})
    if unknown:
        choices = ", ".join(f"{slug} ({_CATEGORY_LABELS[slug]})" for slug in _CATEGORY_ORDER)
        unknown_text = ", ".join(repr(slug) for slug in unknown)
        msg = (
            f"Unknown Navix map categor{'ies' if len(unknown) > 1 else 'y'} {unknown_text}. "
            "Look at the user's request context and pick the best matching category "
            f"key(s) from these keys: {choices}. Call again with the exact key(s)."
        )
        raise ValueError(msg)
    return frozenset(categories)


def _normalize_difficulty_bounds(
    difficulty_min: int | None,
    difficulty_max: int | None,
) -> tuple[int, int]:
    """Return inclusive difficulty bounds validated in ``[1, 100]``."""
    lower = 1 if difficulty_min is None else difficulty_min
    upper = 100 if difficulty_max is None else difficulty_max
    max_bound = 100
    if not 1 <= lower <= max_bound:
        msg = f"difficulty_min must be in [1, {max_bound}], got {difficulty_min!r}."
        raise ValueError(msg)
    if not 1 <= upper <= max_bound:
        msg = f"difficulty_max must be in [1, {max_bound}], got {difficulty_max!r}."
        raise ValueError(msg)
    if lower > upper:
        msg = f"difficulty_min ({lower}) cannot be greater than difficulty_max ({upper})."
        raise ValueError(msg)
    return lower, upper


def _query_tokens(query: str | None) -> tuple[str, ...]:
    """Split a keyword query into casefolded tokens."""
    if query is None:
        return ()
    return tuple(token for token in query.casefold().split() if token)


def _search_haystack(
    *,
    env_id: str,
    category: str,
    category_label: str,
    description: str,
) -> str:
    """Build the searchable text for one map."""
    keywords = " ".join(_CATEGORY_KEYWORDS.get(category, ()))
    parts = [env_id, category, category_label, description, keywords]
    return " ".join(parts).casefold()


def _matches_query_tokens(haystack: str, tokens: Sequence[str]) -> bool:
    """Return whether every query token appears in ``haystack``."""
    return all(token in haystack for token in tokens)
