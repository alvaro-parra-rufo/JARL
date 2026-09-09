"""Internal host-array accessors shared by Navix postprocessors."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from jarl.envs.navix.telemetry.contracts import NavixTraceArrays, Position

__all__ = [
    "action_name",
    "bounded_items",
    "episode_length",
    "player_direction",
    "position",
    "required_state_array",
    "required_transition_array",
    "transition_array",
    "validate_action_indices",
]


def episode_length(trace: NavixTraceArrays, episode_index: int) -> int:
    """Return one episode's active transition count."""
    lengths = np.asarray(trace.episode_lengths)
    if episode_index < 0 or episode_index >= len(lengths):
        msg = f"episode_index {episode_index} is outside batch size {len(lengths)}."
        raise IndexError(msg)
    return int(lengths[episode_index])


def required_state_array(
    array: object | None,
    name: str,
    episode_index: int,
    length: int,
) -> np.ndarray:
    """Return one required state series trimmed to ``T + 1``."""
    if array is None:
        msg = f"{name} is unavailable; use NavixCaptureProfile.analysis()."
        raise ValueError(msg)
    return np.asarray(array)[episode_index, : length + 1]


def required_transition_array(
    array: object | None,
    name: str,
    episode_index: int,
    length: int,
) -> np.ndarray:
    """Return one required transition series trimmed to ``T``."""
    if array is None:
        msg = f"{name} is unavailable; use NavixCaptureProfile.analysis()."
        raise ValueError(msg)
    return np.asarray(array)[episode_index, :length]


def transition_array(
    array: object,
    episode_index: int,
    length: int,
) -> np.ndarray:
    """Return one always-present transition series trimmed to ``T``."""
    return np.asarray(array)[episode_index, :length]


def player_direction(trace: NavixTraceArrays, episode_index: int, timestep: int) -> int:
    """Return the player direction for one state timestep."""
    if trace.player_directions is None:
        msg = "player_directions is unavailable; use NavixCaptureProfile.analysis()."
        raise ValueError(msg)
    return int(np.asarray(trace.player_directions)[episode_index, timestep])


def action_name(action_index: int, action_names: Sequence[str]) -> str:
    """Resolve a previously validated action index."""
    if 0 <= action_index < len(action_names):
        return action_names[action_index]
    return f"action_{action_index}"


def validate_action_indices(actions: np.ndarray, action_names: Sequence[str]) -> None:
    """Reject action indices outside the target environment action set."""
    invalid = sorted({int(action) for action in actions if int(action) < 0 or int(action) >= len(action_names)})
    if not invalid:
        return
    msg = f"Action indices {invalid} are outside the environment action set of size {len(action_names)}."
    raise ValueError(msg)


def position(value: object) -> Position:
    """Convert an array-like grid coordinate into a typed position."""
    coordinates = np.asarray(value).reshape(-1)
    return int(coordinates[0]), int(coordinates[1])


def bounded_items[T](items: Sequence[T], limit: int) -> tuple[T, ...]:
    """Retain bounded head and tail evidence from an ordered sequence."""
    if len(items) <= limit:
        return tuple(items)
    head_count = (limit + 1) // 2
    tail_count = limit - head_count
    if tail_count == 0:
        return tuple(items[:head_count])
    return (*items[:head_count], *items[-tail_count:])
