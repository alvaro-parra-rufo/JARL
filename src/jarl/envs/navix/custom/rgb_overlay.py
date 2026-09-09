"""Full-view RGB debug tint for the EmptyVariant floor cell.

The tint is a pixel overlay on encoded videos. It does not change Navix state,
symbolic observations, or compact summaries shown to the agent.
"""

from __future__ import annotations

import numpy as np
from navix.rendering.registry import TILE_SIZE

from jarl.envs.navix.custom.empty_variant import (
    CELL_ENTRY_POSITION,
    EMPTY_VARIANT_ENV_ID,
    EMPTY_VARIANT_HEIGHT,
    EMPTY_VARIANT_WIDTH,
)

__all__ = [
    "FLOOR_CELL_DEBUG_COLOR",
    "apply_empty_variant_floor_tint",
    "tint_grid_cell_rgb",
]

FLOOR_CELL_DEBUG_COLOR = (220, 32, 180)
"""Magenta marker used on full-view EmptyVariant videos."""

_UINT8_MAX = 255.0
"""Inclusive maximum of an 8-bit RGB channel."""

_ALPHA_IDLE = 0.40
_ALPHA_OCCUPIED = 0.65
_RGB_FRAME_RANK = 4
_RGB_CHANNEL_COUNT = 3
_POSITION_RANK = 2


def tint_grid_cell_rgb(
    frames: np.ndarray,
    cell: tuple[int, int],
    *,
    color: tuple[int, int, int] = FLOOR_CELL_DEBUG_COLOR,
    alpha: float | np.ndarray = _ALPHA_IDLE,
    tile_size: int = TILE_SIZE,
) -> np.ndarray:
    """Blend ``color`` onto one grid cell of full-view RGB frames.

    Args:
        frames: RGB array shaped ``(time, height, width, 3)``.
        cell: Grid coordinates ``(row, column)`` in tile units.
        color: RGB marker blended onto the cell.
        alpha: Blend weight in ``[0, 1]``, or a per-frame vector of length ``time``.
        tile_size: Navix sprite size in pixels.

    Returns:
        A new ``uint8`` frame array. Input frames are not modified.

    Raises:
        ValueError: If ``frames`` rank, cell bounds, or ``alpha`` shape are invalid.
    """
    array = np.asarray(frames)
    if array.ndim != _RGB_FRAME_RANK or array.shape[-1] != _RGB_CHANNEL_COUNT:
        msg = f"RGB frames must have shape (time, height, width, 3), got {array.shape}."
        raise ValueError(msg)
    if tile_size <= 0:
        msg = f"tile_size must be > 0, got {tile_size}."
        raise ValueError(msg)
    height, width = int(array.shape[1]), int(array.shape[2])
    if height % tile_size != 0 or width % tile_size != 0:
        msg = f"Frame size {(height, width)} is not a multiple of tile_size {tile_size}."
        raise ValueError(msg)
    rows, cols = height // tile_size, width // tile_size
    row, col = cell
    if not (0 <= row < rows and 0 <= col < cols):
        msg = f"Cell {cell!r} is outside the {rows}x{cols} grid."
        raise ValueError(msg)

    time_steps = int(array.shape[0])
    alpha_array = np.asarray(alpha, dtype=np.float32)
    if alpha_array.shape == ():
        alpha_array = np.full((time_steps,), float(alpha_array), dtype=np.float32)
    elif alpha_array.shape != (time_steps,):
        msg = f"alpha must be a scalar or shape {(time_steps,)}, got {alpha_array.shape}."
        raise ValueError(msg)
    alpha_array = np.clip(alpha_array, 0.0, 1.0).reshape((time_steps, 1, 1, 1))

    row_slice = slice(row * tile_size, (row + 1) * tile_size)
    col_slice = slice(col * tile_size, (col + 1) * tile_size)
    blended = np.asarray(array, dtype=np.float32)
    patch = blended[:, row_slice, col_slice]
    marker = np.asarray(color, dtype=np.float32)
    blended[:, row_slice, col_slice] = (1.0 - alpha_array) * patch + alpha_array * marker
    return np.clip(blended, 0.0, _UINT8_MAX).astype(np.uint8)


def apply_empty_variant_floor_tint(
    frames: np.ndarray,
    *,
    env_id: str,
    view_mode: str | None,
    player_positions: np.ndarray | None = None,
) -> np.ndarray:
    """Tint the EmptyVariant floor cell on full-view videos; otherwise return ``frames``.

    Args:
        frames: RGB array shaped ``(time, height, width, 3)``.
        env_id: Gymnasium environment identifier of the recorded episode.
        view_mode: Navix camera used to produce ``frames``.
        player_positions: Optional ``(time, 2)`` player coordinates aligned with ``frames``.

    Returns:
        Tinted frames when the clip is a full-view EmptyVariant video; otherwise ``frames``.
    """
    if env_id != EMPTY_VARIANT_ENV_ID or view_mode != "full":
        return frames
    array = np.asarray(frames)
    expected = (EMPTY_VARIANT_HEIGHT * TILE_SIZE, EMPTY_VARIANT_WIDTH * TILE_SIZE)
    if array.ndim != _RGB_FRAME_RANK or (int(array.shape[1]), int(array.shape[2])) != expected:
        return frames
    alpha: float | np.ndarray = _ALPHA_IDLE
    if player_positions is not None:
        positions = np.asarray(player_positions, dtype=np.int32)
        if positions.ndim == _POSITION_RANK and positions.shape[0] == array.shape[0]:
            occupied = np.all(positions == np.asarray(CELL_ENTRY_POSITION, dtype=np.int32), axis=-1)
            alpha = np.where(occupied, _ALPHA_OCCUPIED, _ALPHA_IDLE).astype(np.float32)
    return tint_grid_cell_rgb(array, CELL_ENTRY_POSITION, alpha=alpha)
