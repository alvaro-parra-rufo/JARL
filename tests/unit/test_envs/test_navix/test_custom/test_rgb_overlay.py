"""Tests for EmptyVariant full-view RGB debug tint."""

from __future__ import annotations

import numpy as np
import pytest
from navix.rendering.registry import TILE_SIZE

from jarl.envs.navix.custom.empty_variant import (
    CELL_ENTRY_POSITION,
    EMPTY_VARIANT_ENV_ID,
    EMPTY_VARIANT_HEIGHT,
    EMPTY_VARIANT_WIDTH,
)
from jarl.envs.navix.custom.rgb_overlay import (
    FLOOR_CELL_DEBUG_COLOR,
    apply_empty_variant_floor_tint,
    tint_grid_cell_rgb,
)

_FRAME_H = EMPTY_VARIANT_HEIGHT * TILE_SIZE
_FRAME_W = EMPTY_VARIANT_WIDTH * TILE_SIZE
_CELL_ROW, _CELL_COL = CELL_ENTRY_POSITION
_ROW = slice(_CELL_ROW * TILE_SIZE, (_CELL_ROW + 1) * TILE_SIZE)
_COL = slice(_CELL_COL * TILE_SIZE, (_CELL_COL + 1) * TILE_SIZE)


def _blank_frames(time_steps: int = 2) -> np.ndarray:
    return np.zeros((time_steps, _FRAME_H, _FRAME_W, 3), dtype=np.uint8)


class TestTintGridCellRgb:
    """The marker paints one tile and leaves the rest of the frame unchanged."""

    def test_tints_only_the_target_cell(self) -> None:
        frames = _blank_frames(1)

        tinted = tint_grid_cell_rgb(frames, CELL_ENTRY_POSITION, alpha=1.0)

        np.testing.assert_array_equal(
            tinted[0, _ROW, _COL], np.broadcast_to(FLOOR_CELL_DEBUG_COLOR, (TILE_SIZE, TILE_SIZE, 3))
        )
        assert int(tinted[0, 0, 0].sum()) == 0
        np.testing.assert_array_equal(frames, _blank_frames(1))

    def test_rejects_out_of_bounds_cell(self) -> None:
        with pytest.raises(ValueError, match="outside"):
            tint_grid_cell_rgb(_blank_frames(1), (0, 9))


class TestApplyEmptyVariantFloorTint:
    """Tint applies only to full-view EmptyVariant clips."""

    def test_full_view_empty_variant_is_tinted(self) -> None:
        frames = _blank_frames(1)

        tinted = apply_empty_variant_floor_tint(
            frames,
            env_id=EMPTY_VARIANT_ENV_ID,
            view_mode="full",
        )

        assert int(tinted[0, _ROW, _COL].sum()) > 0
        assert int(tinted[0, 0, 0].sum()) == 0

    def test_first_person_is_unchanged(self) -> None:
        frames = _blank_frames(1)

        tinted = apply_empty_variant_floor_tint(
            frames,
            env_id=EMPTY_VARIANT_ENV_ID,
            view_mode="first_person",
        )

        np.testing.assert_array_equal(tinted, frames)

    def test_other_env_id_is_unchanged(self) -> None:
        frames = _blank_frames(1)

        tinted = apply_empty_variant_floor_tint(
            frames,
            env_id="Navix-Empty-5x5-v0",
            view_mode="full",
        )

        np.testing.assert_array_equal(tinted, frames)

    def test_occupied_cell_is_stronger_than_idle(self) -> None:
        frames = _blank_frames(2)
        positions = np.asarray([(1, 1), CELL_ENTRY_POSITION], dtype=np.int32)

        tinted = apply_empty_variant_floor_tint(
            frames,
            env_id=EMPTY_VARIANT_ENV_ID,
            view_mode="full",
            player_positions=positions,
        )

        idle = float(tinted[0, _ROW, _COL].mean())
        occupied = float(tinted[1, _ROW, _COL].mean())
        assert occupied > idle
