"""List Navix maps from the canonical catalog."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from jarl.envs.navix.maps import NavixMapInfo, list_navix_map_infos
from jarl.operations.contracts.compact import dataclass_to_compact_dict

__all__ = [
    "NavixMapsRequest",
    "NavixMapsResponse",
    "navix_maps",
]


@dataclass(frozen=True, slots=True)
class NavixMapsRequest:
    """Inputs for listing Navix maps."""

    categories: Sequence[str] | None = None
    query: str | None = None
    difficulty_min: int | None = None
    difficulty_max: int | None = None
    jarl_transfer_ready_only: bool = False
    installed_only: bool = False


@dataclass(frozen=True, slots=True)
class NavixMapsResponse:
    """Outcome of ``navix_maps``."""

    maps: tuple[NavixMapInfo, ...]

    _compact_include: ClassVar[frozenset[str] | None] = None

    def to_compact_dict(self) -> dict[str, object]:
        """Return a compact projection for tools and agents."""
        return {
            "maps": [dataclass_to_compact_dict(item, include=self._compact_include) for item in self.maps],
        }


def navix_maps(request: NavixMapsRequest) -> NavixMapsResponse:
    """List Navix maps with optional category, keyword, and difficulty filters."""
    maps = list_navix_map_infos(
        categories=request.categories,
        query=request.query,
        difficulty_min=request.difficulty_min,
        difficulty_max=request.difficulty_max,
        jarl_transfer_ready_only=request.jarl_transfer_ready_only,
        installed_only=request.installed_only,
    )
    return NavixMapsResponse(maps=maps)
