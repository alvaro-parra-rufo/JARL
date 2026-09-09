"""Tests for Navix map listing operation."""

from __future__ import annotations

import pytest

from jarl.operations.env.navix_maps import NavixMapsRequest, navix_maps


class TestNavixMaps:
    def test_navix_maps_lists_catalog(self) -> None:
        response = navix_maps(NavixMapsRequest())

        compact = response.to_compact_dict()

        assert len(response.maps) > 0
        assert compact["maps"][0]["env_id"].startswith("Navix-")

    def test_navix_maps_filters_by_categories(self) -> None:
        response = navix_maps(NavixMapsRequest(categories=["door_key", "lava_gap"]))

        compact = response.to_compact_dict()

        assert response.maps
        assert {item.category for item in response.maps} == {"door_key", "lava_gap"}
        assert all(
            item["env_id"].startswith("Navix-DoorKey") or item["env_id"].startswith("Navix-LavaGap")
            for item in compact["maps"]
        )
        assert all(isinstance(item["difficulty"], int) for item in compact["maps"])

    def test_navix_maps_filters_by_query(self) -> None:
        response = navix_maps(NavixMapsRequest(query="lava"))

        assert response.maps
        assert {item.category for item in response.maps} == {"lava_gap", "lava_crossing"}

    def test_navix_maps_rejects_unknown_category(self) -> None:
        with pytest.raises(ValueError, match="Look at the user's request context"):
            navix_maps(NavixMapsRequest(categories=["invalid"]))
