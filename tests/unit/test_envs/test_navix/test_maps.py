"""Tests for Navix map discovery catalog."""

from __future__ import annotations

import pytest

from jarl.envs.navix.aliases import canonical_navix_env_ids, resolve_navix_registry_id
from jarl.envs.navix.custom import EMPTY_VARIANT_ENV_ID
from jarl.envs.navix.maps import (
    _DESCRIPTIONS,
    _DIFFICULTIES,
    infer_navix_map_category,
    list_navix_map_infos,
    navix_map_categories,
)


class TestNavixMapsCatalog:
    def test_canonical_maps_have_descriptions(self) -> None:
        missing = sorted(set(canonical_navix_env_ids()) - set(_DESCRIPTIONS))
        assert missing == []

    def test_canonical_maps_have_difficulties(self) -> None:
        missing = sorted(set(canonical_navix_env_ids()) - set(_DIFFICULTIES))
        assert missing == []
        assert all(1 <= score <= 100 for score in _DIFFICULTIES.values())

    def test_canonical_catalog_has_69_maps(self) -> None:
        assert len(canonical_navix_env_ids()) == 69

    def test_memory_maps_are_absent_from_transfer_catalog(self) -> None:
        catalog_ids = canonical_navix_env_ids()

        assert not any(env_id.startswith("Navix-Memory") for env_id in catalog_ids)

    def test_instruction_mission_maps_are_absent_from_catalog(self) -> None:
        catalog_ids = canonical_navix_env_ids()
        mission_prefixes = ("Navix-Fetch", "Navix-GoToObject", "Navix-PutNear", "Navix-GoToDoor")

        assert not any(env_id.startswith(prefix) for env_id in catalog_ids for prefix in mission_prefixes)

    def test_mission_categories_are_unknown(self) -> None:
        with pytest.raises(ValueError, match="Unknown Navix map category"):
            list_navix_map_infos(categories=["fetch"])

    def test_empty_variant_is_absent_from_llm_catalog(self) -> None:
        catalog_ids = [item.env_id for item in list_navix_map_infos()]

        assert EMPTY_VARIANT_ENV_ID not in catalog_ids
        assert EMPTY_VARIANT_ENV_ID not in canonical_navix_env_ids()

    def test_empty_variant_is_not_empty_category(self) -> None:
        with pytest.raises(ValueError, match="Unknown Navix map category"):
            infer_navix_map_category(EMPTY_VARIANT_ENV_ID)

    def test_list_navix_map_infos_returns_all_canonical_maps(self) -> None:
        maps = list_navix_map_infos()

        assert len(maps) == len(canonical_navix_env_ids())
        assert [item.env_id for item in maps] == list(canonical_navix_env_ids())

    def test_jarl_transfer_ready_filter(self) -> None:
        maps = list_navix_map_infos(jarl_transfer_ready_only=True)

        assert [item.env_id for item in maps] == list(canonical_navix_env_ids())
        assert all(item.jarl_transfer_ready for item in maps)

    def test_installed_only_filter(self) -> None:
        maps = list_navix_map_infos(installed_only=True)

        assert maps
        assert all(item.installed for item in maps)

    def test_category_filter(self) -> None:
        maps = list_navix_map_infos(categories=["empty"])

        assert maps
        assert all(item.category == "empty" for item in maps)

    def test_categories_filter_or(self) -> None:
        maps = list_navix_map_infos(categories=["empty", "lava_gap"])

        assert maps
        assert {item.category for item in maps} == {"empty", "lava_gap"}

    def test_query_filter_matches_description(self) -> None:
        maps = list_navix_map_infos(query="random starts")

        assert maps
        assert all("random starts" in item.description.casefold() for item in maps)

    def test_query_filter_matches_keyword_aliases(self) -> None:
        maps = list_navix_map_infos(query="llave")

        assert maps
        assert {item.category for item in maps} == {"door_key", "key_corridor"}

    def test_difficulty_filter(self) -> None:
        maps = list_navix_map_infos(difficulty_min=40, difficulty_max=50)

        assert maps
        assert all(40 <= item.difficulty <= 50 for item in maps)

    def test_unknown_category_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown Navix map category"):
            list_navix_map_infos(categories=["unknown"])

    @pytest.mark.parametrize(
        ("env_id", "category"),
        [
            ("Navix-Empty-5x5-v0", "empty"),
            ("Navix-DoorKey-8x8-v0", "door_key"),
            ("Navix-Crossings-S9N1-v0", "crossings"),
            ("Navix-LavaGap-S5-v0", "lava_gap"),
            ("Navix-LavaCrossing-S9N1-v0", "lava_crossing"),
            ("Navix-UnlockPickup-v0", "unlock"),
            ("Navix-BlockedUnlockPickup-v0", "unlock"),
            ("Navix-ObstructedMaze-1Dl-v0", "obstructed_maze"),
        ],
    )
    def test_infer_navix_map_category(self, env_id: str, category: str) -> None:
        assert infer_navix_map_category(env_id) == category

    def test_navix_map_categories_are_ordered(self) -> None:
        categories = navix_map_categories()

        assert categories[0] == "dist_shift"
        assert "door_key" in categories

    def test_resolve_navix_registry_id_maps_canonical_aliases(self) -> None:
        assert resolve_navix_registry_id("Navix-LavaGap-S5-v0") == "Navix-LavaGapS5-v0"
        assert resolve_navix_registry_id("Navix-Crossings-S9N1-v0") == "Navix-SimpleCrossingS9N1-v0"
        assert resolve_navix_registry_id("Navix-LavaCrossing-S9N1-v0") == "Navix-LavaCrossingS9N1-v0"
        assert resolve_navix_registry_id("Navix-DoorKey-5x5-Random-v0") == "Navix-DoorKey-Random-5x5-v0"
