"""Tests for Navix map catalog and transfer compatibility."""

from __future__ import annotations

import pytest

from jarl.envs.navix.aliases import canonical_navix_env_ids
from jarl.envs.navix.catalog import (
    NAVIX_DISCRETE_ACTION_NAMES,
    NavixMapContract,
    assert_transfer_compatible,
    get_contract,
    list_registered_maps,
    register_map,
)
from jarl.envs.navix.custom import EMPTY_VARIANT_ENV_ID


class TestNavixCatalog:
    """Tests for built-in and custom Navix map registration."""

    def test_builtin_maps_are_registered(self) -> None:
        contract = get_contract("Navix-Empty-5x5-v0")

        assert contract.processed_obs_shape == (147,)
        assert contract.action_names == NAVIX_DISCRETE_ACTION_NAMES
        assert contract.transfer_group == "navix_symbolic_fp_147_7"

    def test_register_custom_map(self) -> None:
        custom = NavixMapContract(
            env_id="Navix-Custom-Test-v0",
            processed_obs_shape=(147,),
            action_names=NAVIX_DISCRETE_ACTION_NAMES,
            transfer_group="navix_symbolic_fp_147_7",
        )

        register_map(custom)

        assert get_contract("Navix-Custom-Test-v0") == custom
        assert custom in list_registered_maps()

    def test_canonical_maps_share_empty_transfer_contract(self) -> None:
        empty = get_contract("Navix-Empty-5x5-v0")
        registered_ids = {contract.env_id for contract in list_registered_maps()}

        assert set(canonical_navix_env_ids()) <= registered_ids
        for env_id in canonical_navix_env_ids():
            contract = get_contract(env_id)
            assert contract.processed_obs_shape == empty.processed_obs_shape
            assert contract.action_names == empty.action_names
            assert contract.transfer_group == empty.transfer_group
            assert_transfer_compatible("Navix-Empty-5x5-v0", env_id)

    def test_assert_transfer_compatible_accepts_same_group(self) -> None:
        assert_transfer_compatible("Navix-Empty-5x5-v0", "Navix-DoorKey-5x5-v0")
        assert_transfer_compatible("Navix-Empty-5x5-v0", "Navix-KeyCorridorS3R1-v0")

    def test_assert_transfer_compatible_rejects_different_group(self) -> None:
        register_map(
            NavixMapContract(
                env_id="Navix-Incompatible-Test-v0",
                processed_obs_shape=(99,),
                action_names=("a", "b"),
                transfer_group="other_group",
            )
        )

        with pytest.raises(ValueError, match="transfer groups"):
            assert_transfer_compatible("Navix-Empty-5x5-v0", "Navix-Incompatible-Test-v0")

    def test_get_contract_unknown_env_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown Navix map"):
            get_contract("Navix-Does-Not-Exist-v0")

    def test_empty_variant_matches_empty_5x5_transfer_contract(self) -> None:
        variant = get_contract(EMPTY_VARIANT_ENV_ID)
        empty = get_contract("Navix-Empty-5x5-v0")

        assert variant.processed_obs_shape == (147,)
        assert variant.processed_obs_shape == empty.processed_obs_shape
        assert variant.action_names == empty.action_names
        assert variant.transfer_group == empty.transfer_group
        assert variant in list_registered_maps()
        assert_transfer_compatible("Navix-Empty-5x5-v0", EMPTY_VARIANT_ENV_ID)
