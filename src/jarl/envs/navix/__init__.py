"""Navix environment adapters."""

from jarl.envs.navix.catalog import (
    NavixMapContract,
    assert_transfer_compatible,
    get_contract,
    list_registered_maps,
    register_map,
)
from jarl.envs.navix.custom import EMPTY_VARIANT_ENV_ID
from jarl.envs.navix.full_jit import (
    GeneralProperties,
    NavixFullJITEnv,
    NavixFullJITState,
    make_navix_full_jit_env,
)
from jarl.envs.navix.maps import (
    NavixMapInfo,
    canonical_navix_env_id,
    infer_navix_map_category,
    is_navix_map_installed,
    list_navix_map_infos,
    navix_map_categories,
    resolve_navix_registry_id,
)
from jarl.envs.navix.reward_config import RewardWeightsConfig
from jarl.envs.navix.rewards import (
    CHANNEL_SPECS,
    ChannelSpec,
    channel_catalog,
    compose_reward_fn,
    occupancy_entries,
    occupancy_mask,
    occupancy_steps,
)
from jarl.envs.navix.scenario_rewards import (
    ScenarioRewardError,
    ScenarioRewardSpec,
    effective_reward_fn,
    resolve_scenario_reward_spec,
)

__all__ = [
    "CHANNEL_SPECS",
    "EMPTY_VARIANT_ENV_ID",
    "ChannelSpec",
    "GeneralProperties",
    "NavixFullJITEnv",
    "NavixFullJITState",
    "NavixMapContract",
    "NavixMapInfo",
    "RewardWeightsConfig",
    "ScenarioRewardError",
    "ScenarioRewardSpec",
    "assert_transfer_compatible",
    "canonical_navix_env_id",
    "channel_catalog",
    "compose_reward_fn",
    "effective_reward_fn",
    "get_contract",
    "infer_navix_map_category",
    "is_navix_map_installed",
    "list_navix_map_infos",
    "list_registered_maps",
    "make_navix_full_jit_env",
    "navix_map_categories",
    "occupancy_entries",
    "occupancy_mask",
    "occupancy_steps",
    "register_map",
    "resolve_navix_registry_id",
    "resolve_scenario_reward_spec",
]
