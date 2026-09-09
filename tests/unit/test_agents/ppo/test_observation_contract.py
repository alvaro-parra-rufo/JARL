"""PPO and PPO-GRU consume numeric flat observations, not text."""

from __future__ import annotations

import inspect
from typing import get_type_hints

import jax
import jax.numpy as jnp

from jarl.agents.ppo.networks.factory import create_gru_policy_network, create_policy_network
from jarl.agents.ppo.networks.gru_policy import DiscreteGrUPolicy
from jarl.agents.ppo.networks.policy import DiscretePolicy
from jarl.envs.navix.full_jit import NavixFullJITEnv
from jarl.envs.types import ObservationSpaceType


class TestObservationContract:
    def test_observation_space_types_are_numeric_only(self) -> None:
        assert {item.value for item in ObservationSpaceType} == {"flat_values", "images"}

    def test_discrete_policy_call_takes_array_observations(self) -> None:
        hints = get_type_hints(DiscretePolicy.__call__)

        assert hints["observations"] is jnp.ndarray

    def test_gru_apply_one_step_takes_array_obs_and_carry(self) -> None:
        parameters = inspect.signature(DiscreteGrUPolicy.apply_one_step).parameters
        hints = get_type_hints(DiscreteGrUPolicy.apply_one_step)

        assert list(parameters) == ["self", "obs", "carry"]
        assert hints["obs"] is jax.Array
        assert "text" not in parameters
        assert "mission" not in parameters

    def test_navix_adapter_observation_pipeline_excludes_mission(self) -> None:
        init_source = inspect.getsource(NavixFullJITEnv.__init__)
        preprocess_source = inspect.getsource(NavixFullJITEnv.preprocess_observation)

        assert "symbolic_first_person" in init_source
        assert "mission" not in init_source
        assert "mission" not in preprocess_source
        assert "/ 255.0" in preprocess_source

    def test_policy_factories_require_flat_numeric_observations(self) -> None:
        assert "FLAT_VALUES" in inspect.getsource(create_policy_network)
        assert "FLAT_VALUES" in inspect.getsource(create_gru_policy_network)
