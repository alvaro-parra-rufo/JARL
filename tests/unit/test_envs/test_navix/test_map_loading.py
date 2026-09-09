"""Tests that every installed Navix map loads through the full-JIT adapter."""

from __future__ import annotations

import jax
import pytest

from jarl.envs.navix.aliases import canonical_navix_env_ids, is_navix_map_installed
from jarl.envs.navix.full_jit import make_navix_full_jit_env

_INSTALLED_NAVIX_ENV_IDS = tuple(env_id for env_id in canonical_navix_env_ids() if is_navix_map_installed(env_id))


@pytest.mark.slow
@pytest.mark.parametrize(
    "env_id",
    [pytest.param(env_id, id=env_id) for env_id in _INSTALLED_NAVIX_ENV_IDS],
)
def test_installed_navix_map_loads_and_resets(env_id: str) -> None:
    env = make_navix_full_jit_env(env_id, max_episode_steps=10)
    key = jax.random.PRNGKey(0)
    state = env.reset(key[None], eval_mode=False)

    assert env.env_id == env_id
    assert state.next_observation.shape == (1, *env.get_processed_observation_shape())
    assert int(env.single_action_space.n) == 7
