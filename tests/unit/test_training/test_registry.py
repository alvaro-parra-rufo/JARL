"""Tests for the public trainer registry."""

from __future__ import annotations

import pytest

from jarl.agents.ppo.gru_trainer import ppo_gru_full_jax_trainer
from jarl.agents.ppo.trainer import ppo_full_jax_trainer
from jarl.training.registry import (
    import_trainer_callable,
    list_trainer_names,
    resolve_trainer_from_name,
)


class TestTrainerRegistry:
    """Tests for trainer name resolution."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            pytest.param("ppo.full_jax.navix", ppo_full_jax_trainer, id="ppo"),
            pytest.param("ppo_gru.full_jax.navix", ppo_gru_full_jax_trainer, id="ppo_gru"),
        ],
    )
    def test_resolve_known_trainer(self, name: str, expected: object) -> None:
        assert resolve_trainer_from_name(name) is expected

    def test_unknown_trainer_raises(self) -> None:
        with pytest.raises(ValueError, match=r"Unknown algorithm\.name"):
            resolve_trainer_from_name("unknown.agent")

    def test_list_trainer_names_matches_registry(self) -> None:
        assert list_trainer_names() == ("ppo.full_jax.navix", "ppo_gru.full_jax.navix")

    def test_import_trainer_callable(self) -> None:
        trainer = import_trainer_callable("tests.helpers.fake_trainer:fake_trainer")

        assert callable(trainer)
