"""Tests for experiment-level case.json snapshots."""

from __future__ import annotations

from pathlib import Path

from jarl.envs.navix.custom.empty_variant import EMPTY_VARIANT_ENV_ID
from jarl.envs.navix.scenario_rewards import FLOOR_CELL_SCENARIO_ID, FLOOR_CELL_SCENARIO_VERSION
from jarl.experiments.cases.metadata import ExperimentCaseMetadata


class TestExperimentCaseMetadata:
    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "case.json"
        original = ExperimentCaseMetadata(
            case_id="navix_empty_variant",
            env_id=EMPTY_VARIANT_ENV_ID,
            scenario_reward_id=FLOOR_CELL_SCENARIO_ID,
            scenario_reward_version=FLOOR_CELL_SCENARIO_VERSION,
        )

        loaded = ExperimentCaseMetadata.load(original.save(path))

        assert loaded == original
