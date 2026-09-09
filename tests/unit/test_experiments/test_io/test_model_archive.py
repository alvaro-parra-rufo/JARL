"""Tests for model archive IO."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarl.experiments.io.model_archive import (
    MODEL_MANIFEST_FILENAME,
    canonical_model_filename,
    load_model_archive,
    save_model_archive,
)


class TestModelArchive:
    """Tests for jarl-native .model archives."""

    def test_round_trip_policy_and_critic(self, tmp_path: Path) -> None:
        components = {
            "policy": {"weights": [1.0, 2.0]},
            "critic": {"weights": [3.0]},
        }

        archive_path = save_model_archive(
            tmp_path,
            name="latest",
            step=5,
            components=components,
        )
        restored = load_model_archive(archive_path)

        assert archive_path.name == canonical_model_filename("latest", 5)
        assert restored["policy"] == components["policy"]
        assert restored["critic"] == components["critic"]

    def test_archive_contains_manifest(self, tmp_path: Path) -> None:
        import zipfile

        archive_path = save_model_archive(
            tmp_path,
            name="probe",
            step=1,
            components={"policy": {"x": 1}},
            algorithm_name="ppo.full_jax.navix",
            env_id="Navix-Empty-5x5-v0",
        )

        with zipfile.ZipFile(archive_path, "r") as archive:
            names = archive.namelist()
            manifest = json.loads(archive.read("manifest.json"))

        assert any(name.endswith(MODEL_MANIFEST_FILENAME) for name in names)
        assert manifest["algorithm_name"] == "ppo.full_jax.navix"
        assert manifest["env_id"] == "Navix-Empty-5x5-v0"

    def test_archive_does_not_include_nested_zip(self, tmp_path: Path) -> None:
        import zipfile

        archive_path = save_model_archive(
            tmp_path,
            name="weights",
            step=2,
            components={"policy": {"x": 1}, "critic": {"y": 2}},
        )

        with zipfile.ZipFile(archive_path, "r") as archive:
            names = set(archive.namelist())

        assert "archive.zip" not in names
        assert f"{MODEL_MANIFEST_FILENAME}" in names
        assert any(name.startswith("policy/") for name in names)
        assert any(name.startswith("critic/") for name in names)

    @pytest.mark.parametrize(
        ("name", "components"),
        [
            pytest.param("../../escape", {"policy": {"x": 1}}, id="unsafe_name"),
            pytest.param("weights", {"../escape": {"x": 1}}, id="unsafe_component"),
        ],
    )
    def test_save_rejects_unsafe_path_segments(
        self,
        tmp_path: Path,
        name: str,
        components: dict[str, object],
    ) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        with pytest.raises(ValueError, match="single path segment"):
            save_model_archive(models_dir, name=name, step=1, components=components)

        assert list(models_dir.glob("*.model")) == []
