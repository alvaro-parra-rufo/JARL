"""Tests for run metadata collection."""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import platform
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from jarl.metadata import (
    EnvInfo,
    GitInfo,
    RunMetadata,
    collect_env_info,
    collect_git_info,
    now_iso,
    save_lockfile,
)

# ---- EnvInfo collection ---- #


class TestCollectEnvInfo:
    """Tests for environment info collection."""

    def test_captures_python_version(self) -> None:
        info = collect_env_info()

        assert info.python_version == platform.python_version()

    def test_captures_platform(self) -> None:
        info = collect_env_info()

        assert info.platform == platform.platform()

    def test_captures_hostname(self) -> None:
        info = collect_env_info()

        assert info.hostname == platform.node()

    @pytest.mark.parametrize(
        "pkg",
        [
            pytest.param("jax", id="jax"),
            pytest.param("jaxlib", id="jaxlib"),
            pytest.param("flax", id="flax"),
            pytest.param("optax", id="optax"),
            pytest.param("tensorboard", id="tensorboard"),
        ],
    )
    def test_installed_package_matches_actual_version(self, pkg: str) -> None:
        info = collect_env_info()

        expected = importlib_metadata.version(pkg)

        assert info.packages[pkg] == expected

    def test_missing_package_marked_not_installed(self, mocker: MockerFixture) -> None:
        mocker.patch("importlib.metadata.version", side_effect=importlib_metadata.PackageNotFoundError)

        info = collect_env_info()

        for pkg_version in info.packages.values():
            assert pkg_version == "not installed"


# ---- GitInfo collection ---- #


class TestCollectGitInfo:
    """Tests for git repository info collection."""

    def _mock_git(self, mocker: MockerFixture, responses: dict[tuple[str, ...], str]) -> None:
        """Set up mock git subprocess responses."""
        mocker.patch(
            "jarl.metadata.subprocess.check_output",
            side_effect=lambda cmd, **kw: responses.get(tuple(cmd), ""),  # noqa: ARG005
        )

    def test_captures_commit_and_branch(self, mocker: MockerFixture) -> None:
        self._mock_git(
            mocker,
            {
                ("git", "rev-parse", "HEAD"): "a" * 40,
                ("git", "rev-parse", "--abbrev-ref", "HEAD"): "main",
                ("git", "status", "--porcelain"): "",
                ("git", "config", "--get", "remote.origin.url"): "https://github.com/test/repo",
            },
        )

        info = collect_git_info()

        assert info.commit == "a" * 40
        assert info.branch == "main"
        assert info.dirty is False
        assert info.remote_url == "https://github.com/test/repo"

    def test_dirty_when_working_tree_has_changes(self, mocker: MockerFixture) -> None:
        self._mock_git(
            mocker,
            {
                ("git", "rev-parse", "HEAD"): "b" * 40,
                ("git", "rev-parse", "--abbrev-ref", "HEAD"): "feature",
                ("git", "status", "--porcelain"): "M file.py",
                ("git", "config", "--get", "remote.origin.url"): "",
            },
        )

        info = collect_git_info()

        assert info.dirty is True

    def test_outside_git_repo_returns_empty(self, mocker: MockerFixture) -> None:
        mocker.patch(
            "jarl.metadata.subprocess.check_output",
            side_effect=FileNotFoundError,
        )

        info = collect_git_info()

        assert info == GitInfo()


# ---- RunMetadata serialization ---- #


class TestRunMetadata:
    """Tests for metadata save/load round-trip."""

    def test_save_creates_json_file(self, tmp_path: Path) -> None:
        meta = RunMetadata(start_time=now_iso(), env=collect_env_info(), git=collect_git_info())

        path = meta.save(tmp_path / "meta.json")

        assert path.exists()
        assert path.suffix == ".json"

    def test_load_preserves_all_fields(self, tmp_path: Path) -> None:
        original = RunMetadata(
            start_time="2026-05-11T00:00:00+00:00",
            end_time="2026-05-11T01:00:00+00:00",
            status="completed",
            env=EnvInfo(
                python_version="3.12.0", platform="linux-x86_64", hostname="devbox", packages={"jax": "0.10.0"}
            ),
            git=GitInfo(commit="a" * 40, branch="main", dirty=False, remote_url="https://github.com/test/repo"),
        )

        original.save(tmp_path / "meta.json")
        loaded = RunMetadata.load(tmp_path / "meta.json")

        assert loaded.start_time == original.start_time
        assert loaded.end_time == original.end_time
        assert loaded.status == original.status
        assert loaded.env.python_version == "3.12.0"
        assert loaded.env.packages == {"jax": "0.10.0"}
        assert loaded.git.commit == "a" * 40
        assert loaded.git.remote_url == "https://github.com/test/repo"

    def test_to_dict_contains_all_keys(self) -> None:
        meta = RunMetadata(start_time="2026-01-01T00:00:00", status="running")

        result = meta.to_dict()

        assert result["start_time"] == "2026-01-01T00:00:00"
        assert result["status"] == "running"
        assert "env" in result
        assert "git" in result


# ---- Lockfile ---- #


class TestSaveLockfile:
    """Tests for lockfile copying."""

    def test_copies_lockfile_preserving_content(self, tmp_path: Path) -> None:
        src = tmp_path / "uv.lock"
        src.write_text("lockfile content\nline 2\n")
        dest_dir = tmp_path / "metadata"
        dest_dir.mkdir()

        result = save_lockfile(dest_dir, src)

        assert result is not None
        assert result.name == "uv.lock"
        assert result.read_text() == "lockfile content\nline 2\n"

    def test_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        dest_dir = tmp_path / "metadata"
        dest_dir.mkdir()

        result = save_lockfile(dest_dir, tmp_path / "nonexistent.lock")

        assert result is None


# ---- Timestamp ---- #


class TestNowIso:
    """Tests for ISO timestamp generation."""

    def test_returns_utc_iso_format(self) -> None:
        result = now_iso()

        assert "T" in result
        assert "+00:00" in result
