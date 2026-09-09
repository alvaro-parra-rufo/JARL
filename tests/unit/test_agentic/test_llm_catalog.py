"""Unit tests for the multi-profile LLM catalog."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarl.agentic.llm.catalog import (
    CUSTOM_LLM_YAML_FILENAME,
    DEFAULT_LLM_PROFILE_NAME,
    DEFAULT_LLM_YAML_FILENAME,
    LLMCatalog,
    catalog_from_settings,
    load_llm_catalog,
    load_llm_catalog_from_yaml,
    resolve_llm_yaml_path,
    write_llm_yaml_from_settings,
)
from jarl.agentic.llm.config import LLMSettings


def _settings(model: str) -> LLMSettings:
    return LLMSettings(provider="ollama", model=model)


def _catalog() -> LLMCatalog:
    return LLMCatalog(
        profiles={
            "powerful": _settings("qwen-powerful"),
            "cheap": _settings("qwen-cheap"),
            "metrics_analysis": _settings("qwen-metrics"),
        },
        assignments={
            "main": "powerful",
            "subagents/*": "cheap",
            "subagents/metrics_analysis": "metrics_analysis",
        },
        fallback="cheap",
    )


class TestLLMCatalogResolve:
    def test_exact_assignment_wins_over_prefix(self) -> None:
        resolved = _catalog().resolve("subagents/metrics_analysis")

        assert resolved.profile_name == "metrics_analysis"
        assert resolved.settings.model == "qwen-metrics"

    def test_prefix_assignment_applies_to_other_subagents(self) -> None:
        resolved = _catalog().resolve("subagents/other")

        assert resolved.profile_name == "cheap"
        assert resolved.settings.model == "qwen-cheap"

    def test_main_assignment(self) -> None:
        resolved = _catalog().resolve("main")

        assert resolved.profile_name == "powerful"

    def test_fallback_when_no_assignment_matches(self) -> None:
        resolved = _catalog().resolve("totally_unassigned")

        assert resolved.profile_name == "cheap"

    def test_longer_prefix_wins(self) -> None:
        catalog = LLMCatalog(
            profiles={"near": _settings("near"), "far": _settings("far")},
            assignments={
                "subagents/*": "far",
                "subagents/metrics/*": "near",
            },
            fallback="far",
        )

        resolved = catalog.resolve("subagents/metrics/extra")

        assert resolved.profile_name == "near"

    def test_equal_prefix_length_is_ambiguous(self) -> None:
        catalog = LLMCatalog(
            profiles={"a": _settings("a"), "b": _settings("b")},
            assignments={
                "ab/*": "a",
                "ac/*": "b",
            },
            fallback="a",
        )
        # Force an artificial ambiguity by resolving a synthetic id that both
        # prefixes could match only if matching were looser; with path rules
        # equal-length distinct prefixes cannot both match. Exercise the error
        # path by monkeypatching the matcher through duplicate max lengths via
        # a crafted component that equals one prefix while also matching another
        # is impossible — instead call resolve after injecting two equal winners.
        from jarl.agentic.llm import catalog as catalog_mod

        original = catalog_mod._matching_prefix_assignments

        def _fake_matches(component_id: str, assignments: object) -> list[tuple[int, str, str]]:
            del component_id, assignments
            return [(2, "ab/*", "a"), (2, "ac/*", "b")]

        catalog_mod._matching_prefix_assignments = _fake_matches  # type: ignore[assignment]
        try:
            with pytest.raises(ValueError, match="Ambiguous LLM assignments"):
                catalog.resolve("ab/x")
        finally:
            catalog_mod._matching_prefix_assignments = original  # type: ignore[assignment]

    def test_unknown_fallback_rejected(self) -> None:
        with pytest.raises(ValueError, match="fallback"):
            LLMCatalog(
                profiles={"cheap": _settings("qwen-cheap")},
                assignments={},
                fallback="missing",
            )

    def test_unknown_assignment_profile_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown profile"):
            LLMCatalog(
                profiles={"cheap": _settings("qwen-cheap")},
                assignments={"main": "powerful"},
                fallback="cheap",
            )


class TestLLMCatalogYamlAndCompat:
    def test_load_from_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "llm.yaml"
        path.write_text(
            "\n".join(
                [
                    "defaults:",
                    "  provider: ollama",
                    "  base_url: http://localhost:11434",
                    "profiles:",
                    "  powerful:",
                    "    model: big",
                    "  cheap:",
                    "    model: small",
                    "assignments:",
                    "  main: powerful",
                    '  "subagents/*": cheap',
                    "fallback: cheap",
                ]
            ),
            encoding="utf-8",
        )

        catalog = load_llm_catalog_from_yaml(path)

        assert catalog.resolve("main").settings.model == "big"
        assert catalog.resolve("subagents/x").settings.model == "small"

    def test_doc_openai_compatible_example_parses(self) -> None:
        from jarl.agentic.llm.catalog import packaged_llm_profiles_dir

        path = packaged_llm_profiles_dir() / "catalog.openai_compatible.example.yaml"
        assert path.is_file()

        catalog = load_llm_catalog_from_yaml(path)

        resolved = catalog.resolve("main")
        assert resolved.profile_name == "openai"
        assert resolved.settings.provider == "openai_compatible"
        assert resolved.settings.base_url == "https://api.openai.com/v1"
        assert resolved.settings.model == "gpt-4o"
        assert resolved.settings.api_key == "sk-REPLACE_ME"
        assert catalog.profiles["openai"].api_key == "sk-REPLACE_ME"

    def test_example_templates_are_not_auto_loaded(self) -> None:
        from jarl.agentic.llm.catalog import packaged_llm_profiles_dir

        profiles_dir = packaged_llm_profiles_dir()
        resolved = resolve_llm_yaml_path()
        if resolved is None:
            return
        assert resolved.name in {CUSTOM_LLM_YAML_FILENAME, DEFAULT_LLM_YAML_FILENAME}
        assert resolved.parent == profiles_dir.resolve()
        assert not resolved.name.endswith(".example.yaml")

    def test_loads_packaged_custom_catalog_when_present(self) -> None:
        from jarl.agentic.llm.catalog import packaged_llm_profiles_dir

        profiles_dir = packaged_llm_profiles_dir()
        custom_path = profiles_dir / CUSTOM_LLM_YAML_FILENAME
        if not custom_path.is_file():
            pytest.skip("custom.yaml not present in local profiles dir")

        catalog = load_llm_catalog()

        assert resolve_llm_yaml_path() == custom_path.resolve()
        assert catalog.resolve("main").profile_name == "deepseek"
        assert catalog.resolve("subagents/other").profile_name == "cheap"
        assert catalog.resolve("subagents/metrics_analysis").profile_name == "metrics_analysis"
        assert catalog.profiles["deepseek"].provider == "openai_compatible"

    def test_loads_packaged_default_catalog_when_custom_absent(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        profiles = tmp_path / "profiles"
        profiles.mkdir(parents=True)
        (profiles / DEFAULT_LLM_YAML_FILENAME).write_text(
            "\n".join(
                [
                    "defaults:",
                    "  provider: ollama",
                    "  base_url: http://127.0.0.1:11434",
                    "profiles:",
                    "  powerful:",
                    "    model: qwen-default",
                    "  cheap:",
                    "    model: qwen-cheap",
                    "assignments:",
                    "  main: powerful",
                    '  "subagents/*": cheap',
                    "fallback: cheap",
                ]
            ),
            encoding="utf-8",
        )
        monkeypatch.delenv("JARL_LLM_CONFIG", raising=False)
        monkeypatch.setattr(
            "jarl.agentic.llm.catalog.packaged_llm_profiles_dir",
            lambda: profiles,
        )

        catalog = load_llm_catalog()

        assert resolve_llm_yaml_path() == (profiles / DEFAULT_LLM_YAML_FILENAME).resolve()
        assert catalog.resolve("main").profile_name == "powerful"
        assert catalog.resolve("main").settings.model == "qwen-default"
        assert "deepseek" not in catalog.profiles

    def test_custom_yaml_overrides_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        profiles = tmp_path / "profiles"
        profiles.mkdir(parents=True)
        (profiles / "default.yaml").write_text(
            "\n".join(
                [
                    "profiles:",
                    "  cheap:",
                    "    provider: ollama",
                    "    model: from-default",
                    "fallback: cheap",
                ]
            ),
            encoding="utf-8",
        )
        (profiles / "custom.yaml").write_text(
            "\n".join(
                [
                    "profiles:",
                    "  cheap:",
                    "    provider: ollama",
                    "    model: from-custom",
                    "fallback: cheap",
                ]
            ),
            encoding="utf-8",
        )
        monkeypatch.delenv("JARL_LLM_CONFIG", raising=False)
        monkeypatch.setattr(
            "jarl.agentic.llm.catalog.packaged_llm_profiles_dir",
            lambda: profiles,
        )

        catalog = load_llm_catalog()

        assert catalog.resolve("main").settings.model == "from-custom"

    def test_catalog_from_settings_and_env_load(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.delenv("JARL_LLM_CONFIG", raising=False)
        monkeypatch.setenv("JARL_LLM_PROVIDER", "ollama")
        monkeypatch.setenv("JARL_LLM_MODEL", "from-env")
        monkeypatch.setattr(
            "jarl.agentic.llm.catalog.packaged_llm_profiles_dir",
            lambda: tmp_path / "missing-profiles",
        )

        catalog = load_llm_catalog()

        assert catalog.fallback == DEFAULT_LLM_PROFILE_NAME
        assert catalog.resolve("main").settings.model == "from-env"
        assert catalog.resolve("subagents/metrics_analysis").settings.model == "from-env"

    def test_write_yaml_from_settings(self, tmp_path: Path) -> None:
        path = write_llm_yaml_from_settings(tmp_path / "llm.yaml", _settings("written"))

        catalog = load_llm_catalog_from_yaml(path)
        assert catalog.resolve("anything").profile_name == DEFAULT_LLM_PROFILE_NAME
        assert catalog.resolve("anything").settings.model == "written"

    def test_catalog_from_settings_helper(self) -> None:
        catalog = catalog_from_settings(_settings("one"))

        assert catalog.resolve("main").settings.model == "one"
