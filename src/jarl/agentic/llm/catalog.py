"""LLM profile catalog: YAML config, assignment matching, and resolution."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import Field, model_validator

from jarl.agentic.llm.config import LLMSettings, load_llm_settings_from_env
from jarl.config import BaseConfig

__all__ = [
    "CUSTOM_LLM_YAML_FILENAME",
    "DEFAULT_LLM_PROFILES_DIRNAME",
    "DEFAULT_LLM_PROFILE_NAME",
    "DEFAULT_LLM_YAML_FILENAME",
    "ENV_LLM_CONFIG",
    "MAIN_COMPONENT_ID",
    "LLMCatalog",
    "ResolvedLLM",
    "catalog_from_settings",
    "effective_settings_key",
    "list_packaged_llm_catalog_paths",
    "load_llm_catalog",
    "load_llm_catalog_from_yaml",
    "packaged_llm_profiles_dir",
    "resolve_llm_catalog_ref",
    "resolve_llm_yaml_path",
    "write_llm_yaml_from_settings",
]

MAIN_COMPONENT_ID = "main"
"""Component id for the main experiment agent."""

DEFAULT_LLM_PROFILE_NAME = "default"
"""Profile name used when synthesizing a single-LLM catalog from env."""

DEFAULT_LLM_PROFILES_DIRNAME = "profiles"
"""Directory name under ``jarl.agentic.llm`` that stores LLM catalog YAML files."""

DEFAULT_LLM_YAML_FILENAME = "default.yaml"
"""Baseline Ollama catalog under ``jarl/agentic/llm/profiles`` (gitignored)."""

CUSTOM_LLM_YAML_FILENAME = "custom.yaml"
"""Optional multi-profile override under ``jarl/agentic/llm/profiles`` (gitignored)."""

ENV_LLM_CONFIG = "JARL_LLM_CONFIG"
"""Environment variable that overrides the LLM YAML path."""


@dataclass(frozen=True, slots=True)
class ResolvedLLM:
    """Outcome of resolving a component id against an ``LLMCatalog``."""

    component_id: str
    profile_name: str
    settings: LLMSettings


class LLMCatalog(BaseConfig):
    """Named LLM profiles with explicit component assignments and fallback."""

    profiles: Annotated[
        dict[str, LLMSettings],
        Field(description="Reusable named LLM profiles keyed by profile name."),
    ]
    assignments: Annotated[
        dict[str, str],
        Field(
            default_factory=dict,
            description=(
                "Component pattern to profile name. Exact ids or ``prefix/*`` "
                "patterns; longer prefixes win among patterns."
            ),
        ),
    ]
    fallback: Annotated[
        str,
        Field(
            description=(
                "Profile used when no assignment matches. Has no main/subagent "
                "semantics; it is only the configured last resort."
            ),
        ),
    ]

    @model_validator(mode="after")
    def _validate_profile_references(self) -> LLMCatalog:
        """Ensure fallback and assignments point at declared profiles."""
        if self.fallback not in self.profiles:
            msg = f"LLM catalog fallback {self.fallback!r} is not a declared profile."
            raise ValueError(msg)
        for pattern, profile_name in self.assignments.items():
            if profile_name not in self.profiles:
                msg = f"LLM catalog assignment {pattern!r} references unknown profile {profile_name!r}."
                raise ValueError(msg)
            _validate_assignment_pattern(pattern)
        return self

    def resolve(self, component_id: str) -> ResolvedLLM:
        """Resolve ``component_id`` to a profile via assignments then fallback.

        Precedence: exact assignment, then matching ``prefix/*`` with the longest
        prefix, then ``fallback``. Equal-length prefix ties raise ``ValueError``.
        """
        if not component_id:
            msg = "component_id must be a non-empty string."
            raise ValueError(msg)

        if component_id in self.assignments:
            profile_name = self.assignments[component_id]
            return ResolvedLLM(
                component_id=component_id,
                profile_name=profile_name,
                settings=self.profiles[profile_name],
            )

        matches = _matching_prefix_assignments(component_id, self.assignments)
        if matches:
            max_len = max(length for length, _pattern, _profile in matches)
            winners = [item for item in matches if item[0] == max_len]
            if len(winners) > 1:
                patterns = sorted(pattern for _length, pattern, _profile in winners)
                msg = f"Ambiguous LLM assignments for {component_id!r}: equal-length prefixes {patterns}."
                raise ValueError(msg)
            _length, _pattern, profile_name = winners[0]
            return ResolvedLLM(
                component_id=component_id,
                profile_name=profile_name,
                settings=self.profiles[profile_name],
            )

        return ResolvedLLM(
            component_id=component_id,
            profile_name=self.fallback,
            settings=self.profiles[self.fallback],
        )


def catalog_from_settings(
    settings: LLMSettings,
    *,
    profile_name: str = DEFAULT_LLM_PROFILE_NAME,
) -> LLMCatalog:
    """Build a single-profile catalog equivalent to one ``LLMSettings``."""
    return LLMCatalog(
        profiles={profile_name: settings},
        assignments={},
        fallback=profile_name,
    )


def load_llm_catalog(*, path: str | Path | None = None) -> LLMCatalog:
    """Load an ``LLMCatalog`` from YAML or synthesize one from env settings.

    Resolution order for the YAML path: explicit ``path``, ``JARL_LLM_CONFIG``,
    ``jarl/agentic/llm/profiles/custom.yaml``, then
    ``jarl/agentic/llm/profiles/default.yaml``.
    Documentation templates ``*.example.yaml`` under that directory are never
    loaded. Missing YAML falls back to
    ``catalog_from_settings(load_llm_settings_from_env())``.
    """
    yaml_path = resolve_llm_yaml_path(path)
    if yaml_path is not None:
        return load_llm_catalog_from_yaml(yaml_path)
    return catalog_from_settings(load_llm_settings_from_env())


def load_llm_catalog_from_yaml(path: str | Path) -> LLMCatalog:
    """Parse and validate an LLM catalog YAML file."""
    resolved = Path(path).resolve()
    if not resolved.is_file():
        msg = f"LLM catalog YAML not found: {resolved}"
        raise FileNotFoundError(msg)

    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        msg = f"LLM catalog YAML must be a mapping: {resolved}"
        raise ValueError(msg)

    defaults = payload.get("defaults") or {}
    if defaults is None:
        defaults = {}
    if not isinstance(defaults, dict):
        msg = f"LLM catalog 'defaults' must be a mapping: {resolved}"
        raise ValueError(msg)

    raw_profiles = payload.get("profiles")
    if not isinstance(raw_profiles, dict) or not raw_profiles:
        msg = f"LLM catalog YAML requires a non-empty 'profiles' mapping: {resolved}"
        raise ValueError(msg)

    profiles: dict[str, LLMSettings] = {}
    for name, raw_profile in raw_profiles.items():
        if not isinstance(raw_profile, dict):
            msg = f"LLM profile {name!r} must be a mapping in {resolved}."
            raise ValueError(msg)
        merged = {**defaults, **raw_profile}
        profiles[str(name)] = LLMSettings.model_validate(merged)

    raw_assignments = payload.get("assignments") or {}
    if raw_assignments is None:
        raw_assignments = {}
    if not isinstance(raw_assignments, dict):
        msg = f"LLM catalog 'assignments' must be a mapping: {resolved}"
        raise ValueError(msg)
    assignments = {str(key): str(value) for key, value in raw_assignments.items()}

    fallback = payload.get("fallback")
    if fallback is None:
        msg = f"LLM catalog YAML requires 'fallback': {resolved}"
        raise ValueError(msg)

    return LLMCatalog(
        profiles=profiles,
        assignments=assignments,
        fallback=str(fallback),
    )


def packaged_llm_profiles_dir() -> Path:
    """Return the ``jarl/agentic/llm/profiles`` directory shipped with the package."""
    return Path(__file__).resolve().parent / DEFAULT_LLM_PROFILES_DIRNAME


def list_packaged_llm_catalog_paths() -> tuple[Path, ...]:
    """Return existing packaged catalog YAMLs in load-precedence order.

    Only ``custom.yaml`` and ``default.yaml`` are considered. Documentation
    templates ``*.example.yaml`` are never listed.
    """
    profiles_dir = packaged_llm_profiles_dir()
    found: list[Path] = []
    for filename in (CUSTOM_LLM_YAML_FILENAME, DEFAULT_LLM_YAML_FILENAME):
        candidate = profiles_dir / filename
        if candidate.is_file():
            found.append(candidate.resolve())
    return tuple(found)


def resolve_llm_catalog_ref(ref: str | Path) -> Path:
    """Resolve a catalog path or packaged stem (``custom`` / ``default``)."""
    raw = Path(ref).expanduser()
    if raw.is_file():
        return raw.resolve()

    profiles_dir = packaged_llm_profiles_dir()
    candidates = [
        profiles_dir / raw.name,
        profiles_dir / f"{raw.name}.yaml",
        profiles_dir / f"{raw.stem}.yaml",
    ]
    if not raw.is_absolute():
        candidates.insert(0, (Path.cwd() / raw).resolve())
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    available = ", ".join(path.name for path in list_packaged_llm_catalog_paths()) or "(none)"
    msg = f"LLM catalog not found for {ref!r}. Available packaged catalogs: {available}."
    raise FileNotFoundError(msg)


def resolve_llm_yaml_path(path: str | Path | None = None) -> Path | None:
    """Return the LLM YAML path to load, or ``None`` when absent."""
    if path is not None:
        return Path(path).resolve()

    env_path = os.environ.get(ENV_LLM_CONFIG)
    if env_path:
        return Path(env_path).expanduser().resolve()

    profiles_dir = packaged_llm_profiles_dir()
    for filename in (CUSTOM_LLM_YAML_FILENAME, DEFAULT_LLM_YAML_FILENAME):
        candidate = profiles_dir / filename
        if candidate.is_file():
            return candidate.resolve()
    return None


def write_llm_yaml_from_settings(
    path: str | Path,
    settings: LLMSettings,
    *,
    profile_name: str = DEFAULT_LLM_PROFILE_NAME,
) -> Path:
    """Write a single-profile catalog YAML equivalent to ``settings``.

    Intended for generating a local catalog YAML from ``JARL_LLM_*`` / ``.env``.
    Does not write API keys when the value came only from ambient env defaults
    beyond what is present on ``settings``.
    """
    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    profile_payload: dict[str, Any] = {
        "provider": settings.provider,
        "model": settings.model,
    }
    if settings.base_url is not None:
        profile_payload["base_url"] = settings.base_url
    if settings.api_key is not None:
        profile_payload["api_key"] = settings.api_key
    if settings.temperature is not None:
        profile_payload["temperature"] = settings.temperature

    document = {
        "profiles": {profile_name: profile_payload},
        "assignments": {},
        "fallback": profile_name,
    }
    resolved.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return resolved


def effective_settings_key(settings: LLMSettings) -> tuple[object, ...]:
    """Return a hashable identity for caching clients by effective settings."""
    return (
        settings.provider,
        settings.model,
        settings.base_url,
        settings.api_key,
        settings.temperature,
    )


def _validate_assignment_pattern(pattern: str) -> None:
    """Reject unsupported assignment pattern shapes."""
    if not pattern:
        msg = "LLM assignment pattern must be non-empty."
        raise ValueError(msg)
    if pattern.endswith("/*"):
        prefix = pattern[: -len("/*")]
        if not prefix or "*" in prefix:
            msg = f"Unsupported LLM assignment pattern: {pattern!r}."
            raise ValueError(msg)
        return
    if "*" in pattern:
        msg = (
            f"Unsupported LLM assignment pattern: {pattern!r}. "
            "Use an exact component id or a single ``prefix/*`` pattern."
        )
        raise ValueError(msg)


def _matching_prefix_assignments(
    component_id: str,
    assignments: Mapping[str, str],
) -> list[tuple[int, str, str]]:
    """Return ``(prefix_length, pattern, profile)`` for matching ``prefix/*`` rules."""
    matches: list[tuple[int, str, str]] = []
    for pattern, profile_name in assignments.items():
        if not pattern.endswith("/*"):
            continue
        prefix = pattern[: -len("/*")]
        if component_id == prefix or component_id.startswith(f"{prefix}/"):
            matches.append((len(prefix), pattern, profile_name))
    return matches
