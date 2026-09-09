"""LLM provider factory and multi-profile catalog for the agentic workflow."""

from __future__ import annotations

from jarl.agentic.llm.catalog import (
    CUSTOM_LLM_YAML_FILENAME,
    DEFAULT_LLM_PROFILE_NAME,
    DEFAULT_LLM_PROFILES_DIRNAME,
    DEFAULT_LLM_YAML_FILENAME,
    ENV_LLM_CONFIG,
    MAIN_COMPONENT_ID,
    LLMCatalog,
    ResolvedLLM,
    catalog_from_settings,
    effective_settings_key,
    list_packaged_llm_catalog_paths,
    load_llm_catalog,
    load_llm_catalog_from_yaml,
    packaged_llm_profiles_dir,
    resolve_llm_catalog_ref,
    resolve_llm_yaml_path,
    write_llm_yaml_from_settings,
)
from jarl.agentic.llm.config import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    LLMProvider,
    LLMSettings,
    load_llm_settings_from_env,
)
from jarl.agentic.llm.factory import create_chat_model

__all__ = [
    "CUSTOM_LLM_YAML_FILENAME",
    "DEFAULT_LLM_PROFILES_DIRNAME",
    "DEFAULT_LLM_PROFILE_NAME",
    "DEFAULT_LLM_YAML_FILENAME",
    "DEFAULT_OLLAMA_BASE_URL",
    "DEFAULT_OLLAMA_MODEL",
    "ENV_LLM_CONFIG",
    "MAIN_COMPONENT_ID",
    "LLMCatalog",
    "LLMProvider",
    "LLMSettings",
    "ResolvedLLM",
    "catalog_from_settings",
    "create_chat_model",
    "effective_settings_key",
    "list_packaged_llm_catalog_paths",
    "load_llm_catalog",
    "load_llm_catalog_from_yaml",
    "load_llm_settings_from_env",
    "packaged_llm_profiles_dir",
    "resolve_llm_catalog_ref",
    "resolve_llm_yaml_path",
    "write_llm_yaml_from_settings",
]
