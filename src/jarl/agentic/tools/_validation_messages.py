"""Human-readable validation errors for agent tool requests."""

from __future__ import annotations

from typing import get_args

from pydantic import BaseModel, ValidationError

__all__ = [
    "format_config_overrides_catalog",
    "format_request_validation_error",
]

_MAX_ERRORS_SHOWN = 3
_MIN_OVERRIDE_LOC_LEN = 2
_MAX_DESCRIPTION_LEN = 120

_CATALOG_HEADER = "Allowed config_overrides keys (match intent to descriptions; use exact names):"

_CONFIG_OVERRIDE_SCHEMA_HINT = (
    "Use only property names from this tool's config_overrides JSON schema "
    "(flat dotted keys, not nested objects or config_highlights names)."
)


def format_request_validation_error(
    request_cls: type[BaseModel],
    exc: ValidationError,
) -> str:
    """Return a concise validation message with a config_overrides schema catalog when useful."""
    errors = exc.errors()
    if not errors:
        return str(exc)

    forbidden_keys, other_parts = _classify_validation_errors(errors)
    sections: list[str] = []

    if forbidden_keys:
        keys_text = _format_key_list(forbidden_keys)
        overrides_cls = _resolve_config_overrides_model(request_cls)
        section = f"Invalid config_overrides keys: {keys_text}."
        if overrides_cls is not None:
            section = f"{section}\n{format_config_overrides_catalog(overrides_cls)}"
        else:
            section = f"{section} {_CONFIG_OVERRIDE_SCHEMA_HINT}"
        sections.append(section)

    if other_parts:
        other_text = "; ".join(other_parts[:_MAX_ERRORS_SHOWN])
        if len(other_parts) > _MAX_ERRORS_SHOWN:
            other_text = f"{other_text}; ({len(other_parts) - _MAX_ERRORS_SHOWN} more validation errors)"
        sections.append(other_text)

    if not sections:
        return str(exc)

    message = "\n".join(sections)
    if not forbidden_keys and "config_overrides" in request_cls.model_fields:
        return f"{message}\n{_CONFIG_OVERRIDE_SCHEMA_HINT}"
    return message


def format_config_overrides_catalog(overrides_cls: type[BaseModel]) -> str:
    """Return a compact key-and-description catalog from one overrides JSON schema."""
    properties = overrides_cls.model_json_schema().get("properties", {})
    if not isinstance(properties, dict):
        return _CATALOG_HEADER

    lines = [_CATALOG_HEADER]
    for key in sorted(properties):
        property_schema = properties[key]
        if not isinstance(property_schema, dict):
            lines.append(f"- {key}")
            continue
        description = property_schema.get("description")
        if isinstance(description, str) and description.strip():
            lines.append(f"- {key}: {_truncate(description.strip(), _MAX_DESCRIPTION_LEN)}")
        else:
            lines.append(f"- {key}")
    return "\n".join(lines)


def _classify_validation_errors(
    errors: list[dict[str, object]],
) -> tuple[list[str], list[str]]:
    """Split validation errors into forbidden override keys and other messages."""
    forbidden_keys: list[str] = []
    other_parts: list[str] = []

    for error in errors:
        if error.get("type") == "extra_forbidden":
            invalid_key = _invalid_override_key(error)
            if invalid_key:
                forbidden_keys.append(invalid_key)
                continue

        loc = ".".join(str(segment) for segment in error.get("loc", ()))
        msg = str(error.get("msg", "invalid value"))
        if loc:
            other_parts.append(f"{loc}: {msg}")
        else:
            other_parts.append(msg)

    return forbidden_keys, other_parts


def _resolve_config_overrides_model(request_cls: type[BaseModel]) -> type[BaseModel] | None:
    """Return the sparse overrides model attached to one tool request class."""
    field = request_cls.model_fields.get("config_overrides")
    if field is None or field.annotation is None:
        return None
    return _unwrap_model_type(field.annotation)


def _unwrap_model_type(annotation: object) -> type[BaseModel] | None:
    """Return the concrete Pydantic model from an optional overrides annotation."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    for arg in get_args(annotation):
        if isinstance(arg, type) and issubclass(arg, BaseModel):
            return arg
    return None


def _format_key_list(keys: list[str]) -> str:
    """Return a comma-separated key list with overflow summary."""
    shown = keys[:_MAX_ERRORS_SHOWN]
    keys_text = ", ".join(shown)
    extra = len(keys) - len(shown)
    if extra > 0:
        keys_text = f"{keys_text} (+{extra} more)"
    return keys_text


def _invalid_override_key(error: dict[str, object]) -> str | None:
    """Return the forbidden config_overrides property name from one Pydantic error."""
    loc = error.get("loc", ())
    if not isinstance(loc, tuple | list) or len(loc) < _MIN_OVERRIDE_LOC_LEN or loc[0] != "config_overrides":
        return None
    return str(loc[1])


def _truncate(text: str, max_length: int) -> str:
    """Return text truncated with an ellipsis when it exceeds ``max_length``."""
    if len(text) <= max_length:
        return text
    return f"{text[: max_length - 1].rstrip()}…"
