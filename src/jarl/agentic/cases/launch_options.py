"""Declarative launch options that force tool arguments during case execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "CaseLaunchOption",
    "parse_launch_option_value",
    "resolve_tool_overrides",
]

LaunchOptionType = Literal["bool", "int", "str"]
"""Supported widget/value types for case launch options."""


@dataclass(frozen=True, slots=True)
class CaseLaunchOption:
    """One user-facing option that forces a tool field during a case run.

    Args:
        key: Stable option id used in CLI, UI, and persisted launch requests.
        label: Short human-readable label for Runner Lab and docs.
        tool: Tool name whose handler receives the forced value.
        field: Request field name merged before Pydantic validation.
        option_type: Value coercion applied to CLI/UI inputs.
        default: Default value when the option is omitted.
        description: Optional helper text for Runner Lab.
    """

    key: str
    label: str
    tool: str
    field: str
    option_type: LaunchOptionType = "bool"
    default: bool | int | str = False
    description: str = ""
    choices: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate option metadata."""
        if not self.key.strip():
            raise ValueError("Case launch option key must not be empty.")
        if not self.label.strip():
            raise ValueError("Case launch option label must not be empty.")
        if not self.tool.strip() or not self.field.strip():
            raise ValueError("Case launch option tool and field must not be empty.")
        if self.option_type == "bool" and not isinstance(self.default, bool):
            msg = f"Bool launch option {self.key!r} requires a bool default."
            raise TypeError(msg)
        if self.option_type == "int" and not isinstance(self.default, int):
            msg = f"Int launch option {self.key!r} requires an int default."
            raise TypeError(msg)
        if self.option_type == "str" and not isinstance(self.default, str):
            msg = f"Str launch option {self.key!r} requires a str default."
            raise TypeError(msg)
        if self.choices and self.default not in self.choices:
            msg = f"Launch option {self.key!r} default must be one of {self.choices!r}."
            raise ValueError(msg)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly description for catalogs and Runner Lab."""
        payload: dict[str, object] = {
            "key": self.key,
            "label": self.label,
            "tool": self.tool,
            "field": self.field,
            "type": self.option_type,
            "default": self.default,
            "description": self.description,
        }
        if self.choices:
            payload["choices"] = list(self.choices)
        return payload


def parse_launch_option_value(
    option: CaseLaunchOption,
    raw: object,
) -> bool | int | str:
    """Coerce one launch-option value to the declared option type."""
    if option.option_type == "bool":
        return _parse_bool_option(option, raw)
    if option.option_type == "int":
        return _parse_int_option(option, raw)
    if isinstance(raw, str):
        return raw
    return str(raw)


def _parse_bool_option(option: CaseLaunchOption, raw: object) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    if isinstance(raw, int) and raw in {0, 1}:
        return bool(raw)
    msg = f"Launch option {option.key!r} expects a boolean value, got {raw!r}."
    raise ValueError(msg)


def _parse_int_option(option: CaseLaunchOption, raw: object) -> int:
    if isinstance(raw, bool):
        msg = f"Launch option {option.key!r} expects an integer value, got bool."
        raise ValueError(msg)
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str) and raw.strip().isdigit():
        return int(raw.strip())
    msg = f"Launch option {option.key!r} expects an integer value, got {raw!r}."
    raise ValueError(msg)


def resolve_tool_overrides(
    options: tuple[CaseLaunchOption, ...],
    values: Mapping[str, object] | None,
) -> dict[str, dict[str, object]]:
    """Map declared launch options to per-tool forced request fields."""
    if not options:
        return {}
    resolved_values = dict(values or {})
    overrides: dict[str, dict[str, object]] = {}
    for option in options:
        raw = resolved_values.get(option.key, option.default)
        coerced = parse_launch_option_value(option, raw)
        tool_fields = overrides.setdefault(option.tool, {})
        tool_fields[option.field] = coerced
    return overrides
