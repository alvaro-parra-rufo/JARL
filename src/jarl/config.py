"""Base configuration model for jarl experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict

from jarl.utils import deep_merge_dicts, flatten_dict, unflatten_dict

__all__ = [
    "BaseConfig",
    "ConfigDiff",
    "collect_diff_paths",
]


@dataclass
class ConfigDiff:
    """Result of comparing a baseline config to a target config.

    Always compute as `parent.diff(child)` where `self` is the baseline (parent)
    and `other` is the target (child).

    Args:
        added: Fields present in the target but not the baseline (target values).
        removed: Fields present in the baseline but not the target (baseline values).
        changed: Fields with different values as `(baseline_value, target_value)`.
    """

    added: dict[str, Any]
    removed: dict[str, Any]
    changed: dict[str, tuple[Any, Any]]

    def __bool__(self) -> bool:
        """Return whether there are any changes."""
        return bool(self.added) or bool(self.removed) or bool(self.changed)

    def __contains__(self, item: object) -> bool:
        """Return whether a field is present in this diff."""
        return isinstance(item, str) and (item in self.added or item in self.removed or item in self.changed)

    def to_overrides(
        self,
        side: Literal["baseline", "target", "parent", "child"] = "target",
    ) -> dict[str, Any]:
        """Extract sparse override values suitable for `apply_overrides`.

        Args:
            side: Which side of `changed` tuples to use. `target` and `child`
                return target values; `baseline` and `parent` return baseline
                values. Defaults to `target` for diffs computed as
                `parent.diff(child)`.

        Returns:
            Override dictionary for transforming the baseline into the target.
        """
        index = 0 if side in ("baseline", "parent") else 1
        return {**self.added, **{key: value[index] for key, value in self.changed.items()}}


def collect_diff_paths(diff: ConfigDiff, *, sep: str = ".") -> frozenset[str]:
    """Return dotted config paths touched by a ``ConfigDiff``.

    Args:
        diff: Config delta to inspect.
        sep: Separator used for nested path segments.

    Returns:
        Flattened paths present in ``added``, ``removed``, or ``changed``.
    """
    paths: set[str] = set()
    paths.update(diff.added)
    paths.update(diff.removed)
    for key, (old, new) in diff.changed.items():
        if isinstance(old, dict) and isinstance(new, dict):
            paths.update(_dict_diff_paths(old, new, prefix=f"{key}{sep}", sep=sep))
        else:
            paths.add(key)
    return frozenset(paths)


def _dict_diff_paths(old: dict[str, Any], new: dict[str, Any], *, prefix: str, sep: str) -> set[str]:
    paths: set[str] = set()
    for key in set(old) | set(new):
        full_key = f"{prefix}{key}"
        if key not in old or key not in new:
            paths.add(full_key)
            continue
        old_value = old[key]
        new_value = new[key]
        if old_value == new_value:
            continue
        if isinstance(old_value, dict) and isinstance(new_value, dict):
            paths.update(_dict_diff_paths(old_value, new_value, prefix=f"{full_key}{sep}", sep=sep))
        else:
            paths.add(full_key)
    return paths


class BaseConfig(BaseModel):
    """Base configuration for all jarl experiments.

    Provides config diffing, override application, and serialization.
    Experiment-specific configs should subclass this and add their fields.

    Example:
        ```python
        class MNISTConfig(BaseConfig):
            learning_rate: float = 0.1
            batch_size: int = 128

        parent = MNISTConfig(learning_rate=0.1)
        child = MNISTConfig(learning_rate=0.01)
        parent.diff(child)  # ConfigDiff(changed={"learning_rate": (0.1, 0.01)})
        ```
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    IMMUTABLE_PATHS: ClassVar[frozenset[str]] = frozenset()
    """Dotted paths that must not change after the experiment root is created."""

    @classmethod
    def immutable_paths(cls, *, prefix: str = "") -> frozenset[str]:
        """Collect immutable dotted paths declared on this model and nested blocks.

        Args:
            prefix: Optional dotted prefix prepended to local paths and nested fields.

        Returns:
            Immutable paths for ``cls`` composed with nested ``BaseConfig`` fields.
        """
        paths: set[str] = set()
        for base in cls.__mro__:
            if base is object or base is BaseModel:
                continue
            if not issubclass(base, BaseConfig):
                continue
            for path in base.IMMUTABLE_PATHS:
                paths.add(f"{prefix}{path}" if prefix else path)
        for field_name, field_info in cls.model_fields.items():
            field_type = field_info.annotation
            if isinstance(field_type, type) and issubclass(field_type, BaseConfig):
                nested_prefix = f"{prefix}{field_name}." if prefix else f"{field_name}."
                paths.update(field_type.immutable_paths(prefix=nested_prefix))
        return frozenset(paths)

    def validate_diff_allowed(self, diff: ConfigDiff, *, sep: str = ".") -> None:
        """Reject a diff that touches immutable config paths.

        Args:
            diff: Config delta to validate against ``immutable_paths()``.
            sep: Separator used when flattening nested dict changes.

        Raises:
            ValueError: If any touched path is listed in ``IMMUTABLE_PATHS``.
        """
        touched = collect_diff_paths(diff, sep=sep)
        immutable = type(self).immutable_paths()
        rejected = sorted(touched & immutable)
        if rejected:
            msg = f"Config diff touches immutable paths: {', '.join(rejected)}"
            raise ValueError(msg)

    # ---- Config diffing ---- #

    def diff(
        self,
        other: BaseConfig,
        allow_extra: bool = False,
        allow_missing: bool = False,
        flatten: bool = False,
        sep: str = ".",
        mode: str = "json",
    ) -> ConfigDiff:
        """Compute the delta from this config (baseline) to `other` (target).

        Args:
            other: Target config to compare against (typically the child).
            allow_extra: Whether to allow extra fields in the target config.
            allow_missing: Whether to allow fields missing from the target config.
            flatten: Whether to flatten nested configs before comparing.
            sep: Separator for flattened keys.
            mode: Mode for model dump.

        Returns:
            `ConfigDiff` describing changes from baseline to target.

        Notes:
            Defaults compare nested configs as dicts, not objects. Use
            `parent.diff(child, flatten=True)` for dotted nested keys.
        """
        self_data = self.model_dump(mode=mode)
        other_data = other.model_dump(mode=mode)
        if flatten:
            self_data = flatten_dict(self_data, sep=sep)
            other_data = flatten_dict(other_data, sep=sep)
        extra_fields = set(other_data) - set(self_data)
        missing_fields = set(self_data) - set(other_data)
        if not allow_extra and extra_fields:
            raise ValueError(f"Extra fields in other config: {', '.join(sorted(extra_fields))}")
        if not allow_missing and missing_fields:
            raise ValueError(f"Missing fields in self config: {', '.join(sorted(missing_fields))}")
        return ConfigDiff(
            added={k: v for k, v in other_data.items() if k not in self_data},
            removed={k: v for k, v in self_data.items() if k not in other_data},
            changed={
                k: (self_data[k], other_data[k]) for k in self_data if k in other_data and self_data[k] != other_data[k]
            },
        )

    def apply_overrides(
        self,
        overrides: dict[str, Any],
        deep: bool = False,
        nested: bool = True,
        sep: str = ".",
    ) -> Self:
        """Create a new config with the given overrides applied.

        Args:
            overrides: Field names and values to override. When `nested` is
                `True`, dotted keys (e.g. `jax.default_matmul_precision`) merge
                into nested sub-configs.
            deep: Whether to return a deep copy of the config instead of
                a shallow copy (non-nested path only).
            nested: Whether to support dotted keys and deep-merge nested fields.
            sep: Separator for dotted override keys.

        Returns:
            New config instance with overrides merged in.

        Raises:
            ValueError: If any override key is not a valid field name (when
                `nested=False`).
        """
        if not overrides:
            return self.model_copy(deep=deep)

        if nested:
            expanded = unflatten_dict(overrides, sep=sep)
            merged = deep_merge_dicts(self.model_dump(), expanded)
            return type(self).model_validate(merged)

        valid_fields = set(type(self).model_fields)
        unknown = set(overrides) - valid_fields
        if unknown:
            raise ValueError(f"Unknown config fields: {', '.join(sorted(unknown))}")
        return self.model_copy(update=overrides, deep=deep)

    def resolve_config_overrides(
        self,
        *,
        target: BaseConfig | None = None,
        diff: ConfigDiff | None = None,
        flatten: bool = True,
    ) -> dict[str, Any]:
        """Compute sparse overrides that transform this config into `target`.

        Args:
            target: Target config; overrides are autocomputed via
                `self.diff(target)`.
            diff: Explicit diff precomputed as `self.diff(target)`.
            flatten: Whether to flatten nested configs when computing from
                `target`.

        Returns:
            Override dictionary suitable for `self.apply_overrides()`.

        Raises:
            ValueError: If both `target` and `diff` are provided.
        """
        if target is not None and diff is not None:
            msg = "Provide only one of target or diff."
            raise ValueError(msg)
        if target is not None:
            return self.diff(target, flatten=flatten).to_overrides()
        if diff is not None:
            return diff.to_overrides()
        return {}

    # ---- Serialization ---- #

    def save(self, path: str | Path) -> Path:
        """Serialize this config to a JSON file.

        Args:
            path: Destination file path.

        Returns:
            The path written to.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_dump(), indent=2, default=str))
        return path

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Deserialize a config from a JSON file.

        Args:
            path: Source file path.

        Returns:
            Reconstructed config instance.
        """
        data = json.loads(Path(path).read_text())
        return cls.model_validate(data)
