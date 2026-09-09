"""Typed sparse configuration override models."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Annotated, cast

from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic.fields import FieldInfo

from jarl.config import BaseConfig
from jarl.training.config import RLRunConfig

__all__ = [
    "ForkConfigOverrides",
    "RetrainConfigOverrides",
    "SparseConfigOverrides",
    "build_sparse_config_overrides_model",
]


class SparseConfigOverrides(BaseModel):
    """Base model for typed dotted-path configuration overrides."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_by_alias=True,
        validate_by_name=False,
    )

    def to_dotted_dict(self) -> dict[str, object]:
        """Return only explicitly supplied values keyed by dotted config path."""
        payload = self.model_dump(
            by_alias=True,
            exclude_unset=True,
            mode="python",
        )
        return cast("dict[str, object]", payload)


def build_sparse_config_overrides_model(
    model_name: str,
    *,
    config_cls: type[BaseConfig],
    allowed_paths: frozenset[str],
) -> type[SparseConfigOverrides]:
    """Build a typed sparse model from config fields selected by dotted paths.

    Args:
        model_name: Public name assigned to the generated model.
        config_cls: Configuration model that owns field types and constraints.
        allowed_paths: Dotted leaf paths exposed by the sparse model.

    Returns:
        Generated Pydantic model whose JSON properties use dotted aliases.

    Raises:
        TypeError: If a path traverses a non-model field.
        ValueError: If a path is empty, unknown, or collides after normalization.
    """
    field_definitions: dict[str, object] = {}
    for path in sorted(allowed_paths):
        internal_name = _internal_field_name(path)
        if internal_name in field_definitions:
            msg = f"Config override paths collide as {internal_name!r}."
            raise ValueError(msg)
        annotation, field_info, default = _resolve_config_leaf(config_cls, path)
        annotated_type = _annotated_field_type(annotation, field_info)
        field_definitions[internal_name] = (
            annotated_type,
            Field(
                default_factory=_copy_default_factory(default),
                alias=path,
                description=field_info.description,
                title=field_info.title,
                examples=field_info.examples,
                json_schema_extra=field_info.json_schema_extra,
            ),
        )

    model = create_model(
        model_name,
        __base__=SparseConfigOverrides,
        __module__=__name__,
        **field_definitions,
    )
    model.__doc__ = f"Typed sparse overrides accepted by the {model_name} policy."
    return model


def _internal_field_name(path: str) -> str:
    """Return a valid internal field name for one dotted alias."""
    if not path or any(not segment.isidentifier() for segment in path.split(".")):
        msg = f"Invalid dotted config override path: {path!r}."
        raise ValueError(msg)
    return path.replace(".", "__")


def _resolve_config_leaf(
    config_cls: type[BaseConfig],
    path: str,
) -> tuple[object, FieldInfo, object]:
    """Resolve one dotted path to its annotation, metadata, and default."""
    current_cls: type[BaseModel] = config_cls
    current_default: BaseModel = config_cls()
    segments = path.split(".")
    for index, segment in enumerate(segments):
        try:
            field_info = current_cls.model_fields[segment]
        except KeyError as exc:
            msg = f"Unknown config override path: {path!r}."
            raise ValueError(msg) from exc
        default = getattr(current_default, segment)
        if index == len(segments) - 1:
            if field_info.annotation is None:
                msg = f"Config override path has no annotation: {path!r}."
                raise TypeError(msg)
            return field_info.annotation, field_info, default
        nested_cls = field_info.annotation
        if not isinstance(nested_cls, type) or not issubclass(nested_cls, BaseModel):
            msg = f"Config override path traverses non-model field: {path!r}."
            raise TypeError(msg)
        if not isinstance(default, BaseModel):
            msg = f"Config override path has no model default: {path!r}."
            raise TypeError(msg)
        current_cls = nested_cls
        current_default = default

    raise RuntimeError("Config override path resolution completed without a leaf.")


def _annotated_field_type(annotation: object, field_info: FieldInfo) -> object:
    """Combine a field annotation with its Pydantic constraint metadata."""
    if not field_info.metadata:
        return annotation
    return Annotated.__class_getitem__((annotation, *field_info.metadata))


def _copy_default_factory(default: object) -> Callable[[], object]:
    """Return a factory that isolates mutable config defaults."""

    def factory() -> object:
        return deepcopy(default)

    return factory


RetrainConfigOverrides = build_sparse_config_overrides_model(
    "RetrainConfigOverrides",
    config_cls=RLRunConfig,
    allowed_paths=RLRunConfig.RETRAIN_MUTABLE_PATHS,
)
"""Typed sparse overrides accepted by train and resume operations."""

ForkConfigOverrides = build_sparse_config_overrides_model(
    "ForkConfigOverrides",
    config_cls=RLRunConfig,
    allowed_paths=RLRunConfig.RETRAIN_MUTABLE_PATHS | RLRunConfig.FORK_EXTRA_MUTABLE_PATHS,
)
"""Typed sparse overrides accepted by fork and extend operations."""
