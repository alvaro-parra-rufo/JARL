"""Shared environment contract types for JARL adapters."""

from __future__ import annotations

from enum import Enum

__all__ = [
    "ActionSpaceType",
    "DataInterfaceType",
    "DeepLearningFrameworkType",
    "ObservationSpaceType",
]


class ActionSpaceType(Enum):
    """Supported action-space categories."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


class ObservationSpaceType(Enum):
    """Supported observation-space categories."""

    FLAT_VALUES = "flat_values"
    IMAGES = "images"


class DataInterfaceType(Enum):
    """Supported environment data interfaces."""

    NUMPY = "numpy"
    JAX = "jax"


class DeepLearningFrameworkType(Enum):
    """Supported deep-learning frameworks."""

    JAX = "jax"
    PYTORCH = "pytorch"
