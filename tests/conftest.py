"""Project-wide test configuration."""

from __future__ import annotations

import os
from collections.abc import Generator

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

PATH_TO_MARKER = {
    "unit": "unit",
    "integration": "integration",
    "functional": "functional",
    "llm_eval": "llm_eval",
    "test_helpers": "test_helpers",
}
"""Mapping from test directory name to pytest marker name."""


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-apply markers based on test file path."""
    for item in items:
        rel_path = item.path.as_posix()
        for dirname, marker_name in PATH_TO_MARKER.items():
            if f"tests/{dirname}/" in rel_path:
                item.add_marker(getattr(pytest.mark, marker_name))
                break


@pytest.fixture()
def with_gpu() -> Generator[None]:
    """Enable GPU backend for a single test. Use with `@pytest.mark.gpu`.

    Overrides `JAX_PLATFORMS` to include CUDA, verifies a GPU is visible to JAX,
    and restores the original setting on teardown.
    """
    original = os.environ.get("JAX_PLATFORMS")
    os.environ["JAX_PLATFORMS"] = ""

    import jax

    if not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("No GPU available")

    yield

    if original is not None:
        os.environ["JAX_PLATFORMS"] = original
    else:
        os.environ.pop("JAX_PLATFORMS", None)
