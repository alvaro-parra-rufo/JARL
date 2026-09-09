"""Tests for JIT-safe scalar metric logging callbacks."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from pytest_mock import MockerFixture

from jarl.experiments.io.jit_metrics import ScalarMetricLogPayload, make_scalar_metric_log_callback
from jarl.experiments.io.metrics import JsonlMetricReader
from jarl.experiments.node import NodeMetadata, NodeWorkspace


class TestScalarMetricLogPayload:
    """Tests for the scalar metric payload contract."""

    @pytest.mark.parametrize(
        ("step", "name", "value"),
        [
            pytest.param(0, "loss", 0.5, id="step_zero"),
            pytest.param(42, "accuracy", 0.91, id="mid_training"),
        ],
    )
    def test_decode_normalizes_host_values(self, step: int, name: str, value: float) -> None:
        payload = ScalarMetricLogPayload.decode(step, name, value)

        assert payload == ScalarMetricLogPayload(step=step, name=name, value=value)

    def test_decode_converts_jax_scalars(self) -> None:
        payload = ScalarMetricLogPayload.decode(jnp.int32(7), "loss", jnp.float32(0.25))

        assert payload == ScalarMetricLogPayload(step=7, name="loss", value=0.25)


class TestScalarMetricLogCallback:
    """Tests for host callbacks crossing the JIT boundary."""

    def test_host_callback_writes_jsonl_without_workspace_in_loop(self, tmp_path: Path) -> None:
        workspace = NodeWorkspace(
            node_dir=tmp_path / "node",
            node_metadata=NodeMetadata(id="main_test_ab12cd34"),
        )
        host_log = make_scalar_metric_log_callback(workspace.log_scalar)

        with workspace:
            host_log(3, "loss", 0.25)

        assert JsonlMetricReader(workspace.metrics_jsonl_path).latest() == {"loss": 0.25}

    def test_jax_debug_callback_delegates_io_to_host_logger(self, tmp_path: Path) -> None:
        workspace = NodeWorkspace(
            node_dir=tmp_path / "node",
            node_metadata=NodeMetadata(id="main_test_ab12cd34"),
        )
        host_log = workspace.make_scalar_metric_log_callback()

        with workspace:

            @jax.jit
            def train_step(step: jax.Array, loss: jax.Array) -> jax.Array:
                jax.debug.callback(host_log, step, "loss", loss)
                return step + 1

            result = train_step(jnp.int32(1), jnp.float32(0.5))

        assert int(result) == 2
        assert JsonlMetricReader(workspace.metrics_jsonl_path).latest() == {"loss": 0.5}

    def test_jitted_loop_does_not_take_workspace_argument(self, tmp_path: Path) -> None:
        workspace = NodeWorkspace(
            node_dir=tmp_path / "node",
            node_metadata=NodeMetadata(id="main_test_ab12cd34"),
        )
        host_log = workspace.make_scalar_metric_log_callback()

        def build_train_step(
            log_callback: Callable[[Any, Any, Any], None],
        ) -> Callable[[jax.Array, jax.Array], jax.Array]:
            @jax.jit
            def train_step(step: jax.Array, loss: jax.Array) -> jax.Array:
                jax.debug.callback(log_callback, step, "loss", loss)
                return step + 1

            return train_step

        train_step = build_train_step(host_log)
        parameter_names = set(inspect.signature(train_step).parameters)

        assert parameter_names == {"loss", "step"}
        assert "workspace" not in parameter_names

        with workspace:
            train_step(jnp.int32(0), jnp.float32(0.1))

    def test_callback_receives_only_scalar_arguments(self, mocker: MockerFixture, tmp_path: Path) -> None:
        workspace = NodeWorkspace(
            node_dir=tmp_path / "node",
            node_metadata=NodeMetadata(id="main_test_ab12cd34"),
        )
        mock_log = mocker.Mock()
        host_log = make_scalar_metric_log_callback(mock_log)

        with workspace:

            @jax.jit
            def train_step(step: jax.Array, loss: jax.Array) -> None:
                jax.debug.callback(host_log, step, "accuracy", loss)

            train_step(jnp.int32(4), jnp.float32(0.75))

        mock_log.assert_called_once_with(4, "accuracy", 0.75)
