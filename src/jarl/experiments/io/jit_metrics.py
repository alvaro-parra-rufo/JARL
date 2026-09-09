"""JIT-safe scalar metric logging callbacks for node workspaces."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = [
    "MetricLogFn",
    "ScalarMetricLogPayload",
    "make_scalar_metric_log_callback",
]

MetricLogFn = Callable[[int, str, float], None]
"""Host-side logger invoked with decoded ``(step, name, value)`` scalars."""


@dataclass(frozen=True, slots=True)
class ScalarMetricLogPayload:
    """Metric data allowed to cross the full-JIT logging boundary.

    Training loops compiled with ``jax.jit`` must not receive
    ``NodeWorkspace`` instances or other complex Python objects. Only plain
    scalars should reach the host callback produced by
    ``make_scalar_metric_log_callback``.

    Args:
        step: Training step associated with the measurement.
        name: Metric identifier.
        value: Scalar metric value.
    """

    step: int
    name: str
    value: float

    @classmethod
    def decode(cls, step: Any, name: Any, value: Any) -> ScalarMetricLogPayload:
        """Convert JAX-traced or host values into a validated payload.

        Args:
            step: Step scalar from JAX or Python.
            name: Metric name from JAX or Python.
            value: Value scalar from JAX or Python.

        Returns:
            Normalized payload with Python scalar types.
        """
        return cls(step=int(step), name=str(name), value=float(value))


def make_scalar_metric_log_callback(log_fn: MetricLogFn) -> Callable[[Any, Any, Any], None]:
    """Build a host callback compatible with ``jax.debug.callback``.

    Create the callback outside compiled code and bind it to a workspace logger
    on the host, for example ``workspace.log_scalar``. Inside a ``jax.jit``
    training loop, invoke it with traced scalars only:

    ```python
    host_log = make_scalar_metric_log_callback(workspace.log_scalar)

    @jax.jit
    def train_step(step, loss):
        jax.debug.callback(host_log, step, "loss", loss)
        return step + 1
    ```

    ``log_fn`` performs all filesystem and tracking IO. The compiled loop must
    never capture or receive ``NodeWorkspace`` directly.

    Args:
        log_fn: Host logger that persists one scalar metric.

    Returns:
        Callback accepting ``(step, name, value)`` suitable for
        ``jax.debug.callback``.
    """

    def callback(step: Any, name: Any, value: Any) -> None:
        payload = ScalarMetricLogPayload.decode(step, name, value)
        log_fn(payload.step, payload.name, payload.value)

    return callback
