"""Host-side JIT callbacks for PPO full-JAX training."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from jarl.agents.ppo.metric_schema import MetricSchema

__all__ = [
    "MetricBundleLogFn",
    "make_metric_bundle_callback",
]

MetricBundleLogFn = Callable[[int, dict[str, float]], None]
"""Host logger invoked with ``(step, metrics_dict)``."""


def make_metric_bundle_callback(
    log_fn: MetricBundleLogFn,
    schema: MetricSchema,
) -> Callable[..., None]:
    """Build a host callback that logs a fixed-order metric vector.

    Args:
        log_fn: Host callable such as ``workspace.log_scalars``.
        schema: Metric schema defining the expected ``values`` order.

    Returns:
        Callback ``(step, values)`` for ``jax.debug.callback``.
    """
    names = schema.all_names

    def callback(step: object, values: object) -> None:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.shape[0] != len(names):
            msg = f"Expected {len(names)} metric values, got {array.shape[0]}."
            raise ValueError(msg)
        log_fn(int(step), **{name: float(array[index]) for index, name in enumerate(names)})

    return callback
