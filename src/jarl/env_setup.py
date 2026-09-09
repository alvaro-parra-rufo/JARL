"""Setup the environment for the jarl package."""

from __future__ import annotations

import contextlib
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jarl.experiments.run_config import JaxConfig

__all__ = [
    "IS_JAX_SETUP",
    "log_jax_training_devices",
    "setup_jax",
]

# -------------------------------------------------------------------------------------------------------------------- #
# ---                                                  JAX setup                                                   --- #
# -------------------------------------------------------------------------------------------------------------------- #

IS_JAX_SETUP = False
"""Whether `setup_jax()` has completed."""


def setup_jax(config: JaxConfig | None = None) -> bool:
    """Configure JAX and hide TensorFlow GPUs before other jarl imports.

    Applies ``JaxConfig`` before device discovery when ``config`` is provided.
    TensorFlow is only used for data loading in learning examples and must not
    claim GPU memory from JAX.

    Args:
        config: Optional JAX runtime settings from the resolved run config.

    Returns:
        `True` when setup completed.
    """
    global IS_JAX_SETUP

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    import jax

    if config is not None:
        jax.config.update("jax_default_matmul_precision", config.default_matmul_precision)
        jax.config.update("jax_exec_time_optimization_effort", config.exec_time_optimization_effort)
        jax.config.update("jax_memory_fitting_effort", config.memory_fitting_effort)
        jax.config.update("jax_compilation_cache_dir", config.compilation_cache_dir)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

    jax.devices()

    with contextlib.suppress(Exception):
        # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make it unavailable to JAX.
        import tensorflow as tf

        tf.config.experimental.set_visible_devices([], "GPU")

    IS_JAX_SETUP = True
    return True


def log_jax_training_devices(
    logger: logging.Logger,
    *,
    require_gpu: bool = False,
) -> bool:
    """Log JAX backend/devices before training and return whether GPU is active.

    Logs the selected devices for the full-JAX trainers. When ``require_gpu`` is
    ``True``, raises if JAX is not on the GPU backend.

    Args:
        logger: Logger used for ``train.log`` / trainer output.
        require_gpu: When ``True``, fail fast if JAX backend is not ``gpu``.

    Returns:
        ``True`` when ``jax.default_backend()`` is ``gpu``.

    Raises:
        RuntimeError: If ``require_gpu`` is ``True`` and no GPU backend is active.
    """
    if not IS_JAX_SETUP:
        setup_jax()

    import jax

    backend = jax.default_backend()
    devices = jax.devices()
    device_labels = ", ".join(f"{device.platform}:{device.id}" for device in devices)
    using_gpu = backend == "gpu"

    logger.info("Using device: %s", backend)
    logger.info("JAX devices: %s", device_labels or "none")

    if require_gpu and not using_gpu:
        msg = f"Training requires GPU but JAX backend is {backend!r}. Visible devices: {device_labels or 'none'}."
        raise RuntimeError(msg)

    if not using_gpu:
        logger.warning(
            "Training is running on JAX backend %r. Unset JAX_PLATFORMS or configure CUDA for GPU execution.",
            backend,
        )

    return using_gpu
