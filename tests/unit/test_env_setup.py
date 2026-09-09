"""Tests for JAX environment setup helpers."""

from __future__ import annotations

import logging

import pytest

from jarl.env_setup import log_jax_training_devices


class TestLogJaxTrainingDevices:
    """Tests for training-time JAX device logging."""

    def test_logs_backend_on_cpu(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO)
        logger = logging.getLogger("test.jax.devices")

        using_gpu = log_jax_training_devices(logger)

        assert isinstance(using_gpu, bool)
        assert any("Using device:" in record.message for record in caplog.records)
        assert any("JAX devices:" in record.message for record in caplog.records)

    def test_require_gpu_raises_on_cpu(self) -> None:
        import jax

        if jax.default_backend() == "gpu":
            pytest.skip("Test targets CPU-only environments.")

        logger = logging.getLogger("test.jax.devices.require_gpu")
        with pytest.raises(RuntimeError, match="requires GPU"):
            log_jax_training_devices(logger, require_gpu=True)
