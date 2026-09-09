"""Tests for PPO metric schemas and JIT-safe packing."""

from __future__ import annotations

import numpy as np
import pytest

from jarl.agents.ppo.callbacks import make_metric_bundle_callback
from jarl.agents.ppo.metric_schema import (
    PPO_EVAL_METRIC_SCHEMA,
    PPO_TRAIN_METRIC_SCHEMA,
    MetricSchema,
    assemble_metric_bundle,
    pack_device_metrics,
)


def _sentinel_mapping(names: tuple[str, ...], *, start: float = 1.0) -> dict[str, float]:
    return {name: start + index for index, name in enumerate(names)}


class TestMetricSchemaValidation:
    """Schema construction rejects invalid name sets."""

    def test_rejects_duplicate_device_names(self) -> None:
        with pytest.raises(ValueError, match="Duplicate metric names"):
            MetricSchema(device_names=("loss/a", "loss/a"))

    def test_rejects_duplicate_host_names(self) -> None:
        with pytest.raises(ValueError, match="Duplicate metric names"):
            MetricSchema(device_names=("loss/a",), host_names=("time/sps", "time/sps"))

    def test_rejects_overlap_between_device_and_host(self) -> None:
        with pytest.raises(ValueError, match="both device and host"):
            MetricSchema(device_names=("loss/a",), host_names=("loss/a",))


class TestPackDeviceMetrics:
    """Device packing maps sentinel values to the expected names."""

    def test_train_schema_assigns_sentinel_per_device_name(self) -> None:
        values = _sentinel_mapping(PPO_TRAIN_METRIC_SCHEMA.device_names)
        packed = np.asarray(pack_device_metrics(values, PPO_TRAIN_METRIC_SCHEMA))

        for index, name in enumerate(PPO_TRAIN_METRIC_SCHEMA.device_names):
            assert packed[index] == pytest.approx(values[name])

    def test_eval_schema_assigns_sentinel_per_device_name(self) -> None:
        values = _sentinel_mapping(PPO_EVAL_METRIC_SCHEMA.device_names, start=10.0)
        packed = np.asarray(pack_device_metrics(values, PPO_EVAL_METRIC_SCHEMA))

        for index, name in enumerate(PPO_EVAL_METRIC_SCHEMA.device_names):
            assert packed[index] == pytest.approx(values[name])

    def test_missing_device_metric_raises(self) -> None:
        values = _sentinel_mapping(PPO_EVAL_METRIC_SCHEMA.device_names)
        del values[PPO_EVAL_METRIC_SCHEMA.device_names[0]]

        with pytest.raises(KeyError, match="Missing device metric values"):
            pack_device_metrics(values, PPO_EVAL_METRIC_SCHEMA)

    def test_extra_device_metric_raises(self) -> None:
        values = _sentinel_mapping(PPO_EVAL_METRIC_SCHEMA.device_names)
        values["eval/extra"] = 99.0

        with pytest.raises(KeyError, match="Unexpected device metric values"):
            pack_device_metrics(values, PPO_EVAL_METRIC_SCHEMA)


class TestAssembleMetricBundle:
    """Host assembly preserves full schema order."""

    def test_train_bundle_maps_sentinels_to_all_names(self) -> None:
        device_values = _sentinel_mapping(PPO_TRAIN_METRIC_SCHEMA.device_names)
        host_values = _sentinel_mapping(PPO_TRAIN_METRIC_SCHEMA.host_names, start=100.0)
        device_body = np.asarray(pack_device_metrics(device_values, PPO_TRAIN_METRIC_SCHEMA))
        bundle = assemble_metric_bundle(device_body, host_values, PPO_TRAIN_METRIC_SCHEMA)

        expected = {**device_values, **host_values}
        for index, name in enumerate(PPO_TRAIN_METRIC_SCHEMA.all_names):
            assert bundle[index] == pytest.approx(expected[name])

    def test_missing_host_metric_raises(self) -> None:
        device_body = np.asarray(
            pack_device_metrics(_sentinel_mapping(PPO_TRAIN_METRIC_SCHEMA.device_names), PPO_TRAIN_METRIC_SCHEMA)
        )

        with pytest.raises(KeyError, match="Missing host metric values"):
            assemble_metric_bundle(device_body, {}, PPO_TRAIN_METRIC_SCHEMA)

    def test_extra_host_metric_raises(self) -> None:
        device_body = np.asarray(
            pack_device_metrics(_sentinel_mapping(PPO_TRAIN_METRIC_SCHEMA.device_names), PPO_TRAIN_METRIC_SCHEMA)
        )
        host_values = _sentinel_mapping(PPO_TRAIN_METRIC_SCHEMA.host_names, start=100.0)
        host_values["steps/extra"] = 1.0

        with pytest.raises(KeyError, match="Unexpected host metric values"):
            assemble_metric_bundle(device_body, host_values, PPO_TRAIN_METRIC_SCHEMA)


class TestMetricOrderAntiCorruption:
    """Permuting schema order without updating values must break mapping."""

    def test_permuted_device_order_maps_wrong_sentinels(self) -> None:
        values = _sentinel_mapping(PPO_TRAIN_METRIC_SCHEMA.device_names)
        canonical = np.asarray(pack_device_metrics(values, PPO_TRAIN_METRIC_SCHEMA))
        permuted_schema = MetricSchema(
            device_names=tuple(reversed(PPO_TRAIN_METRIC_SCHEMA.device_names)),
            host_names=PPO_TRAIN_METRIC_SCHEMA.host_names,
        )
        permuted = np.asarray(pack_device_metrics(values, permuted_schema))

        assert not np.allclose(canonical, permuted)

    def test_permuted_device_order_logs_wrong_name_value_pairs(self) -> None:
        values = _sentinel_mapping(PPO_TRAIN_METRIC_SCHEMA.device_names)
        device_body = np.asarray(pack_device_metrics(values, PPO_TRAIN_METRIC_SCHEMA))
        host_values = _sentinel_mapping(PPO_TRAIN_METRIC_SCHEMA.host_names, start=100.0)
        canonical_bundle = assemble_metric_bundle(device_body, host_values, PPO_TRAIN_METRIC_SCHEMA)

        permuted_schema = MetricSchema(
            device_names=tuple(reversed(PPO_TRAIN_METRIC_SCHEMA.device_names)),
            host_names=PPO_TRAIN_METRIC_SCHEMA.host_names,
        )
        permuted_bundle = assemble_metric_bundle(device_body, host_values, permuted_schema)

        logged: dict[str, float] = {}

        def log_fn(_step: int, **metrics: float) -> None:
            logged.update(metrics)

        callback = make_metric_bundle_callback(log_fn, permuted_schema)
        callback(42, permuted_bundle)

        mismatched_device = [
            name
            for name in PPO_TRAIN_METRIC_SCHEMA.device_names
            if logged[name] != pytest.approx(canonical_bundle[PPO_TRAIN_METRIC_SCHEMA.all_names.index(name)])
        ]
        assert mismatched_device


class TestDebugMetricPacking:
    """New debug metrics map packed indices to the intended rollout tensors."""

    _DEBUG_METRIC_NAMES: tuple[str, ...] = (
        "rollout/value_mean",
        "rollout/return_mean",
        "rollout/advantage_std",
        "policy_ratio/max",
    )

    def test_debug_metrics_present_in_train_schema(self) -> None:
        for name in self._DEBUG_METRIC_NAMES:
            assert name in PPO_TRAIN_METRIC_SCHEMA.device_names

    def test_pack_device_metrics_maps_debug_values_to_schema_indices(self) -> None:
        values = _sentinel_mapping(PPO_TRAIN_METRIC_SCHEMA.device_names)
        values["rollout/value_mean"] = 0.25
        values["rollout/return_mean"] = 1.75
        values["rollout/advantage_std"] = 0.5
        values["policy_ratio/max"] = 3.0
        packed = np.asarray(pack_device_metrics(values, PPO_TRAIN_METRIC_SCHEMA))

        for name, expected in (
            ("rollout/value_mean", 0.25),
            ("rollout/return_mean", 1.75),
            ("rollout/advantage_std", 0.5),
            ("policy_ratio/max", 3.0),
        ):
            index = PPO_TRAIN_METRIC_SCHEMA.device_names.index(name)
            assert packed[index] == pytest.approx(expected)

    def test_permuted_debug_metric_order_maps_wrong_values(self) -> None:
        values = _sentinel_mapping(PPO_TRAIN_METRIC_SCHEMA.device_names)
        values["rollout/value_mean"] = 0.25
        values["rollout/return_mean"] = 1.75
        values["rollout/advantage_std"] = 0.5
        values["policy_ratio/max"] = 3.0
        canonical = np.asarray(pack_device_metrics(values, PPO_TRAIN_METRIC_SCHEMA))

        swapped_schema = MetricSchema(
            device_names=_swap_schema_names(
                PPO_TRAIN_METRIC_SCHEMA.device_names,
                "rollout/value_mean",
                "rollout/return_mean",
            ),
            host_names=PPO_TRAIN_METRIC_SCHEMA.host_names,
        )
        permuted = np.asarray(pack_device_metrics(values, swapped_schema))

        value_index = PPO_TRAIN_METRIC_SCHEMA.device_names.index("rollout/value_mean")
        assert canonical[value_index] != pytest.approx(permuted[value_index])


def _swap_schema_names(
    names: tuple[str, ...],
    first_name: str,
    second_name: str,
) -> tuple[str, ...]:
    names_list = list(names)
    first_index = names_list.index(first_name)
    second_index = names_list.index(second_name)
    names_list[first_index], names_list[second_index] = names_list[second_index], names_list[first_index]
    return tuple(names_list)


class TestMetricBundleCallback:
    """Callback validates bundle length outside compiled code."""

    def test_rejects_wrong_bundle_length(self) -> None:
        callback = make_metric_bundle_callback(lambda _step, **_metrics: None, PPO_EVAL_METRIC_SCHEMA)

        with pytest.raises(ValueError, match="Expected 2 metric values"):
            callback(1, np.array([1.0]))
