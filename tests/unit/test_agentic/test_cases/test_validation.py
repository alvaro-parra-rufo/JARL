"""Tests for deterministic agentic validation reports."""

from __future__ import annotations

import pytest

from jarl.agentic.cases import ValidationReport


class TestValidationReport:
    def test_empty_report_passes(self) -> None:
        report = ValidationReport(case_id="sample")

        report.assert_passed()

        assert report.passed is True

    def test_check_accumulates_only_failed_expectations(self) -> None:
        report = ValidationReport(case_id="sample")

        report.check(True, "must not be added")
        report.check(False, "first failure")
        report.extend(["second failure"])

        assert report.passed is False
        assert report.failures == ["first failure", "second failure"]
        assert report.halted is False

    def test_halted_is_sticky_after_a_halt_failure(self) -> None:
        report = ValidationReport(case_id="sample")

        report.check(False, "missing tool")
        report.add_failure("irreversible mutation", halted=True)
        report.check(False, "still incomplete")

        assert report.passed is False
        assert report.halted is True
        assert report.failures == ["missing tool", "irreversible mutation", "still incomplete"]

    def test_check_forwards_halted_to_add_failure(self) -> None:
        report = ValidationReport(case_id="sample")

        report.check(False, "training is forbidden", halted=True)

        assert report.halted is True
        assert report.failures == ["training is forbidden"]

    def test_successful_halted_check_does_not_halt(self) -> None:
        report = ValidationReport(case_id="sample")

        report.check(True, "must not halt", halted=True)

        assert report.passed is True
        assert report.halted is False

    def test_assert_passed_reports_all_failures(self) -> None:
        report = ValidationReport(
            case_id="sample",
            failures=["first failure", "second failure"],
        )

        with pytest.raises(AssertionError, match=r"(?s)first failure.*second failure"):
            report.assert_passed()

    def test_rejects_empty_failure_message(self) -> None:
        report = ValidationReport(case_id="sample")

        with pytest.raises(ValueError, match="must not be empty"):
            report.add_failure(" ")
