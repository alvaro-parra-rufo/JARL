"""Simple deterministic validation reports for agentic cases."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

__all__ = ["ValidationReport"]


@dataclass(slots=True)
class ValidationReport:
    """Accumulated deterministic validation failures for one case.

    Args:
        case_id: Stable identifier of the validated case.
        failures: Human-readable failed expectations. An empty list passes.
    """

    case_id: str
    failures: list[str] = field(default_factory=list)
    _halted: bool = field(default=False, repr=False)

    @property
    def passed(self) -> bool:
        """Return whether no validation failure was recorded."""
        return not self.failures

    @property
    def halted(self) -> bool:
        """Return whether an irreversible failure was recorded.

        Once true, the flag stays true for the rest of the report.
        """
        return self._halted

    def add_failure(self, message: str, *, halted: bool = False) -> None:
        """Record one failed expectation.

        Args:
            message: Human-readable failed expectation.
            halted: When true, mark the case as irreversibly failed.
        """
        normalized = message.strip()
        if not normalized:
            raise ValueError("Validation failure message must not be empty.")
        self.failures.append(normalized)
        if halted:
            self._halted = True

    def check(self, condition: bool, message: str, *, halted: bool = False) -> None:
        """Record ``message`` when ``condition`` is false.

        Args:
            condition: Expectation that must hold.
            message: Failure text recorded when ``condition`` is false.
            halted: Forwarded to ``add_failure`` when the check fails.
        """
        if not condition:
            self.add_failure(message, halted=halted)

    def extend(self, failures: Iterable[str]) -> None:
        """Record multiple failed expectations."""
        for failure in failures:
            self.add_failure(failure)

    def assert_passed(self) -> None:
        """Raise an assertion with all failures when validation did not pass."""
        if self.passed:
            return
        details = "\n".join(f"- {failure}" for failure in self.failures)
        msg = f"Agentic case {self.case_id!r} failed validation:\n{details}"
        raise AssertionError(msg)
