"""Metadata shared by reusable experiment and agentic cases."""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["CaseSpec"]

_CASE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


@dataclass(frozen=True, slots=True)
class CaseSpec:
    """Stable identity and discovery metadata for a reusable case.

    Args:
        id: Stable machine-readable identifier.
        title: Short human-readable name.
        description: Purpose and expected behavior of the case.
        objective: Optional LLM goal override. Empty uses the non-interactive
            default from the case driver.
        tags: Labels used to filter case catalogs.
        requirements: Runtime capabilities required to execute the case.
    """

    id: str
    title: str
    description: str = ""
    objective: str = ""
    tags: frozenset[str] = frozenset()
    requirements: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        """Validate identity and filter metadata."""
        if not _CASE_ID_PATTERN.fullmatch(self.id):
            msg = f"Invalid case id {self.id!r}; use lowercase letters, digits, dots, underscores, or hyphens."
            raise ValueError(msg)
        if not self.title.strip():
            raise ValueError("Case title must not be empty.")
        if any(not value.strip() for value in self.tags):
            raise ValueError("Case tags must not contain empty values.")
        if any(not value.strip() for value in self.requirements):
            raise ValueError("Case requirements must not contain empty values.")

    @property
    def resolved_objective(self) -> str | None:
        """Return a non-empty `objective`, or `None` to use the driver default."""
        text = self.objective.strip()
        return text or None
