"""Discovery and filtering for reusable case catalogs."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Iterable, Iterator

from jarl.experiments.cases.case import CaseDefinition
from jarl.utils.pattern_filter import match_patterns

__all__ = ["CaseRegistry"]


class CaseRegistry[CaseT: CaseDefinition]:
    """Index of reusable cases keyed by stable case id."""

    def __init__(self, cases: Iterable[CaseT] = ()) -> None:
        """Initialize and validate a registry from case instances."""
        entries: dict[str, CaseT] = {}
        for case in cases:
            case_id = case.spec.id
            if case_id in entries:
                msg = f"Duplicate case id: {case_id!r}"
                raise ValueError(msg)
            entries[case_id] = case
        self._entries = entries

    def __len__(self) -> int:
        """Return the number of registered cases."""
        return len(self._entries)

    def __iter__(self) -> Iterator[CaseT]:
        """Iterate cases in discovery or construction order."""
        return iter(self._entries.values())

    @classmethod
    def from_package(
        cls,
        package: str,
        *,
        case_type: type[CaseT],
    ) -> CaseRegistry[CaseT]:
        """Discover explicit ``CASE`` exports under a package.

        Args:
            package: Importable package whose leaf modules define cases.
            case_type: Required runtime type for each ``CASE`` export.

        Returns:
            Validated registry containing all discovered cases.

        Raises:
            TypeError: If a discovered ``CASE`` has the wrong type.
            ValueError: If the package is invalid or a leaf omits ``CASE``.
        """
        package_module = importlib.import_module(package)
        package_path = getattr(package_module, "__path__", None)
        if package_path is None:
            msg = f"Package {package!r} has no __path__ for discovery."
            raise ValueError(msg)

        cases: list[CaseT] = []
        prefix = f"{package}."
        for module_info in pkgutil.walk_packages(package_path, prefix=prefix):
            if module_info.ispkg:
                continue
            module_basename = module_info.name.rsplit(".", maxsplit=1)[-1]
            if module_basename.startswith("_"):
                continue
            module = importlib.import_module(module_info.name)
            case = getattr(module, "CASE", None)
            if case is None:
                msg = f"Case module {module_info.name!r} is missing CASE."
                raise ValueError(msg)
            if not isinstance(case, case_type):
                msg = f"CASE in {module_info.name!r} must be an instance of {case_type.__name__}."
                raise TypeError(msg)
            cases.append(case)
        return cls(cases)

    def ids(self) -> frozenset[str]:
        """Return all registered case ids."""
        return frozenset(self._entries)

    def values(self) -> tuple[CaseT, ...]:
        """Return registered cases in stable registry order."""
        return tuple(self._entries.values())

    def get(self, case_id: str) -> CaseT:
        """Return the case identified by ``case_id``.

        Raises:
            KeyError: If the case is not registered.
        """
        return self._entries[case_id]

    def filter_by(
        self,
        *,
        include_ids: set[str] | None = None,
        exclude_ids: set[str] | None = None,
        include_tags: set[str] | None = None,
        exclude_tags: set[str] | None = None,
    ) -> CaseRegistry[CaseT]:
        """Return a registry filtered by case ids and tags."""
        cases = (
            case
            for case in self._entries.values()
            if match_patterns(
                [case.spec.id],
                include=include_ids,
                exclude=exclude_ids,
            )
            and match_patterns(
                case.spec.tags,
                include=include_tags,
                exclude=exclude_tags,
            )
        )
        return CaseRegistry(cases)
