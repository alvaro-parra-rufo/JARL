"""Tests for agentic case application services."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from jarl.agentic.cases import (
    AGENTIC_CASE_RESULT_NAME,
    AgenticCaseResult,
    ValidationReport,
    describe_agentic_case,
    execute_agentic_case,
    filter_agentic_case_runs,
    get_agentic_case_registry,
    list_agentic_case_runs,
    list_agentic_cases,
    load_agentic_case_result,
    load_favorite_run_names,
    set_favorite_run,
)
from jarl.agentic.cases.services import (
    AGENTIC_CASE_FAVORITES_NAME,
    AgenticCaseStatus,
    write_agentic_case_result,
)
from jarl.agentic.progress import load_progress_events


class TestAgenticCaseCatalogServices:
    def test_list_exposes_all_builtin_cases(self) -> None:
        cases = list_agentic_cases()

        assert {case.id for case in cases} == get_agentic_case_registry().ids()
        assert all("llm" in case.requirements for case in cases)

    def test_describe_includes_scenario_and_turns(self) -> None:
        case = describe_agentic_case("graph_extend_ppo_overrides")

        assert case.experiment_case_id == "navix_prepared_root"
        assert len(case.turns) == 1
        assert case.audit_reads is True


class TestAgenticCaseExecutionServices:
    def test_execute_persists_passed_report(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
    ) -> None:
        case = mocker.Mock()
        case.run.return_value = ValidationReport(case_id="stub")
        registry = mocker.Mock()
        registry.get.return_value = case
        mocker.patch(
            "jarl.agentic.cases.services.resolve_agentic_case_registry",
            return_value=registry,
        )

        result = execute_agentic_case("stub", tmp_path / "run")
        loaded = load_agentic_case_result(tmp_path / "run")

        assert result.status == "passed"
        assert result.passed is True
        assert loaded == result

    def test_execute_persists_validation_failures(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
    ) -> None:
        case = mocker.Mock()
        case.run.return_value = ValidationReport(
            case_id="stub",
            failures=["Expected graph mutation."],
        )
        registry = mocker.Mock()
        registry.get.return_value = case
        mocker.patch(
            "jarl.agentic.cases.services.resolve_agentic_case_registry",
            return_value=registry,
        )

        result = execute_agentic_case("stub", tmp_path / "run")
        end_events = [event for event in load_progress_events(tmp_path / "run") if event.kind == "case_end"]

        assert result.status == "failed"
        assert result.failures == ("Expected graph mutation.",)
        assert result.error is None
        assert end_events
        assert "Expected graph mutation." in end_events[0].message
        assert end_events[0].data is not None
        assert end_events[0].data["failures"] == ["Expected graph mutation."]

    def test_execute_converts_runtime_exception_to_error_result(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
    ) -> None:
        case = mocker.Mock()
        case.run.side_effect = RuntimeError("provider unavailable")
        registry = mocker.Mock()
        registry.get.return_value = case
        mocker.patch(
            "jarl.agentic.cases.services.resolve_agentic_case_registry",
            return_value=registry,
        )

        result = execute_agentic_case("stub", tmp_path / "run")

        assert result.status == "error"
        assert result.error == "RuntimeError: provider unavailable"
        assert load_agentic_case_result(tmp_path / "run") == result


def _write_run(
    parent: Path,
    name: str,
    *,
    case_id: str,
    status: AgenticCaseStatus,
    finished_at: str,
    started_at: str = "2026-08-01T00:00:00+00:00",
) -> AgenticCaseResult:
    dest = parent / name
    dest.mkdir()
    result = AgenticCaseResult(
        case_id=case_id,
        experiment_dir=dest,
        status=status,
        failures=(),
        error=None,
        started_at=started_at,
        finished_at=finished_at,
    )
    write_agentic_case_result(result)
    return result


class TestLoadAgenticCaseResult:
    def test_anchors_experiment_dir_to_load_path(self, tmp_path: Path) -> None:
        dest = tmp_path / "moved-run"
        dest.mkdir()
        payload = {
            "case_id": "graph_create_root",
            "experiment_dir": "/old/location/graph_create_root-stale",
            "status": "passed",
            "passed": True,
            "failures": [],
            "error": None,
            "started_at": "2026-08-01T00:00:00+00:00",
            "finished_at": "2026-08-01T00:01:00+00:00",
        }
        (dest / AGENTIC_CASE_RESULT_NAME).write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

        loaded = load_agentic_case_result(dest)

        assert loaded is not None
        assert loaded.experiment_dir == dest.resolve()
        assert loaded.case_id == "graph_create_root"


class TestListAgenticCaseRuns:
    def test_lists_finished_runs_newest_first(self, tmp_path: Path) -> None:
        _write_run(
            tmp_path,
            "older-aaaa",
            case_id="graph_create_root",
            status="passed",
            finished_at="2026-08-01T10:00:00+00:00",
        )
        _write_run(
            tmp_path,
            "newer-bbbb",
            case_id="env_navix_maps_read",
            status="failed",
            finished_at="2026-08-02T10:00:00+00:00",
        )
        (tmp_path / "incomplete").mkdir()
        (tmp_path / "stray.log").write_text("log\n", encoding="utf-8")

        runs = list_agentic_case_runs(cases_root=tmp_path)

        assert [run.experiment_dir.name for run in runs] == ["newer-bbbb", "older-aaaa"]
        assert runs[0].status == "failed"
        assert runs[1].status == "passed"

    def test_skips_malformed_result_and_missing_parent(self, tmp_path: Path) -> None:
        broken = tmp_path / "broken-cccc"
        broken.mkdir()
        (broken / AGENTIC_CASE_RESULT_NAME).write_text("{not-json", encoding="utf-8")

        runs = list_agentic_case_runs(cases_root=tmp_path)
        missing = list_agentic_case_runs(cases_root=tmp_path / "absent")

        assert runs == ()
        assert missing == ()

    def test_anchors_listed_run_to_discovered_directory(self, tmp_path: Path) -> None:
        dest = tmp_path / "navix_empty_variant_spec-20260826-223109-aa6e07a8"
        dest.mkdir()
        payload = {
            "case_id": "navix_empty_variant_spec",
            "experiment_dir": "/old/location/navix_empty_variant_spec-stale",
            "status": "passed",
            "passed": True,
            "failures": [],
            "error": None,
            "started_at": "2026-08-26T22:31:20+00:00",
            "finished_at": "2026-08-26T22:42:20+00:00",
        }
        (dest / AGENTIC_CASE_RESULT_NAME).write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

        runs = list_agentic_case_runs(cases_root=tmp_path)

        assert len(runs) == 1
        assert runs[0].experiment_dir == dest.resolve()


class TestFilterAgenticCaseRuns:
    def test_filters_by_case_id_status_and_query(self, tmp_path: Path) -> None:
        first = _write_run(
            tmp_path,
            "env_navix_maps_read-20260821-152843-5abb44fd",
            case_id="env_navix_maps_read",
            status="passed",
            finished_at="2026-08-21T15:28:43+00:00",
        )
        second = _write_run(
            tmp_path,
            "graph_create_root-20260820-100000-deadbeef",
            case_id="graph_create_root",
            status="error",
            finished_at="2026-08-20T10:00:00+00:00",
        )
        third = _write_run(
            tmp_path,
            "env_navix_maps_read-20260811-184814-8e6c4087",
            case_id="env_navix_maps_read",
            status="failed",
            finished_at="2026-08-11T18:48:14+00:00",
        )
        runs = (first, second, third)

        by_id = filter_agentic_case_runs(runs, case_id="env_navix_maps_read")
        by_status = filter_agentic_case_runs(runs, statuses=["error"])
        by_query = filter_agentic_case_runs(runs, query="5abb44fd")
        empty_filters = filter_agentic_case_runs(runs, case_id="", statuses=[], query="  ")

        assert [run.experiment_dir.name for run in by_id] == [
            first.experiment_dir.name,
            third.experiment_dir.name,
        ]
        assert by_status == (second,)
        assert by_query == (first,)
        assert empty_filters == runs

    @pytest.mark.parametrize(
        ("query", "expected_names"),
        [
            ("GRAPH_CREATE", ("graph_create_root-aaaa",)),
            ("zzzz", ()),
        ],
    )
    def test_query_is_case_insensitive(
        self,
        tmp_path: Path,
        query: str,
        expected_names: tuple[str, ...],
    ) -> None:
        _write_run(
            tmp_path,
            "graph_create_root-aaaa",
            case_id="graph_create_root",
            status="passed",
            finished_at="2026-08-01T00:00:00+00:00",
        )
        _write_run(
            tmp_path,
            "env_navix_maps_read-bbbb",
            case_id="env_navix_maps_read",
            status="passed",
            finished_at="2026-08-02T00:00:00+00:00",
        )
        runs = list_agentic_case_runs(cases_root=tmp_path)

        filtered = filter_agentic_case_runs(runs, query=query)

        assert tuple(run.experiment_dir.name for run in filtered) == expected_names

    def test_filters_by_catalog_tags(self, tmp_path: Path) -> None:
        env_run = _write_run(
            tmp_path,
            "env-run",
            case_id="env_navix_maps_read",
            status="passed",
            finished_at="2026-08-02T00:00:00+00:00",
        )
        graph_run = _write_run(
            tmp_path,
            "graph-run",
            case_id="graph_create_root",
            status="passed",
            finished_at="2026-08-01T00:00:00+00:00",
        )
        tags_by_case_id = {
            "env_navix_maps_read": ("env", "read"),
            "graph_create_root": ("graph", "write"),
        }

        by_tag = filter_agentic_case_runs(
            (env_run, graph_run),
            tags=("env",),
            tags_by_case_id=tags_by_case_id,
        )
        by_query_tag = filter_agentic_case_runs(
            (env_run, graph_run),
            query="READ",
            tags_by_case_id=tags_by_case_id,
        )

        assert by_tag == (env_run,)
        assert by_query_tag == (env_run,)

    def test_favorites_only_keeps_named_runs(self, tmp_path: Path) -> None:
        _write_run(
            tmp_path,
            "kept-run",
            case_id="graph_create_root",
            status="passed",
            finished_at="2026-08-02T00:00:00+00:00",
        )
        _write_run(
            tmp_path,
            "other-run",
            case_id="env_navix_maps_read",
            status="passed",
            finished_at="2026-08-01T00:00:00+00:00",
        )
        runs = list_agentic_case_runs(cases_root=tmp_path)

        filtered = filter_agentic_case_runs(
            runs,
            favorite_names=["kept-run"],
            favorites_only=True,
        )
        empty = filter_agentic_case_runs(runs, favorites_only=True)

        assert tuple(run.experiment_dir.name for run in filtered) == ("kept-run",)
        assert empty == ()


class TestFavoriteRunNames:
    def test_roundtrip_add_and_remove(self, tmp_path: Path) -> None:
        added = set_favorite_run("navix_empty_variant_spec-aaaa", favorite=True, cases_root=tmp_path)
        loaded = load_favorite_run_names(cases_root=tmp_path)
        removed = set_favorite_run(
            "navix_empty_variant_spec-aaaa",
            favorite=False,
            cases_root=tmp_path,
        )

        assert added == frozenset({"navix_empty_variant_spec-aaaa"})
        assert loaded == added
        assert removed == frozenset()

    def test_skips_unsafe_names_and_missing_file(self, tmp_path: Path) -> None:
        (tmp_path / AGENTIC_CASE_FAVORITES_NAME).write_text(
            json.dumps({"names": ["ok-run", "../escape", "a/b"]}) + "\n",
            encoding="utf-8",
        )

        names = load_favorite_run_names(cases_root=tmp_path)
        missing = load_favorite_run_names(cases_root=tmp_path / "absent")

        assert names == frozenset({"ok-run"})
        assert missing == frozenset()

    def test_rejects_unsafe_name_on_write(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="single path component"):
            set_favorite_run("../escape", favorite=True, cases_root=tmp_path)
