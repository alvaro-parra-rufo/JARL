"""Tests for agentic session metadata."""

from __future__ import annotations

from pathlib import Path

from jarl.agentic.session import AgenticSession, CompiledNodeInfo, load_session, save_session


class TestCompiledNodes:
    def test_with_compiled_nodes_replaces_the_entire_map(self) -> None:
        session = AgenticSession.create_new()
        first = session.with_compiled_nodes(
            {
                "setup": CompiledNodeInfo(system_prompt="setup-a"),
                "stale": CompiledNodeInfo(system_prompt="gone"),
            }
        )

        replaced = first.with_compiled_nodes(
            {"operate": CompiledNodeInfo(system_prompt="operate-b")},
        )

        assert set(replaced.compiled_nodes) == {"operate"}
        assert replaced.compiled_nodes["operate"].system_prompt == "operate-b"
        assert "stale" not in replaced.compiled_nodes
        assert "setup" not in replaced.compiled_nodes


class TestCompiledNodeInfoFilterBy:
    def test_from_bind_persists_sorted_lists(self) -> None:
        info = CompiledNodeInfo.from_bind(
            system_prompt="setup",
            filter_by={"include_labels": {"setup"}},
            tool_names=("session_status",),
        )

        assert info.filter_by == {"include_labels": ["setup"]}
        assert info.as_filter_by() == {"include_labels": {"setup"}}

    def test_from_bind_persists_set_keys_as_sorted_lists(self) -> None:
        info = CompiledNodeInfo.from_bind(
            system_prompt=None,
            filter_by={
                "include_labels": {"operate"},
                "exclude_labels": {"finish"},
                "include_names": {"graph_fork"},
                "exclude_names": set(),
            },
            tool_names=("graph_fork",),
        )

        assert info.filter_by == {
            "include_labels": ["operate"],
            "exclude_labels": ["finish"],
            "include_names": ["graph_fork"],
        }
        assert "exclude_names" not in info.filter_by
        assert info.as_filter_by() == {
            "include_labels": {"operate"},
            "exclude_labels": {"finish"},
            "include_names": {"graph_fork"},
        }

    def test_legacy_payload_without_filter_by_defaults_empty(self) -> None:
        info = CompiledNodeInfo.model_validate({"system_prompt": "hello"})

        assert info.filter_by == {}
        assert info.tool_names == ()
        assert info.as_filter_by() == {}

    def test_filter_by_roundtrip_json(self, tmp_path: Path) -> None:
        session = AgenticSession.create_new().with_compiled_nodes(
            {
                "setup": CompiledNodeInfo.from_bind(
                    system_prompt="setup",
                    filter_by={"include_labels": {"setup"}, "exclude_labels": {"finish"}},
                    tool_names=("session_status",),
                )
            }
        )

        save_session(tmp_path, session)
        loaded = load_session(tmp_path)

        assert loaded is not None
        info = loaded.compiled_nodes["setup"]
        assert info.filter_by == {"include_labels": ["setup"], "exclude_labels": ["finish"]}
        assert info.tool_names == ("session_status",)
        assert info.as_filter_by()["exclude_labels"] == {"finish"}
