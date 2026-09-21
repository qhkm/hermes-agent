"""A browser whose owner has taken it back refuses tool calls immediately.

The failure this prevents is a silent one: the agent reaches for a browser a
person is sitting in front of, the call does not come back, and the task is
killed by its own deadline minutes later with nothing to show. Refusing costs
one tool result the model can act on.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tools import browser_tool  # noqa: E402


def _configure(monkeypatch, hold_file):
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {"browser": {"hold_file": str(hold_file)} if hold_file is not None else {}},
    )


def _unreachable(*args, **kwargs):
    raise AssertionError("a held browser must not be reached at all")


class TestBrowserOwnerHold:
    def test_held_browser_refuses_before_anything_is_spawned(self, tmp_path, monkeypatch):
        hold_file = tmp_path / "browser-control.json"
        hold_file.write_text(json.dumps({"paused": True}), encoding="utf-8")
        _configure(monkeypatch, hold_file)
        monkeypatch.setattr(browser_tool, "_find_agent_browser", _unreachable)
        monkeypatch.setattr(browser_tool.subprocess, "Popen", _unreachable)

        result = browser_tool._run_browser_command("task-1", "console", [])

        assert result["success"] is False
        assert "owner" in result["error"]

    def test_released_hold_lets_the_command_run(self, tmp_path, monkeypatch):
        hold_file = tmp_path / "browser-control.json"
        hold_file.write_text(json.dumps({"paused": False}), encoding="utf-8")
        _configure(monkeypatch, hold_file)

        def missing_cli(*args, **kwargs):
            raise FileNotFoundError("agent-browser is not installed here")

        monkeypatch.setattr(browser_tool, "_find_agent_browser", missing_cli)

        result = browser_tool._run_browser_command("task-1", "console", [])

        # It got past the hold and failed further in, on its own terms.
        assert result["error"] == "agent-browser is not installed here"

    @pytest.mark.parametrize("state", ['{"paused": "yes"}', "not json at all", "[]", ""])
    def test_a_hold_file_it_cannot_read_holds_nothing(self, tmp_path, monkeypatch, state):
        hold_file = tmp_path / "browser-control.json"
        hold_file.write_text(state, encoding="utf-8")
        _configure(monkeypatch, hold_file)

        assert browser_tool._browser_hold_error() is None

    def test_unconfigured_and_absent_files_hold_nothing(self, tmp_path, monkeypatch):
        _configure(monkeypatch, None)
        assert browser_tool._browser_hold_error() is None

        _configure(monkeypatch, tmp_path / "never-written.json")
        assert browser_tool._browser_hold_error() is None

    def test_unreadable_config_holds_nothing(self, monkeypatch):
        def broken():
            raise RuntimeError("config is unavailable")

        monkeypatch.setattr("hermes_cli.config.read_raw_config", broken)

        assert browser_tool._browser_hold_error() is None

    def test_raw_cdp_is_refused_by_the_same_hold(self, tmp_path, monkeypatch):
        from tools.browser_cdp_tool import browser_cdp

        hold_file = tmp_path / "browser-control.json"
        hold_file.write_text(json.dumps({"paused": True}), encoding="utf-8")
        _configure(monkeypatch, hold_file)

        result = json.loads(browser_cdp(method="Target.getTargets"))

        assert "owner" in result["error"]
