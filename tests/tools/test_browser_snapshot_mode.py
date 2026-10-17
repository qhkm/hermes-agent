"""JENTERA_BROWSER_SNAPSHOT: interactive-only, capped, truncated snapshots."""

import json
from unittest.mock import patch

import pytest

import tools.browser_tool as bt


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("JENTERA_BROWSER_SNAPSHOT", raising=False)
    monkeypatch.delenv("JENTERA_BROWSER_SNAPSHOT_MAX", raising=False)


def _snap(text):
    return {"success": True, "data": {"snapshot": text, "refs": {"e1": {}}}}


def _big(n=20000):
    return "\n".join(f'- button "Item {i}" [ref=e{i}]' for i in range(n // 20))


class TestHelpers:
    def test_mode_default_and_interactive(self, monkeypatch):
        assert bt._snapshot_mode() == "default"
        monkeypatch.setenv("JENTERA_BROWSER_SNAPSHOT", "interactive")
        assert bt._snapshot_mode() == "interactive"
        monkeypatch.setenv("JENTERA_BROWSER_SNAPSHOT", "other")
        assert bt._snapshot_mode() == "default"

    def test_cap_defaults(self, monkeypatch):
        assert bt._snapshot_cap() == bt.SNAPSHOT_SUMMARIZE_THRESHOLD
        monkeypatch.setenv("JENTERA_BROWSER_SNAPSHOT", "interactive")
        assert bt._snapshot_cap() == 6000

    def test_cap_env_and_fallbacks(self, monkeypatch):
        monkeypatch.setenv("JENTERA_BROWSER_SNAPSHOT", "interactive")
        monkeypatch.setenv("JENTERA_BROWSER_SNAPSHOT_MAX", "2500")
        assert bt._snapshot_cap() == 2500
        monkeypatch.setenv("JENTERA_BROWSER_SNAPSHOT_MAX", "abc")
        assert bt._snapshot_cap() == 6000
        monkeypatch.setenv("JENTERA_BROWSER_SNAPSHOT_MAX", "999999")
        assert bt._snapshot_cap() == 15000
        monkeypatch.setenv("JENTERA_BROWSER_SNAPSHOT_MAX", "5")
        assert bt._snapshot_cap() == 1000


class TestBrowserSnapshot:
    def _run(self, text, full=False, user_task="do it"):
        calls = []

        def fake(task, cmd, args=None, **kw):
            calls.append((cmd, list(args or [])))
            return _snap(text)

        with patch.object(bt, "_run_browser_command", fake), \
             patch.object(bt, "_is_camofox_mode", return_value=False), \
             patch.object(bt, "_is_local_backend", return_value=True), \
             patch.object(bt, "_extract_relevant_content", return_value="SUMMARY") as summ:
            out = json.loads(bt.browser_snapshot(full=full, task_id="t", user_task=user_task))
        return calls, out, summ

    def test_default_uses_summariser(self):
        calls, out, summ = self._run(_big())
        assert calls[0] == ("snapshot", ["-c"])
        summ.assert_called_once()
        assert out["snapshot"] == "SUMMARY"

    def test_interactive_args_and_truncate(self, monkeypatch):
        monkeypatch.setenv("JENTERA_BROWSER_SNAPSHOT", "interactive")
        with patch.object(bt, "_truncate_snapshot", wraps=bt._truncate_snapshot) as trunc:
            calls, out, summ = self._run(_big())
        assert calls[0] == ("snapshot", ["-i", "-c"])
        summ.assert_not_called()
        trunc.assert_called_once()
        assert trunc.call_args.kwargs["max_chars"] == 6000
        assert len(out["snapshot"]) <= 6000
        assert "truncated" in out["snapshot"]

    def test_interactive_saved_file_note(self, monkeypatch):
        monkeypatch.setenv("JENTERA_BROWSER_SNAPSHOT", "interactive")
        with patch.object(bt, "_store_full_snapshot", return_value="/tmp/snap.txt"):
            _, out, _ = self._run(_big())
        assert 'read_file path="/tmp/snap.txt"' in out["snapshot"]

    def test_full_args_unchanged(self, monkeypatch):
        monkeypatch.setenv("JENTERA_BROWSER_SNAPSHOT", "interactive")
        calls, _, _ = self._run("small", full=True)
        assert calls[0] == ("snapshot", [])


class TestNavigate:
    def _nav(self, text):
        calls = []

        def fake(task, cmd, args=None, **kw):
            calls.append((cmd, list(args or [])))
            if cmd == "open":
                return {"success": True, "data": {"title": "T", "url": "https://example.com"}}
            return _snap(text)

        with patch.object(bt, "_run_browser_command", fake), \
             patch.object(bt, "_is_camofox_mode", return_value=False), \
             patch.object(bt, "_is_local_backend", return_value=True), \
             patch.object(bt, "_get_session_info", return_value={"_first_nav": False}), \
             patch.object(bt, "_maybe_start_recording"), \
             patch.object(bt, "check_website_access", return_value=None):
            out = json.loads(bt.browser_navigate("https://example.com", task_id="t"))
        return calls, out

    def test_default_args(self):
        calls, out = self._nav("small")
        assert ("snapshot", ["-c"]) in calls

    def test_interactive_args_and_cap(self, monkeypatch):
        monkeypatch.setenv("JENTERA_BROWSER_SNAPSHOT", "interactive")
        monkeypatch.setenv("JENTERA_BROWSER_SNAPSHOT_MAX", "2500")
        calls, out = self._nav(_big())
        assert ("snapshot", ["-i", "-c"]) in calls
        assert len(out["snapshot"]) <= 2500
