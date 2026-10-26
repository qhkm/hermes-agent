"""Tests for the fast open path in browser_navigate (script navigation, open as fallback)."""

import json

import pytest

import tools.browser_tool as bt


class FakeBrowser:
    """Answers _run_browser_command calls from a script of page states."""

    def __init__(self, before="https://old.example/", after=None, eval_ok=True, open_result=None):
        self.before = before
        self.after = after
        self.eval_ok = eval_ok
        self.open_result = open_result or {
            "success": True,
            "data": {"url": "https://slow.example/", "title": "Slow"},
        }
        self.calls = []

    def __call__(self, session_key, command, args=None, timeout=None, _engine_override=None):
        args = args or []
        self.calls.append((command, args))
        if command == "eval":
            if not self.eval_ok:
                return {"success": False, "error": "eval failed"}
            expr = args[0]
            if expr.startswith("(function(u)"):
                return {"success": True, "data": {"result": json.dumps(self.before)}}
            if expr.startswith("JSON.stringify"):
                return {"success": True, "data": {"result": json.dumps(json.dumps(self.after))}}
            return {"success": True, "data": {"result": ""}}
        if command == "wait":
            return {"success": True, "data": {}}
        if command == "open":
            return self.open_result
        if command == "snapshot":
            return {"success": True, "data": {"snapshot": "- heading \"Hi\"", "refs": {}}}
        return {"success": True, "data": {}}

    def commands(self):
        return [c for c, _ in self.calls]


@pytest.fixture
def local_browser(monkeypatch):
    monkeypatch.setattr(bt, "_get_cloud_provider", lambda: None)
    monkeypatch.setattr(bt, "_is_local_backend", lambda: True)
    monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(bt, "_get_browser_engine", lambda: "chrome")
    monkeypatch.setattr(bt, "_snapshot_setting", lambda name: "")
    monkeypatch.setattr(bt, "check_website_access", lambda url: None)
    monkeypatch.setattr(bt, "_maybe_start_recording", lambda key: None)
    monkeypatch.setattr(bt, "_get_session_info", lambda key: {"_first_nav": False})


def install(monkeypatch, fake):
    monkeypatch.setattr(bt, "_run_browser_command", fake)
    return fake


class TestFastOpen:
    def test_new_document_skips_open(self, monkeypatch, local_browser):
        fake = install(monkeypatch, FakeBrowser(after={"m": False, "u": "https://new.example/", "t": "New"}))
        out = json.loads(bt.browser_navigate("https://new.example/", task_id="t1"))
        assert out["success"] is True
        assert out["url"] == "https://new.example/"
        assert out["title"] == "New"
        assert "open" not in fake.commands()
        assert fake.commands()[:3] == ["eval", "wait", "eval"]

    def test_url_is_passed_as_json_literal(self, monkeypatch, local_browser):
        fake = install(monkeypatch, FakeBrowser(after={"m": False, "u": "https://new.example/", "t": ""}))
        bt.browser_navigate("https://new.example/?q=a'b", task_id="t1")
        first = fake.calls[0][1][0]
        assert json.dumps("https://new.example/?q=a'b") in first

    def test_marker_still_present_falls_back_to_open(self, monkeypatch, local_browser):
        fake = install(monkeypatch, FakeBrowser(after={"m": True, "u": "https://old.example/", "t": "Old"}))
        out = json.loads(bt.browser_navigate("https://slow.example/", task_id="t1"))
        assert "open" in fake.commands()
        assert out["url"] == "https://slow.example/"

    def test_fragment_only_change_falls_back_to_open(self, monkeypatch, local_browser):
        fake = install(monkeypatch, FakeBrowser(before="https://slow.example/#a"))
        bt.browser_navigate("https://slow.example/#b", task_id="t1")
        assert "wait" not in fake.commands()
        assert "open" in fake.commands()

    def test_chrome_error_page_falls_back_to_open(self, monkeypatch, local_browser):
        fake = install(
            monkeypatch,
            FakeBrowser(
                after={"m": False, "u": "chrome-error://chromewebdata/", "t": "x"},
                open_result={"success": False, "error": "Navigation failed: net::ERR_CONNECTION_REFUSED"},
            ),
        )
        out = json.loads(bt.browser_navigate("https://down.example/", task_id="t1"))
        assert "open" in fake.commands()
        assert out["success"] is False
        assert "ERR_CONNECTION_REFUSED" in out["error"]

    def test_eval_failure_falls_back_to_open(self, monkeypatch, local_browser):
        fake = install(monkeypatch, FakeBrowser(eval_ok=False))
        out = json.loads(bt.browser_navigate("https://slow.example/", task_id="t1"))
        assert "open" in fake.commands()
        assert out["success"] is True

    def test_switch_off_uses_open(self, monkeypatch, local_browser):
        monkeypatch.setattr(bt, "_snapshot_setting", lambda name: "off" if name == "JENTERA_BROWSER_FAST_OPEN" else "")
        fake = install(monkeypatch, FakeBrowser(after={"m": False, "u": "https://new.example/", "t": ""}))
        bt.browser_navigate("https://slow.example/", task_id="t1")
        assert fake.commands()[0] == "open"

    def test_cloud_provider_uses_open(self, monkeypatch, local_browser):
        monkeypatch.setattr(bt, "_get_cloud_provider", lambda: object())
        monkeypatch.setattr(bt, "_is_local_backend", lambda: False)
        monkeypatch.setattr(bt, "_allow_private_urls", lambda: True)
        monkeypatch.setattr(bt, "_navigation_session_key", lambda task_id, url: task_id)
        fake = install(monkeypatch, FakeBrowser(after={"m": False, "u": "https://new.example/", "t": ""}))
        bt.browser_navigate("https://slow.example/", task_id="t1")
        assert fake.commands()[0] == "open"

    def test_redirect_to_metadata_is_still_blocked(self, monkeypatch, local_browser):
        fake = install(
            monkeypatch,
            FakeBrowser(after={"m": False, "u": "http://169.254.169.254/latest/meta-data/", "t": ""}),
        )
        out = json.loads(bt.browser_navigate("https://redirector.example/", task_id="t1"))
        assert out["success"] is False
        assert "metadata" in out["error"]
        assert ("open", ["about:blank"]) in fake.calls
