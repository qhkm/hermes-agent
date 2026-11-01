from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from plugins.jentera import page_scan, private_form

SCAN = {
    "ok": True,
    "host": "mytax.hasil.gov.my",
    "kind": "login",
    "title": "MyTax",
    "boxes": [
        {"words": "ID Type", "kind": "select", "options": ["MyKad", "Passport"], "pick": "#idType"},
        {"words": "Identification No.", "kind": "text", "pick": "#idNo"},
        {"words": "", "kind": "password", "pick": "input[name=pw]"},
    ],
    "submits": ["Next"],
}


class _Runner(BaseHTTPRequestHandler):
    status = 200
    body: dict = SCAN
    delay = 0.0
    calls: list = []

    def log_message(self, *a):
        pass

    def do_GET(self):  # private_form's active check: nothing is live
        self._send(200, {"ok": True, "active": False})

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        _Runner.calls.append((self.path, self.headers.get("Authorization")))
        if _Runner.delay:
            time.sleep(_Runner.delay)
        self._send(_Runner.status, _Runner.body)

    def _send(self, status, body):
        data = json.dumps(body).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except OSError:
            pass


@pytest.fixture()
def runner(monkeypatch):
    _Runner.status, _Runner.body, _Runner.delay, _Runner.calls = 200, SCAN, 0.0, []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Runner)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("JENTERA_RUNNER_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("API_SERVER_KEY", "hermes-key-for-tests")
    page_scan._seen.clear()
    monkeypatch.setitem(page_scan._state, "unavailable_until", 0.0)
    monkeypatch.setitem(private_form._state, "active", None)
    monkeypatch.setitem(private_form._state, "checked_at", 0.0)
    yield _Runner
    server.shutdown()


def _nav(url="https://mytax.hasil.gov.my/"):
    return json.dumps({"success": True, "url": url, "title": "MyTax"})


CLICK = json.dumps({"success": True, "clicked": "@e5"})


def _scans(runner):
    return [c for c in runner.calls if c[0] == "/v1/page/scan"]


def test_navigate_gains_the_form_summary(runner):
    out = page_scan.augment(tool_name="browser_navigate", result=_nav(), task_id="t1")
    parsed = json.loads(out)
    assert parsed["url"] == "https://mytax.hasil.gov.my/", "the original result is kept"
    summary = parsed["page_form"]
    assert "host mytax.hasil.gov.my, kind login" in summary
    assert '- {"label": "ID Type", "kind": "select", "options": ["MyKad", "Passport"], "pick": "#idType"}' in summary
    assert '"pick": "input[name=pw]"' in summary and "give it your own label" in summary
    assert summary.endswith("Buttons: Next")
    assert _scans(runner) == [("/v1/page/scan", "Bearer hermes-key-for-tests")]


def test_once_per_page_url_per_task(runner):
    assert page_scan.augment(tool_name="browser_navigate", result=_nav(), task_id="t1")
    assert page_scan.augment(tool_name="browser_navigate", result=_nav(), task_id="t1") is None
    assert len(_scans(runner)) == 1, "a page already summarised is not scanned again"
    # The same form reached by a click in the same task is not repeated.
    assert page_scan.augment(tool_name="browser_click", result=CLICK, task_id="t1") is None
    # Another task gets it again.
    assert page_scan.augment(tool_name="browser_navigate", result=_nav(), task_id="t2")


def test_a_click_onto_a_new_form_is_summarised_once(runner):
    assert page_scan.augment(tool_name="browser_click", result=CLICK, task_id="t1")
    assert page_scan.augment(tool_name="browser_click", result=CLICK, task_id="t1") is None
    runner.body = {**SCAN, "boxes": [{"words": "TAC", "kind": "password", "pick": "#tac"}]}
    assert "#tac" in json.loads(page_scan.augment(tool_name="browser_click", result=CLICK, task_id="t1"))["page_form"]


@pytest.mark.parametrize("status,body", [
    (404, {"ok": False, "code": "unavailable"}),
    (200, {"ok": False, "code": "no_page"}),
    (200, {"ok": False, "code": "busy"}),
    (200, {"ok": True, "host": "example.com", "kind": "none", "boxes": [], "submits": []}),
    (500, {"ok": False}),
])
def test_nothing_is_added_without_boxes(runner, status, body):
    runner.status, runner.body = status, body
    assert page_scan.augment(tool_name="browser_navigate", result=_nav(), task_id="t1") is None


def test_a_404_is_remembered_for_a_minute(runner):
    runner.status, runner.body = 404, {"ok": False, "code": "unavailable"}
    page_scan.augment(tool_name="browser_navigate", result=_nav("https://a.example/"), task_id="t1")
    page_scan.augment(tool_name="browser_navigate", result=_nav("https://b.example/"), task_id="t1")
    assert len(_scans(runner)) == 1


def test_the_scan_has_three_seconds(runner, monkeypatch):
    monkeypatch.setattr(page_scan, "SCAN_BUDGET_SECONDS", 0.3)
    runner.delay = 1.0
    started = time.monotonic()
    assert page_scan.augment(tool_name="browser_navigate", result=_nav(), task_id="t1") is None
    assert time.monotonic() - started < 0.9


def test_default_budget_is_three_seconds():
    assert page_scan.SCAN_BUDGET_SECONDS == 3.0


def test_other_tools_and_failures_are_left_alone(runner):
    assert page_scan.augment(tool_name="browser_snapshot", result=_nav(), task_id="t1") is None
    failed = json.dumps({"success": False, "error": "Failed to click @e5"})
    assert page_scan.augment(tool_name="browser_click", result=failed, task_id="t1") is None
    assert page_scan.augment(tool_name="browser_navigate", result="not json", task_id="t1") is None
    assert _scans(runner) == []


def test_the_summary_stays_compact():
    boxes = [{"words": f"Box number {i} with some words", "kind": "select",
              "options": [f"Option {j}" for j in range(10)], "pick": f"#box-{i}"} for i in range(40)]
    summary = page_scan.format_summary({"host": "x.example", "kind": "other", "boxes": boxes, "submits": ["Go"]})
    assert len(summary) <= page_scan.SUMMARY_MAX_CHARS
    assert "more boxes: run jentera-page form" in summary
    assert summary.endswith("Buttons: Go")


def test_the_plugin_hook_adds_the_form_and_still_scrubs(runner):
    out = private_form._on_transform_tool_result(
        tool_name="browser_navigate", result=_nav(), task_id="t9", session_id="s")
    assert "page_form" in json.loads(out)
    # A non-browser result with nothing live is left as it is.
    assert private_form._on_transform_tool_result(tool_name="terminal", result="ok", task_id="t9") is None
