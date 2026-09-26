from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from plugins.jentera import handoff

RUN_ID = "run_" + "a" * 32


class _Runner(BaseHTTPRequestHandler):
    received: list = []
    reply = (200, {})
    available = True

    def log_message(self, *args):
        pass

    def _send(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        _Runner.received.append(("GET", self.path, self.headers.get("Authorization"), None))
        self._send(200, {"ok": True, "available": _Runner.available})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        _Runner.received.append(("POST", self.path, self.headers.get("Authorization"), body))
        self._send(*_Runner.reply)


def _start_runner_server(monkeypatch):
    _Runner.received = []
    _Runner.available = True
    _Runner.reply = (200, {"ok": True, "specialist": "records", "name": "Finance and records",
                           "answer": "3 invoices are unpaid."})
    server = HTTPServer(("127.0.0.1", 0), _Runner)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("JENTERA_RUNNER_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("API_SERVER_KEY", "hermes-key-for-tests")
    return server


@pytest.fixture()
def runner(monkeypatch):
    server = _start_runner_server(monkeypatch)
    monkeypatch.setattr(handoff, "_calling_run_id", lambda: RUN_ID)
    yield _Runner
    server.shutdown()


@pytest.fixture()
def runner_without_run_patch(monkeypatch):
    # Same live fake runner as `runner`, but leaves the real _calling_run_id
    # in place so it can be exercised directly (unbound / bound cases).
    server = _start_runner_server(monkeypatch)
    yield _Runner
    server.shutdown()


def test_hands_off_and_returns_the_specialists_answer(runner):
    out = json.loads(handoff.handle_ask_specialist({"specialist": "records", "brief": "Which invoices are unpaid?"}))
    assert out == {"specialist": "records", "name": "Finance and records", "answer": "3 invoices are unpaid."}
    method, path, auth, body = runner.received[-1]
    assert (method, path, auth) == ("POST", "/v1/handoff", "Bearer hermes-key-for-tests")
    assert body == {"runId": RUN_ID, "specialist": "records", "brief": "Which invoices are unpaid?"}


def test_a_refusal_carries_its_code_and_asks_for_honesty(runner):
    runner.reply = (200, {"ok": False, "code": "limit_count", "message": "This task has already used its five hand-offs."})
    out = json.loads(handoff.handle_ask_specialist({"specialist": "records", "brief": "More"}))
    assert out["code"] == "limit_count"
    assert "which part is missing" in out["error"]


def test_an_unreachable_runner_is_a_failure_not_an_exception(runner, monkeypatch):
    monkeypatch.setenv("JENTERA_RUNNER_URL", "http://127.0.0.1:9")
    out = json.loads(handoff.handle_ask_specialist({"specialist": "records", "brief": "x"}))
    assert out["code"] == "failed"


def test_a_long_or_empty_brief_is_refused_before_anything_is_sent(runner):
    overlong = json.loads(handoff.handle_ask_specialist({"specialist": "records", "brief": "x" * 2_001}))
    assert overlong["code"] == "invalid"
    assert "which part is missing" in overlong["error"]
    empty = json.loads(handoff.handle_ask_specialist({"specialist": "records", "brief": "  "}))
    assert empty["code"] == "invalid"
    assert "which part is missing" in empty["error"]
    assert runner.received == []


def test_refuses_without_a_bound_run_id(runner_without_run_patch):
    # `_calling_run_id` is left un-monkeypatched here, so this exercises the
    # real no-run guard: outside a bound API-server run, it must come back
    # empty rather than the literal "default" get_current_session_key would
    # otherwise hand back, which would be truthy and defeat the guard.
    out = json.loads(handoff.handle_ask_specialist({"specialist": "records", "brief": "Which invoices are unpaid?"}))
    assert out["code"] == "unavailable"
    assert "which part is missing" in out["error"]
    assert runner_without_run_patch.received == []


def test_calling_run_id_returns_the_bound_session_key():
    import contextvars

    from tools.approval import set_current_session_key

    def _read_inside_a_bound_run():
        set_current_session_key(RUN_ID)
        return handoff._calling_run_id()

    # Runs in a copied context so the binding cannot leak into other tests.
    assert contextvars.copy_context().run(_read_inside_a_bound_run) == RUN_ID


def test_offered_only_when_the_runner_says_hand_offs_are_on(runner, monkeypatch):
    assert handoff.check_handoff_available() is True
    runner.available = False
    assert handoff.check_handoff_available() is False
    monkeypatch.delenv("API_SERVER_KEY")
    assert handoff.check_handoff_available() is False


def test_registered_beside_business_records():
    import plugins.jentera as plugin

    registered = []

    class Ctx:
        def register_tool(self, **kwargs):
            registered.append(kwargs["name"])

    plugin.register(Ctx())
    assert registered == ["business_records", "connect_service", "ask_specialist"]
