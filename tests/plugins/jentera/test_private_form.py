from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import agent.redact as redact
from plugins.jentera import private_form


class _Runner(BaseHTTPRequestHandler):
    values = ["4111 1111 1111 1111", "123"]
    auth = None
    mode = "ok"

    def log_message(self, *a):
        pass

    def do_GET(self):
        _Runner.auth = (self.path, self.headers.get("Authorization"))
        if _Runner.mode == "slow":
            import time
            time.sleep(1.6)
        data = json.dumps({"ok": True, "values": _Runner.values}).encode()
        try:
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except OSError:
            pass


@pytest.fixture()
def runner(monkeypatch):
    _Runner.mode = "ok"
    _Runner.values = ["4111 1111 1111 1111", "123"]
    server = HTTPServer(("127.0.0.1", 0), _Runner)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("JENTERA_RUNNER_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("API_SERVER_KEY", "hermes-key-for-tests")
    redact.clear_vault_redaction_values()
    yield _Runner
    server.shutdown()
    redact.clear_vault_redaction_values()


def test_hook_registers_what_the_runner_returns(runner):
    assert private_form._on_pre_api_request() is None
    assert _Runner.auth == ("/v1/private-form/redactions", "Bearer hermes-key-for-tests")
    assert redact.redact_registered_vault_values("a 4111 1111 1111 1111 b") == "a [hidden] b"


def test_list_is_replaced_not_merged(runner):
    private_form.refresh_values()
    _Runner.values = ["999999"]
    private_form.refresh_values()
    assert redact.redact_registered_vault_values("4111 1111 1111 1111 999999") == "4111 1111 1111 1111 [hidden]"


def test_timeout_keeps_previous_list(runner, monkeypatch):
    private_form.refresh_values()
    monkeypatch.setattr(private_form, "TIMEOUT_SECONDS", 0.2)
    _Runner.mode = "slow"
    _Runner.values = []
    assert private_form.refresh_values() is False
    assert redact.redact_registered_vault_values("4111 1111 1111 1111") == "[hidden]"


def test_unreachable_runner_keeps_previous_list(runner, monkeypatch):
    private_form.refresh_values()
    monkeypatch.setenv("JENTERA_RUNNER_URL", "http://127.0.0.1:1")
    assert private_form.refresh_values() is False
    assert redact.redact_registered_vault_values("123") == "[hidden]"


def test_tool_result_hook_blanks_after_fetching(runner):
    out = private_form._on_transform_tool_result(result="paid with 4111 1111 1111 1111")
    assert out == "paid with [hidden]"
    assert private_form._on_transform_tool_result(result="nothing here") is None
