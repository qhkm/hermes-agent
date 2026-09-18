"""The Jentera plugin reads records it holds no credential for."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from plugins.jentera import tools


class _Recorder(BaseHTTPRequestHandler):
    """Stands in for the control plane; records what the plugin sent."""

    received: list[dict] = []
    reply: tuple[int, dict] = (200, {"ok": True, "detail": "IV-00231 · Alex Wong · MYR 2000.00"})

    def do_POST(self):  # noqa: N802 - http.server's spelling
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).received.append({
            "path": self.path,
            "auth": self.headers.get("Authorization"),
            "body": body,
        })
        status, payload = type(self).reply
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


@pytest.fixture()
def control_plane(monkeypatch):
    _Recorder.received = []
    _Recorder.reply = (200, {"ok": True, "detail": "IV-00231 · Alex Wong · MYR 2000.00"})
    server = HTTPServer(("127.0.0.1", 0), _Recorder)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("JENTERA_API_BASE", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-jentera-v1.test-credential")
    yield _Recorder
    server.shutdown()


def test_reads_overdue_invoices(control_plane):
    out = json.loads(tools.handle_business_records(
        {"resource": "invoices", "payment_status": "overdue"}))
    assert "IV-00231" in out["summary"]
    sent = control_plane.received[0]
    assert sent["path"] == "/v1/runtime/connector"
    # The credential travels, and it is the one already on the sprite for
    # the model channel — nothing new is transferred to make this work.
    assert sent["auth"] == "Bearer sk-jentera-v1.test-credential"
    assert sent["body"]["op"] == "list"
    assert sent["body"]["args"] == {"resource": "invoices", "paymentStatus": "OVERDUE"}


def test_does_not_let_the_model_choose_the_operation(control_plane):
    """A read tool that could be talked into writing is not a read tool."""
    tools.handle_business_records({"resource": "contacts", "op": "send", "connector": "WhatsApp"})
    sent = control_plane.received[0]
    assert sent["body"]["op"] == "list"
    assert sent["body"]["connector"] == "Bukku"


def test_refusal_that_needs_the_owner_says_so(control_plane):
    control_plane.reply = (403, {"ok": False, "code": "needs_approval", "err": "send needs the owner's approval"})
    out = json.loads(tools.handle_business_records({"resource": "invoices"}))
    assert out["needs_owner_approval"] is True
    assert "approval" in out["error"]


def test_unreachable_control_plane_is_a_message_not_a_crash(monkeypatch):
    monkeypatch.setenv("JENTERA_API_BASE", "http://127.0.0.1:9")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-jentera-v1.test-credential")
    out = json.loads(tools.handle_business_records({"resource": "invoices"}))
    assert "error" in out


def test_unavailable_without_a_credential(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("JENTERA_API_BASE", raising=False)
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://api.jentera.ai/v1/model")
    assert tools.check_available() is False
    assert "error" in json.loads(tools.handle_business_records({"resource": "invoices"}))


def test_api_base_comes_from_the_model_channel(monkeypatch):
    """No new bootstrap field: the origin is derived from what is there."""
    monkeypatch.delenv("JENTERA_API_BASE", raising=False)
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://api.jentera.ai/v1/model")
    assert tools._api_base() == "https://api.jentera.ai"
    # A plaintext or malformed base is refused rather than used.
    monkeypatch.setenv("OPENROUTER_BASE_URL", "http://api.jentera.ai/v1/model")
    assert tools._api_base() == ""
