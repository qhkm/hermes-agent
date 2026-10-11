"""The website tools send the agent's edits to Jentera and pass refusals back in words."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from plugins.jentera import site


class _Recorder(BaseHTTPRequestHandler):
    received: list[dict] = []
    reply: tuple[int, dict] = (200, {"ok": True})

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        type(self).received.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": json.loads(self.rfile.read(length) or b"{}")})
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
    server = HTTPServer(("127.0.0.1", 0), _Recorder)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("JENTERA_API_BASE", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-jentera-v1.test-credential")
    yield _Recorder
    server.shutdown()


def test_draft_posts_the_op_with_the_runtime_credential(control_plane):
    control_plane.reply = (200, {"ok": True, "created": True, "address": "kedai", "document": {"v": 1}, "guide": "hero — …"})
    out = json.loads(site.handle_site_draft({}))
    assert control_plane.received[0]["path"] == "/v1/runtime/site"
    assert control_plane.received[0]["body"] == {"op": "draft"}
    assert control_plane.received[0]["auth"] == "Bearer sk-jentera-v1.test-credential"
    assert out["address"] == "kedai"


def test_apply_sends_ops_and_returns_problems_on_refusal(control_plane):
    control_plane.reply = (422, {"ok": False, "problems": ['section "hero" (hero) heading is empty']})
    out = json.loads(site.handle_site_apply({"ops": [{"op": "remove_section", "id": "hero"}], "note": "x"}))
    assert control_plane.received[0]["body"] == {"op": "apply", "ops": [{"op": "remove_section", "id": "hero"}], "note": "x"}
    assert "heading is empty" in json.dumps(out)


def test_apply_refuses_a_non_list_before_calling(control_plane):
    out = json.loads(site.handle_site_apply({"ops": "make it pink"}))
    assert control_plane.received == []
    assert "list" in json.dumps(out)


def test_image_needs_a_licence(control_plane):
    out = json.loads(site.handle_site_image({"url": "https://example.com/a.png"}))
    assert control_plane.received == []
    assert "licence" in json.dumps(out)


def test_unavailable_is_passed_on_in_words(control_plane):
    control_plane.reply = (404, {"ok": False, "code": "unavailable", "err": "Websites are not turned on for this business."})
    out = json.loads(site.handle_site_get({}))
    assert "not turned on" in json.dumps(out)
