"""The website tools send the agent's edits to Jentera and pass refusals back in words."""

from __future__ import annotations

import json
import socket
import threading
import time
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
    site._available_cache = None
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


def test_offered_only_where_the_control_plane_says_websites_are_on(control_plane):
    control_plane.reply = (200, {"ok": True, "available": True})
    assert site.site_available() is True
    assert control_plane.received[0]["body"] == {"op": "available"}
    site._available_cache = None
    control_plane.reply = (200, {"ok": True, "available": False})
    assert site.site_available() is False


def test_availability_is_cached_for_a_minute(control_plane, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(site.time, "monotonic", lambda: clock[0])
    control_plane.reply = (200, {"ok": True, "available": True})
    assert site.site_available() is True
    clock[0] += 30
    assert site.site_available() is True
    assert len(control_plane.received) == 1
    clock[0] += 31
    control_plane.reply = (200, {"ok": True, "available": False})
    assert site.site_available() is False
    assert len(control_plane.received) == 2


def test_availability_keeps_the_last_answer_when_unreachable(control_plane, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(site.time, "monotonic", lambda: clock[0])
    control_plane.reply = (200, {"ok": True, "available": True})
    assert site.site_available() is True
    monkeypatch.setenv("JENTERA_API_BASE", "http://127.0.0.1:9")
    clock[0] += 120
    assert site.site_available() is True
    site._available_cache = None
    assert site.site_available() is False


def test_draft_refusal_is_worded_as_a_failed_first_draft(control_plane):
    control_plane.reply = (422, {"ok": False, "problems": ["no name"]})
    out = json.dumps(json.loads(site.handle_site_draft({})))
    assert "first draft" in out and "no name" in out


@pytest.fixture()
def hanging_control_plane(monkeypatch):
    """Accepts connections and never answers, like a wedged upstream."""
    site._available_cache = None
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(16)
    accepted: list[socket.socket] = []

    def accept():
        while True:
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            accepted.append(conn)

    threading.Thread(target=accept, daemon=True).start()
    monkeypatch.setenv("JENTERA_API_BASE", f"http://127.0.0.1:{listener.getsockname()[1]}")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-jentera-v1.test-credential")
    yield accepted
    listener.close()
    for conn in accepted:
        conn.close()


def test_the_availability_probe_is_short_like_the_hand_off_probe():
    assert site.AVAILABLE_TIMEOUT_SECONDS <= 3


def test_a_hanging_control_plane_hides_the_tools_quickly_and_is_not_asked_again(hanging_control_plane, monkeypatch):
    monkeypatch.setattr(site, "AVAILABLE_TIMEOUT_SECONDS", 0.3)
    started = time.monotonic()
    assert site.site_available() is False
    assert time.monotonic() - started < 2
    assert site.site_available() is False
    assert len(hanging_control_plane) == 1


def test_an_unreachable_control_plane_is_not_asked_again_within_the_minute(control_plane, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(site.time, "monotonic", lambda: clock[0])
    calls = []
    real_post = site._post
    monkeypatch.setattr(site, "_post", lambda *a, **kw: calls.append(1) or real_post(*a, **kw))
    monkeypatch.setenv("JENTERA_API_BASE", "http://127.0.0.1:9")
    assert site.site_available() is False
    clock[0] += 30
    assert site.site_available() is False
    assert len(calls) == 1
    clock[0] += 31
    assert site.site_available() is False
    assert len(calls) == 2
