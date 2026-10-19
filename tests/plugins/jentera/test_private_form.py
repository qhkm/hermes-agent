from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import agent.redact as redact
from plugins.jentera import private_form

CARD = "4111 1111 1111 1111"


class _Runner(BaseHTTPRequestHandler):
    """The runner's two routes: `active` never sees a text, `blank` never returns a value it was not sent."""

    values: list = []
    calls: list = []
    active_mode = "ok"
    blank_mode = "ok"

    def log_message(self, *a):
        pass

    def _send(self, status, body):
        data = json.dumps(body).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except OSError:
            pass

    def do_GET(self):
        _Runner.calls.append(("GET", self.path, self.headers.get("Authorization"), None))
        if self.path != "/v1/private-form/active" or _Runner.active_mode == "fail":
            return self._send(500, {"ok": False})
        self._send(200, {"ok": True, "active": bool(_Runner.values)})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        _Runner.calls.append(("POST", self.path, self.headers.get("Authorization"), len(body.get("texts", []))))
        if self.path != "/v1/private-form/blank" or _Runner.blank_mode == "fail":
            return self._send(500, {"ok": False})
        texts = []
        for text in body["texts"]:
            for value in _Runner.values:
                text = text.replace(value, "[hidden]")
            texts.append(text)
        self._send(200, {"ok": True, "texts": texts})


@pytest.fixture()
def runner(monkeypatch):
    _Runner.values = []
    _Runner.calls = []
    _Runner.active_mode = "ok"
    _Runner.blank_mode = "ok"
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Runner)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("JENTERA_RUNNER_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("API_SERVER_KEY", "hermes-key-for-tests")
    monkeypatch.setitem(private_form._state, "active", None)
    monkeypatch.setitem(private_form._state, "checked_at", 0.0)
    yield _Runner
    server.shutdown()


def _expire_cache():
    private_form._state["checked_at"] = 0.0


def test_nothing_live_passes_through_without_sending_any_text(runner):
    assert private_form.scrub_texts([f"a {CARD}", "b"]) == [f"a {CARD}", "b"]
    assert [c[:3] for c in runner.calls] == [("GET", "/v1/private-form/active", "Bearer hermes-key-for-tests")]


def test_live_values_are_blanked_by_the_runner(runner):
    runner.values = [CARD]
    assert private_form.scrub_texts([f"paid {CARD}", "", "fine"]) == ["paid [hidden]", "", "fine"]
    posts = [c for c in runner.calls if c[0] == "POST"]
    assert posts == [("POST", "/v1/private-form/blank", "Bearer hermes-key-for-tests", 2)], "empty texts are not sent"


def test_the_active_answer_is_cached_for_two_seconds(runner, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(private_form.time, "monotonic", lambda: clock[0])
    private_form.scrub_texts(["x"])
    private_form.scrub_texts(["x"])
    clock[0] += 1.9
    private_form.scrub_texts(["x"])
    assert sum(1 for c in runner.calls if c[0] == "GET") == 1
    clock[0] += 0.2
    private_form.scrub_texts(["x"])
    assert sum(1 for c in runner.calls if c[0] == "GET") == 2


def test_a_failed_blank_while_live_withholds_the_text(runner):
    runner.values = [CARD]
    runner.blank_mode = "fail"
    assert private_form.scrub_texts([f"paid {CARD}", ""]) == [redact.MODEL_TEXT_CHECK_FAILED, ""]


def test_a_failed_check_after_live_withholds_the_text(runner):
    runner.values = [CARD]
    private_form.scrub_texts(["x"])
    _expire_cache()
    runner.active_mode = "fail"
    assert private_form.scrub_texts([f"paid {CARD}"]) == [redact.MODEL_TEXT_CHECK_FAILED]


def test_a_failed_check_with_nothing_live_last_passes_through(runner, monkeypatch):
    runner.active_mode = "fail"
    assert private_form.scrub_texts(["hello"]) == ["hello"], "never known live"
    runner.active_mode = "ok"
    private_form.scrub_texts(["x"])
    _expire_cache()
    runner.active_mode = "fail"
    assert private_form.scrub_texts(["hello"]) == ["hello"], "last known not live"


def test_an_unreachable_runner_after_live_withholds(runner, monkeypatch):
    runner.values = [CARD]
    private_form.scrub_texts(["x"])
    _expire_cache()
    monkeypatch.setenv("JENTERA_RUNNER_URL", "http://127.0.0.1:1")
    assert private_form.scrub_texts([CARD]) == [redact.MODEL_TEXT_CHECK_FAILED]


def test_many_texts_go_in_batches_within_the_runner_limits(runner):
    runner.values = [CARD]
    texts = [f"{i} {CARD}" for i in range(150)]
    assert private_form.scrub_texts(texts) == [f"{i} [hidden]" for i in range(150)]
    sizes = [c[3] for c in runner.calls if c[0] == "POST"]
    assert sizes == [64, 64, 22]


def test_a_long_text_is_sent_in_pieces(runner):
    runner.values = [CARD]
    text = ("x" * (private_form.PIECE_CHARS - 100)) + CARD + ("y" * private_form.PIECE_CHARS)
    out = private_form.scrub_texts([text])[0]
    assert CARD not in out
    assert out == text.replace(CARD, "[hidden]")


def test_a_value_across_a_cut_withholds_that_text(runner):
    runner.values = [CARD]
    text = ("x" * (private_form.PIECE_CHARS - 5)) + CARD + ("y" * 100)
    assert private_form.scrub_texts([text, "ok"]) == [redact.MODEL_TEXT_CHECK_FAILED, "ok"]


def test_the_tool_result_hook_blanks_or_passes(runner):
    assert private_form._on_transform_tool_result(result=f"paid with {CARD}") is None, "nothing live"
    runner.values = [CARD]
    _expire_cache()
    assert private_form._on_transform_tool_result(result=f"paid with {CARD}") == "paid with [hidden]"
    assert private_form._on_transform_tool_result(result="nothing here") is None
    runner.blank_mode = "fail"
    assert private_form._on_transform_tool_result(result="anything") == redact.MODEL_TEXT_CHECK_FAILED


def test_register_installs_the_send_time_scrubber():
    class Ctx:
        hooks = []

        def register_hook(self, name, fn):
            self.hooks.append(name)

    try:
        ctx = Ctx()
        private_form.register_hooks(ctx)
        assert ctx.hooks == ["transform_tool_result"]
        assert redact._MODEL_TEXT_SCRUBBER is private_form.scrub_texts
    finally:
        redact.set_model_text_scrubber(None)
