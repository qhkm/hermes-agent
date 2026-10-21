"""``turn_notes`` on POST /v1/runs, and the warm endpoint.

Jentera's per-message context used to ride in ``instructions``, which lands on
the system message, so every turn's system message differed and the provider's
prompt cache missed the whole history. ``turn_notes`` is injected into the
CURRENT turn's user message at call time instead: it is on every model call of
the turn, never persisted, and absent from earlier turns when they are
replayed.

These run the real adapter and a real ``AIAgent`` against an in-process mock
provider, so what is asserted is the bytes that go on the wire and what lands
in the session store.
"""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    cors_middleware,
    security_headers_middleware,
)
from hermes_state import SessionDB

OPEN = "<jentera-context>\n"
CLOSE = "\n</jentera-context>\n\n"


# ---------------------------------------------------------------------------
# In-process mock provider
# ---------------------------------------------------------------------------


class _Provider(BaseHTTPRequestHandler):
    requests: list = []
    queue: list = []

    def do_POST(self):  # noqa: N802 (http.server API)
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length).decode())
        type(self).requests.append(req)
        # Only chat-completions bodies consume a scripted reply; the agent
        # also probes the endpoint (context length) with other bodies.
        scripted = "messages" in req and type(self).queue
        resp = type(self).queue.pop(0) if scripted else _text("ok")
        msg = resp["choices"][0]["message"]
        if req.get("stream") is True:
            content = msg.get("content") or ""
            calls = msg.get("tool_calls")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunks = [{"id": "m", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}]
            if content:
                chunks.append({"id": "m", "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]})
            for i, call in enumerate(calls or []):
                chunks.append({"id": "m", "choices": [{"index": 0, "delta": {"tool_calls": [{
                    "index": i, "id": call["id"], "type": "function",
                    "function": {"name": call["function"]["name"], "arguments": call["function"]["arguments"]}}]},
                    "finish_reason": None}]})
            chunks.append({"id": "m", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if calls else "stop"}]})
            for chunk in chunks:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        body = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args, **kwargs):
        pass


def _text(text: str) -> dict:
    return {
        "id": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
    }


def _tool_call(name: str, args: str) -> dict:
    return {
        "id": "m",
        "choices": [{"index": 0, "message": {
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": name, "arguments": args}}]},
            "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
    }


def _chat(provider) -> list:
    """Chat-completions bodies only (the context-length probe also lands here)."""
    return [r for r in provider.requests if "messages" in r]


def _system(req: dict) -> str:
    return next(m["content"] for m in req["messages"] if m["role"] == "system")


def _users(req: dict) -> list:
    return [m for m in req["messages"] if m["role"] == "user"]


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


@pytest.fixture()
def provider():
    _Provider.requests = []
    _Provider.queue = []
    srv = HTTPServer(("127.0.0.1", 0), _Provider)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    _Provider.url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    try:
        yield _Provider
    finally:
        srv.shutdown()


def _wire_adapter(monkeypatch, tmp_path, provider, *, toolsets=(), max_iterations=90, api_key=""):
    """A real adapter whose ``_create_agent`` builds a real AIAgent on the mock."""
    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs",
        lambda: {"api_key": "test-key", "base_url": provider.url, "provider": "openai-compat"},
    )
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "test-model")
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})
    monkeypatch.setattr("gateway.run._current_max_iterations", lambda: max_iterations)
    monkeypatch.setattr("gateway.run.GatewayRunner._load_reasoning_config", staticmethod(lambda: {}))
    monkeypatch.setattr("gateway.run.GatewayRunner._load_fallback_model", staticmethod(lambda: None))
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *_: set(toolsets))

    extra = {"key": api_key} if api_key else {}
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra=extra))
    db = SessionDB(db_path=tmp_path / "state.db")
    adapter._session_db = db
    return adapter, db


def _app(adapter) -> web.Application:
    app = web.Application(middlewares=[mw for mw in (cors_middleware, security_headers_middleware) if mw is not None])
    app["api_server_adapter"] = adapter
    for method, path, handler in adapter._http_route_table():
        app.router.add_route(method, path, handler)
    return app


async def _run(cli, **body) -> dict:
    """Start a run, wait for it to settle, return its final status."""
    resp = await cli.post("/v1/runs", json=body)
    assert resp.status == 202, await resp.text()
    run_id = (await resp.json())["run_id"]
    for _ in range(400):
        status = await (await cli.get(f"/v1/runs/{run_id}")).json()
        if status.get("status") in ("completed", "failed", "cancelled"):
            return status
        await asyncio.sleep(0.05)
    raise AssertionError("run did not settle")


# ---------------------------------------------------------------------------
# turn_notes on /v1/runs
# ---------------------------------------------------------------------------


class TestTurnNotes:
    @pytest.mark.asyncio
    async def test_notes_ride_on_the_current_user_message_and_leave_the_system_message_alone(
        self, monkeypatch, tmp_path, provider
    ):
        adapter, db = _wire_adapter(monkeypatch, tmp_path, provider)
        try:
            async with TestClient(TestServer(_app(adapter))) as cli:
                a = await _run(cli, input="hello", instructions="RULES", turn_notes="N1", session_id="sess-a")
                b = await _run(cli, input="hello", instructions="RULES", session_id="sess-b")
            assert a["status"] == "completed" and b["status"] == "completed"

            with_notes, without = _chat(provider)[0], _chat(provider)[1]

            # The system message is exactly what the same run sends without notes.
            assert _system(with_notes) == _system(without)
            assert "RULES" in _system(with_notes)
            assert "N1" not in _system(with_notes)

            # The notes are the first thing on the current user message.
            sent = _users(with_notes)[-1]["content"]
            assert sent == f"{OPEN}N1{CLOSE}hello"
            # And a run without notes is untouched.
            assert _users(without)[-1]["content"] == "hello"
        finally:
            db.close()

    @pytest.mark.asyncio
    async def test_every_request_of_the_turn_carries_the_notes_including_the_iteration_limit_summary(
        self, monkeypatch, tmp_path, provider
    ):
        # One iteration and a tool call: the turn ends by asking the model for
        # a toolless summary, which is a second, hand-built request.
        adapter, db = _wire_adapter(monkeypatch, tmp_path, provider, toolsets={"file"}, max_iterations=1)
        provider.queue = [
            _tool_call("read_file", json.dumps({"path": "/nonexistent-turn-notes-path"})),
            _text("summary"),
        ]
        try:
            async with TestClient(TestServer(_app(adapter))) as cli:
                status = await _run(cli, input="hello", instructions="RULES", turn_notes="N1", session_id="sess-s")
            assert status["status"] == "completed"
            assert status["output"].startswith("summary")

            reqs = _chat(provider)
            assert len(reqs) == 2, [r["messages"][-1] for r in reqs]
            for req in reqs:
                assert _users(req)[0]["content"] == f"{OPEN}N1{CLOSE}hello"
                assert "N1" not in _system(req)
            # The summary request is the second one: its own prompt comes last
            # and carries no notes.
            assert "maximum number of tool-calling iterations" in _users(reqs[1])[-1]["content"]
            assert "jentera-context" not in _users(reqs[1])[-1]["content"]
        finally:
            db.close()

    @pytest.mark.asyncio
    async def test_next_turn_replays_the_previous_user_message_without_its_notes_and_nothing_is_stored(
        self, monkeypatch, tmp_path, provider
    ):
        adapter, db = _wire_adapter(monkeypatch, tmp_path, provider)
        try:
            async with TestClient(TestServer(_app(adapter))) as cli:
                await _run(cli, input="first", instructions="RULES", turn_notes="N1", session_id="sess-c")
                await _run(cli, input="second", instructions="RULES", turn_notes="N2", session_id="sess-c")

            one, two = _chat(provider)[0], _chat(provider)[1]
            users = _users(two)
            assert [u["content"] for u in users] == ["first", f"{OPEN}N2{CLOSE}second"]
            assert "jentera-context" not in json.dumps(users[0])
            assert "N1" not in json.dumps(two)

            # The system message is byte-stable from one turn to the next.
            assert _system(one) == _system(two)

            # Nothing the notes touched reached the store, in either column.
            rows = db.get_messages("sess-c")
            assert rows
            for row in rows:
                assert "jentera-context" not in (row.get("content") or "")
                assert "jentera-context" not in (row.get("api_content") or "")
                assert "N1" not in (row.get("content") or "") + (row.get("api_content") or "")
                assert "N2" not in (row.get("content") or "") + (row.get("api_content") or "")
        finally:
            db.close()


class TestTurnNotesLimits:
    @pytest.mark.asyncio
    async def test_over_the_limit_is_a_400_and_starts_no_run(self, monkeypatch, tmp_path, provider):
        adapter, db = _wire_adapter(monkeypatch, tmp_path, provider)
        try:
            async with TestClient(TestServer(_app(adapter))) as cli:
                resp = await cli.post("/v1/runs", json={"input": "hi", "turn_notes": "x" * 20_001})
                assert resp.status == 400
                assert "turn_notes" in (await resp.json())["error"]["message"]
            assert adapter._run_streams == {}
            assert provider.requests == []
        finally:
            db.close()

    @pytest.mark.asyncio
    async def test_exactly_the_limit_is_accepted(self, monkeypatch, tmp_path, provider):
        adapter, db = _wire_adapter(monkeypatch, tmp_path, provider)
        try:
            async with TestClient(TestServer(_app(adapter))) as cli:
                status = await _run(cli, input="hi", turn_notes="x" * 20_000)
            assert status["status"] == "completed"
        finally:
            db.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad", [123, ["N1"], {"a": 1}, True])
    async def test_non_string_notes_are_a_400(self, monkeypatch, tmp_path, provider, bad):
        adapter, db = _wire_adapter(monkeypatch, tmp_path, provider)
        try:
            async with TestClient(TestServer(_app(adapter))) as cli:
                resp = await cli.post("/v1/runs", json={"input": "hi", "turn_notes": bad})
                assert resp.status == 400
        finally:
            db.close()

    @pytest.mark.asyncio
    async def test_empty_or_null_notes_change_nothing(self, monkeypatch, tmp_path, provider):
        adapter, db = _wire_adapter(monkeypatch, tmp_path, provider)
        try:
            async with TestClient(TestServer(_app(adapter))) as cli:
                await _run(cli, input="hi", turn_notes="", session_id="sess-e1")
                await _run(cli, input="hi", turn_notes=None, session_id="sess-e2")
            for req in _chat(provider)[:2]:
                assert _users(req)[-1]["content"] == "hi"
        finally:
            db.close()


# ---------------------------------------------------------------------------
# /health and /v1/jentera/warm
# ---------------------------------------------------------------------------


class TestHealthAndWarm:
    @pytest.mark.asyncio
    async def test_health_lists_both_features(self, monkeypatch, tmp_path, provider):
        adapter, db = _wire_adapter(monkeypatch, tmp_path, provider)
        try:
            async with TestClient(TestServer(_app(adapter))) as cli:
                for path in ("/health", "/v1/health"):
                    data = await (await cli.get(path)).json()
                    assert data["status"] == "ok"
                    assert data["jentera_features"] == ["turn_notes", "warm"]
        finally:
            db.close()

    @pytest.mark.asyncio
    async def test_warm_builds_the_tool_list_once_per_window(self, monkeypatch, tmp_path, provider):
        adapter, db = _wire_adapter(monkeypatch, tmp_path, provider, toolsets={"web", "file"})
        built = []
        loop_thread = threading.get_ident()

        def _fake_definitions(**kwargs):
            built.append((kwargs, threading.get_ident()))
            return []

        monkeypatch.setattr("model_tools.get_tool_definitions", _fake_definitions)
        try:
            async with TestClient(TestServer(_app(adapter))) as cli:
                first = await (await cli.post("/v1/jentera/warm")).json()
                second = await (await cli.post("/v1/jentera/warm")).json()
            assert first == {"ok": True, "warmed": True}
            assert second == {"ok": True, "warmed": False}

            # Built exactly once, for this gateway's toolsets, quietly, and
            # off the event loop's thread.
            assert len(built) == 1
            kwargs, thread = built[0]
            assert kwargs == {"enabled_toolsets": ["file", "web"], "quiet_mode": True}
            assert thread != loop_thread
        finally:
            db.close()

    @pytest.mark.asyncio
    async def test_warm_runs_again_after_the_window(self, monkeypatch, tmp_path, provider):
        adapter, db = _wire_adapter(monkeypatch, tmp_path, provider)
        monkeypatch.setattr("model_tools.get_tool_definitions", lambda **_: [])
        monkeypatch.setattr("gateway.platforms.api_server._JENTERA_WARM_INTERVAL_S", 0.0)
        try:
            async with TestClient(TestServer(_app(adapter))) as cli:
                first = await (await cli.post("/v1/jentera/warm")).json()
                second = await (await cli.post("/v1/jentera/warm")).json()
            assert first["warmed"] is True
            assert second["warmed"] is True
        finally:
            db.close()

    @pytest.mark.asyncio
    async def test_a_failing_warm_is_not_an_error(self, monkeypatch, tmp_path, provider):
        adapter, db = _wire_adapter(monkeypatch, tmp_path, provider)
        monkeypatch.setattr("model_tools.get_tool_definitions", MagicMock(side_effect=RuntimeError("probe down")))
        try:
            async with TestClient(TestServer(_app(adapter))) as cli:
                resp = await cli.post("/v1/jentera/warm")
                assert resp.status == 200
                assert await resp.json() == {"ok": True, "warmed": False}
        finally:
            db.close()

    @pytest.mark.asyncio
    async def test_warm_needs_the_same_key_as_runs(self, monkeypatch, tmp_path, provider):
        adapter, db = _wire_adapter(monkeypatch, tmp_path, provider, api_key="sk-secret")
        built = MagicMock(return_value=[])
        monkeypatch.setattr("model_tools.get_tool_definitions", built)
        try:
            async with TestClient(TestServer(_app(adapter))) as cli:
                assert (await cli.post("/v1/jentera/warm")).status == 401
                assert (await cli.post("/v1/jentera/warm", headers={"Authorization": "Bearer nope"})).status == 401
                built.assert_not_called()
                ok = await cli.post("/v1/jentera/warm", headers={"Authorization": "Bearer sk-secret"})
                assert ok.status == 200
                assert (await ok.json())["warmed"] is True
        finally:
            db.close()

    def test_warm_is_in_the_route_table_beside_runs(self, monkeypatch, tmp_path, provider):
        adapter, db = _wire_adapter(monkeypatch, tmp_path, provider)
        try:
            rows = {(method, path) for method, path, _ in adapter._http_route_table()}
            assert ("POST", "/v1/runs") in rows
            assert ("POST", "/v1/jentera/warm") in rows
        finally:
            db.close()
