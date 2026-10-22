"""``jentera_run_id`` on POST /v1/runs becomes ``JENTERA_RUN_ID`` in the run's commands.

The Jentera runner finds a run's leftover processes by this marker, so it must reach
every command the run starts (foreground and background terminal, execute_code), must
be absent (not empty) when the run carries none, must never leak between concurrent
runs, and must never be on the shared agent-browser daemon or MCP servers.
"""

import asyncio
import json
import time

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway.test_api_server_turn_notes import (  # noqa: F401  (fixtures + harness)
    _app,
    _chat,
    _run,
    _text,
    _tool_call,
    _wire_adapter,
    provider,
)

RUN_A = "0a1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4d"
RUN_B = "ffffffff-eeee-4ddd-8ccc-bbbbbbbbbbbb"


def _tool_output(provider, call_index=1) -> str:
    """The tool result the agent sent back to the model on request ``call_index``."""
    msgs = _chat(provider)[call_index]["messages"]
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    assert tool_msgs, msgs
    return tool_msgs[-1]["content"]


def _no_approvals(monkeypatch):
    """These tests run harmless commands; nobody is there to approve them."""
    monkeypatch.setattr("tools.approval._get_approval_mode", lambda: "off")


def _terminal_adapter(monkeypatch, tmp_path, provider):
    _no_approvals(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.delenv("JENTERA_RUN_ID", raising=False)
    return _wire_adapter(monkeypatch, tmp_path, provider, toolsets={"terminal"})


class TestMarkerReachesCommands:
    @pytest.mark.asyncio
    async def test_terminal_command_sees_marker(self, monkeypatch, tmp_path, provider):
        adapter, db = _terminal_adapter(monkeypatch, tmp_path, provider)
        provider.queue = [
            _tool_call("terminal", json.dumps({"command": "printenv JENTERA_RUN_ID"})),
            _text("done"),
        ]
        try:
            async with TestClient(TestServer(_app(adapter))) as cli:
                status = await _run(cli, input="go", jentera_run_id=RUN_A, session_id="s-fg")
            assert status["status"] == "completed"
            assert RUN_A in _tool_output(provider)
        finally:
            db.close()

    @pytest.mark.asyncio
    async def test_background_process_sees_marker(self, monkeypatch, tmp_path, provider):
        adapter, db = _terminal_adapter(monkeypatch, tmp_path, provider)
        out = tmp_path / "bg-env.txt"
        provider.queue = [
            _tool_call("terminal", json.dumps({
                "command": f"printenv JENTERA_RUN_ID > {out}",
                "background": True,
            })),
            _text("started"),
        ]
        try:
            async with TestClient(TestServer(_app(adapter))) as cli:
                status = await _run(cli, input="go", jentera_run_id=RUN_A, session_id="s-bg")
            assert status["status"] == "completed"
            for _ in range(100):
                if out.exists() and out.read_text().strip():
                    break
                await asyncio.sleep(0.05)
            assert out.read_text().strip() == RUN_A
        finally:
            db.close()

    @pytest.mark.asyncio
    async def test_execute_code_sees_marker(self, monkeypatch, tmp_path, provider):
        _no_approvals(monkeypatch)
        monkeypatch.setenv("TERMINAL_ENV", "local")
        adapter, db = _wire_adapter(monkeypatch, tmp_path, provider, toolsets={"code_execution"})
        provider.queue = [
            _tool_call("execute_code", json.dumps({
                "code": "import os\nprint('MARK=' + os.environ.get('JENTERA_RUN_ID', '<unset>'))",
            })),
            _text("done"),
        ]
        try:
            async with TestClient(TestServer(_app(adapter))) as cli:
                status = await _run(cli, input="go", jentera_run_id=RUN_A, session_id="s-code")
            assert status["status"] == "completed"
            assert f"MARK={RUN_A}" in _tool_output(provider)
        finally:
            db.close()

    @pytest.mark.asyncio
    async def test_no_marker_without_field(self, monkeypatch, tmp_path, provider):
        # Even a stale copy in the gateway's own environment must not reach the child.
        adapter, db = _terminal_adapter(monkeypatch, tmp_path, provider)
        monkeypatch.setenv("JENTERA_RUN_ID", "stale-from-gateway-env")
        provider.queue = [
            _tool_call("terminal", json.dumps({
                "command": 'if [ "${JENTERA_RUN_ID+set}" = set ]; then echo "SET:[$JENTERA_RUN_ID]"; else echo UNSET; fi',
            })),
            _text("done"),
        ]
        try:
            async with TestClient(TestServer(_app(adapter))) as cli:
                status = await _run(cli, input="go", session_id="s-none")
            assert status["status"] == "completed"
            out = _tool_output(provider)
            assert "UNSET" in out and "SET:" not in out
        finally:
            db.close()

    @pytest.mark.asyncio
    async def test_concurrent_runs_never_see_each_others_marker(self, monkeypatch, tmp_path, provider):
        adapter, db = _terminal_adapter(monkeypatch, tmp_path, provider)
        out_a, out_b = tmp_path / "a.txt", tmp_path / "b.txt"
        # Each run sleeps briefly inside its command so the two overlap.
        def call(out):
            return _tool_call("terminal", json.dumps({"command": f"sleep 0.5; printenv JENTERA_RUN_ID > {out}"}))
        provider.queue = [call(out_a), call(out_b), _text("a"), _text("b")]
        try:
            async with TestClient(TestServer(_app(adapter))) as cli:
                a, b = await asyncio.gather(
                    _run(cli, input="go", jentera_run_id=RUN_A, session_id="s-ca"),
                    _run(cli, input="go", jentera_run_id=RUN_B, session_id="s-cb"),
                )
            assert a["status"] == b["status"] == "completed"
            written = {out_a.read_text().strip(), out_b.read_text().strip()}
            assert written == {RUN_A, RUN_B}
        finally:
            db.close()

    @pytest.mark.asyncio
    async def test_later_run_in_the_same_shell_session_gets_its_own_marker_not_the_earlier_one(
        self, monkeypatch, tmp_path, provider
    ):
        # The terminal keeps one shell per session and replays its exports before every
        # command, so the first run's marker must not survive into the second run.
        adapter, db = _terminal_adapter(monkeypatch, tmp_path, provider)
        def probe(n):
            call = _tool_call("terminal", json.dumps({"command": f'echo "M=[${{JENTERA_RUN_ID-unset}}]" # {n}'}))
            call["choices"][0]["message"]["tool_calls"][0]["id"] = f"call_{n}"
            return call
        provider.queue = [probe(1), _text("one"), probe(2), _text("two"), probe(3), _text("three")]
        try:
            async with TestClient(TestServer(_app(adapter))) as cli:
                await _run(cli, input="go", jentera_run_id=RUN_A, session_id="s-shared")
                await _run(cli, input="go", jentera_run_id=RUN_B, session_id="s-shared")
                await _run(cli, input="go", session_id="s-shared")
            assert f"M=[{RUN_A}]" in _tool_output(provider, 1)
            assert f"M=[{RUN_B}]" in _tool_output(provider, 3)
            assert "M=[unset]" in _tool_output(provider, 5)
        finally:
            db.close()


class TestValidation:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad", ["not-a-uuid", "", "A" * 36, "-" * 36, "0" * 36, RUN_A + "x", RUN_A[:-1], 123, ["x"], True])
    async def test_invalid_marker_400(self, monkeypatch, tmp_path, provider, bad):
        adapter, db = _wire_adapter(monkeypatch, tmp_path, provider)
        try:
            async with TestClient(TestServer(_app(adapter))) as cli:
                resp = await cli.post("/v1/runs", json={"input": "hi", "jentera_run_id": bad})
                assert resp.status == 400
                assert "jentera_run_id" in (await resp.json())["error"]["message"]
            assert adapter._run_streams == {}
            assert provider.requests == []
        finally:
            db.close()

    @pytest.mark.asyncio
    async def test_null_marker_is_accepted_as_absent(self, monkeypatch, tmp_path, provider):
        adapter, db = _wire_adapter(monkeypatch, tmp_path, provider)
        try:
            async with TestClient(TestServer(_app(adapter))) as cli:
                status = await _run(cli, input="hi", jentera_run_id=None, session_id="s-null")
            assert status["status"] == "completed"
        finally:
            db.close()


class TestExemptLaunches:
    def test_browser_daemon_unmarked(self, monkeypatch):
        from gateway.session_context import clear_session_vars, set_session_vars
        from tools.browser_tool import _build_browser_env

        tokens = set_session_vars(platform="api_server", run_marker=RUN_A)
        try:
            monkeypatch.setenv("JENTERA_RUN_ID", RUN_A)
            assert "JENTERA_RUN_ID" not in _build_browser_env()
        finally:
            clear_session_vars(tokens)

    def test_mcp_server_env_unmarked(self, monkeypatch):
        from gateway.session_context import clear_session_vars, set_session_vars
        from tools.mcp_tool import _build_safe_env

        tokens = set_session_vars(platform="api_server", run_marker=RUN_A)
        try:
            monkeypatch.setenv("JENTERA_RUN_ID", RUN_A)
            assert "JENTERA_RUN_ID" not in _build_safe_env(None)
        finally:
            clear_session_vars(tokens)


class TestInternalSpawnsUnmarked:
    """Long-lived Hermes-internal children are shared across runs; none may carry a marker."""

    @pytest.fixture()
    def marked(self, monkeypatch):
        from gateway.session_context import clear_session_vars, set_session_vars

        monkeypatch.setenv("JENTERA_RUN_ID", RUN_A)
        tokens = set_session_vars(platform="api_server", run_marker=RUN_A)
        yield
        clear_session_vars(tokens)

    @pytest.mark.parametrize("inherit", [False, True])
    def test_generic_subprocess_env_unmarked(self, marked, inherit):
        # agent/lsp/client.py _spawn (language servers) and the Codex app-server
        # transport both build their env with this helper.
        from tools.environments.local import hermes_subprocess_env

        assert "JENTERA_RUN_ID" not in hermes_subprocess_env(inherit_credentials=inherit)

    def test_lsp_spawn_env_unmarked(self, marked):
        import inspect

        from agent.lsp import client
        from tools.environments.local import hermes_subprocess_env

        assert "hermes_subprocess_env()" in inspect.getsource(client.LSPClient._spawn)
        assert "JENTERA_RUN_ID" not in hermes_subprocess_env()

    def test_cua_driver_env_unmarked(self, marked):
        from tools.computer_use.cua_backend import cua_driver_child_env
        from tools.environments.local import _sanitize_subprocess_env

        assert "JENTERA_RUN_ID" not in _sanitize_subprocess_env(cua_driver_child_env())
        assert "JENTERA_RUN_ID" not in _sanitize_subprocess_env({"JENTERA_RUN_ID": RUN_A})

    def test_codex_transport_env_unmarked(self, marked):
        import inspect

        from agent.transports import codex_app_server
        from tools.environments.local import hermes_subprocess_env

        assert "hermes_subprocess_env(inherit_credentials=True)" in inspect.getsource(codex_app_server)
        assert "JENTERA_RUN_ID" not in hermes_subprocess_env(inherit_credentials=True)

    def test_agents_own_background_command_is_the_only_stamped_sanitize(self, marked):
        from tools.environments.local import _sanitize_subprocess_env

        assert _sanitize_subprocess_env({}, stamp_run_marker=True)["JENTERA_RUN_ID"] == RUN_A


class TestSessionContext:
    def test_marker_is_task_local_and_cleared(self):
        from gateway.session_context import clear_session_vars, get_run_marker, set_session_vars

        tokens = set_session_vars(platform="api_server", run_marker=RUN_A)
        assert get_run_marker() == RUN_A
        clear_session_vars(tokens)
        assert get_run_marker() == ""


class TestHealth:
    @pytest.mark.asyncio
    async def test_health_lists_run_marker(self, monkeypatch, tmp_path, provider):
        adapter, db = _wire_adapter(monkeypatch, tmp_path, provider)
        try:
            async with TestClient(TestServer(_app(adapter))) as cli:
                for path in ("/health", "/v1/health"):
                    data = await (await cli.get(path)).json()
                    assert "run_marker" in data["jentera_features"]
        finally:
            db.close()
