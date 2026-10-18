"""Tests for /v1/runs endpoints: start, status, events, steer, and stop.

Covers:
- POST /v1/runs — start a run (202)
- GET /v1/runs/{run_id} — poll run status
- GET /v1/runs/{run_id}/events — SSE event stream
- POST /v1/runs/{run_id}/steer — inject guidance into a running agent
- POST /v1/runs/{run_id}/stop — interrupt a running agent
- Auth, error handling, and cleanup
"""

import asyncio
import json
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _approval_event_choices,
    cors_middleware,
    security_headers_middleware,
)
from tools import approval as approval_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("smart_denied", "allow_permanent", "expected"),
    [
        (False, True, ["once", "session", "always", "deny"]),
        (False, False, ["once", "session", "deny"]),
        (True, True, ["once", "deny"]),
        (True, False, ["once", "deny"]),
    ],
)
def test_approval_event_choices_follow_backend_capabilities(
    smart_denied, allow_permanent, expected
):
    assert _approval_event_choices(
        smart_denied=smart_denied,
        allow_permanent=allow_permanent,
    ) == expected


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    """Create an adapter with optional API key."""
    extra = {}
    if api_key:
        extra["key"] = api_key
    config = PlatformConfig(enabled=True, extra=extra)
    adapter = APIServerAdapter(config)
    return adapter


def _create_runs_app(adapter: APIServerAdapter) -> web.Application:
    """Create an aiohttp app with /v1/runs routes registered."""
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
    app.router.add_post("/v1/runs/{run_id}/approval", adapter._handle_run_approval)
    app.router.add_post("/v1/runs/{run_id}/steer", adapter._handle_steer_run)
    app.router.add_post("/v1/runs/{run_id}/stop", adapter._handle_stop_run)
    return app


def _make_slow_agent(**kwargs):
    """Create a mock agent that blocks in run_conversation until interrupted.

    Returns (mock_agent, agent_ready_event, interrupt_event) where
    agent_ready_event is set once run_conversation starts, and
    interrupt_event is set when interrupt() is called.
    """
    ready = threading.Event()
    interrupted = threading.Event()

    mock_agent = MagicMock()

    def _do_interrupt(message=None):
        interrupted.set()

    mock_agent.interrupt = MagicMock(side_effect=_do_interrupt)

    def _slow_run(user_message=None, conversation_history=None, task_id=None):
        ready.set()
        # Block until interrupt() is called
        interrupted.wait(timeout=10)
        return {"final_response": "interrupted"}

    mock_agent.run_conversation.side_effect = _slow_run
    mock_agent.session_prompt_tokens = 0
    mock_agent.session_completion_tokens = 0
    mock_agent.session_total_tokens = 0

    return mock_agent, ready, interrupted


@pytest.fixture
def adapter():
    return _make_adapter()


@pytest.fixture
def auth_adapter():
    return _make_adapter(api_key="sk-secret")


# ---------------------------------------------------------------------------
# POST /v1/runs — start a run
# ---------------------------------------------------------------------------


class TestStartRun:
    @pytest.mark.asyncio
    async def test_start_returns_202(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 10
                mock_agent.session_completion_tokens = 5
                mock_agent.session_total_tokens = 15
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                assert data["status"] == "started"
                assert data["run_id"].startswith("run_")

                status_resp = await cli.get(f"/v1/runs/{data['run_id']}")
                assert status_resp.status == 200
                status = await status_resp.json()
                assert status["run_id"] == data["run_id"]
                assert status["status"] in {"queued", "running", "completed"}
                assert status["object"] == "hermes.run"

    @pytest.mark.asyncio
    async def test_start_invalid_json_returns_400(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs",
                data="not json",
                headers={"Content-Type": "application/json"},
            )
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_start_missing_input_returns_400(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json={"model": "test"})
            assert resp.status == 400
            data = await resp.json()
            assert "input" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_start_empty_input_returns_400(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json={"input": ""})
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_start_invalid_history_does_not_allocate_run(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs",
                json={"input": "hello", "conversation_history": {"role": "user"}},
            )
        assert resp.status == 400
        assert adapter._run_streams == {}
        assert adapter._run_statuses == {}

    @pytest.mark.asyncio
    async def test_start_requires_auth(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json={"input": "hello"})
        assert resp.status == 401

    @pytest.mark.asyncio
    async def test_start_with_valid_auth(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "ok"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello"},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                assert resp.status == 202


# ---------------------------------------------------------------------------
# /v1/runs — persisted-transcript hydration for stable session ids
# ---------------------------------------------------------------------------


class TestStartRunSessionHydration:
    """Follow-up turns on a stable session_id must receive the stored
    transcript, mirroring /api/sessions/{id}/chat.

    Regression: Jentera's Telegram worker sends the same per-chat
    session_id (``telegram:<businessId>:<chatId>``) on every message, but
    /v1/runs never reloaded the persisted transcript — every turn started
    with ``conversation_history=[]``. A bare follow-up like "yes pls" had no
    referent and the bot answered unrelated recent-work hints instead of the
    thread the user was actually in.
    """

    @pytest.mark.asyncio
    async def test_start_with_session_id_hydrates_persisted_history(self, adapter):
        app = _create_runs_app(adapter)
        stored = [
            {"role": "user", "content": "What is our deployment plan?"},
            {"role": "assistant", "content": "Ship the worker fix first."},
        ]

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "right, do that"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                with patch.object(
                    adapter,
                    "_conversation_history_for_session",
                    new=AsyncMock(return_value=stored),
                ) as mock_history:
                    resp = await cli.post(
                        "/v1/runs",
                        json={"input": "yes pls", "session_id": "telegram:acme:12345"},
                    )
                    assert resp.status == 202
                    data = await resp.json()
                    run_id = data["run_id"]

                    mock_history.assert_awaited_once_with("telegram:acme:12345")

                    for _ in range(20):
                        status_resp = await cli.get(f"/v1/runs/{run_id}")
                        status = await status_resp.json()
                        if status["status"] == "completed":
                            break
                        await asyncio.sleep(0.05)

                assert status["status"] == "completed"
                mock_agent.run_conversation.assert_called_once()
                assert mock_agent.run_conversation.call_args.kwargs["conversation_history"] == stored
                assert mock_agent.run_conversation.call_args.kwargs["task_id"] == "telegram:acme:12345"

    @pytest.mark.asyncio
    async def test_start_stateless_run_skips_hydration(self, adapter):
        """No client session_id -> fresh run stays empty; the SessionDB
        lookup is skipped entirely (a brand-new run_id has no history)."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "ok"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                with patch.object(
                    adapter,
                    "_conversation_history_for_session",
                    new=AsyncMock(return_value=[{"role": "user", "content": "stale"}]),
                ) as mock_history:
                    resp = await cli.post("/v1/runs", json={"input": "hello"})
                    assert resp.status == 202
                    data = await resp.json()
                    run_id = data["run_id"]
                    mock_history.assert_not_called()

                    for _ in range(20):
                        status_resp = await cli.get(f"/v1/runs/{run_id}")
                        status = await status_resp.json()
                        if status["status"] == "completed":
                            break
                        await asyncio.sleep(0.05)

                assert status["status"] == "completed"
                mock_agent.run_conversation.assert_called_once()
                assert mock_agent.run_conversation.call_args.kwargs["conversation_history"] == []

    @pytest.mark.asyncio
    async def test_start_explicit_history_wins_over_persisted(self, adapter):
        """Explicit conversation_history takes precedence — hydration must
        not override client-supplied context on the same session_id."""
        app = _create_runs_app(adapter)
        explicit = [{"role": "user", "content": "explicit context"}]
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "ok"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                with patch.object(
                    adapter,
                    "_conversation_history_for_session",
                    new=AsyncMock(return_value=[{"role": "user", "content": "stale"}]),
                ) as mock_history:
                    resp = await cli.post(
                        "/v1/runs",
                        json={
                            "input": "hello",
                            "session_id": "telegram:acme:12345",
                            "conversation_history": explicit,
                        },
                    )
                    assert resp.status == 202
                    data = await resp.json()
                    run_id = data["run_id"]
                    mock_history.assert_not_called()

                    for _ in range(20):
                        status_resp = await cli.get(f"/v1/runs/{run_id}")
                        status = await status_resp.json()
                        if status["status"] == "completed":
                            break
                        await asyncio.sleep(0.05)

                assert status["status"] == "completed"
                mock_agent.run_conversation.assert_called_once()
                assert mock_agent.run_conversation.call_args.kwargs["conversation_history"] == explicit


# ---------------------------------------------------------------------------
# GET /v1/runs/{run_id} — poll run status
# ---------------------------------------------------------------------------


class TestRunStatus:
    @pytest.mark.asyncio
    async def test_status_completed_run_includes_output_and_usage(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 4
                mock_agent.session_completion_tokens = 2
                mock_agent.session_total_tokens = 6
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                data = await resp.json()
                run_id = data["run_id"]

                for _ in range(20):
                    status_resp = await cli.get(f"/v1/runs/{run_id}")
                    assert status_resp.status == 200
                    status = await status_resp.json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)

                assert status["status"] == "completed"
                assert status["output"] == "done"
                assert status["usage"]["total_tokens"] == 6
                assert status["last_event"] == "run.completed"

    @pytest.mark.asyncio
    async def test_status_reflects_explicit_session_id(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello", "session_id": "space-session"},
                )
                data = await resp.json()
                run_id = data["run_id"]

                for _ in range(20):
                    status_resp = await cli.get(f"/v1/runs/{run_id}")
                    status = await status_resp.json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)

                mock_agent.run_conversation.assert_called_once()
                assert mock_agent.run_conversation.call_args.kwargs["task_id"] == "space-session"
                assert status["session_id"] == "space-session"

    @pytest.mark.asyncio
    async def test_status_not_found_returns_404(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/runs/run_nonexistent")
        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_status_requires_auth(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/runs/run_any")
        assert resp.status == 401


# ---------------------------------------------------------------------------
# GET /v1/runs/{run_id}/events — SSE event stream
# ---------------------------------------------------------------------------


class TestRunEvents:
    @pytest.mark.asyncio
    async def test_events_stream_returns_completed(self, adapter):
        """Events stream should receive run.completed when agent finishes."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "Hello!"}
                mock_agent.session_prompt_tokens = 10
                mock_agent.session_completion_tokens = 5
                mock_agent.session_total_tokens = 15
                mock_create.return_value = mock_agent

                # Start run
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                # Subscribe to events
                events_resp = await cli.get(f"/v1/runs/{run_id}/events")
                assert events_resp.status == 200
                body = await events_resp.text()

                # Should contain run.completed
                assert "run.completed" in body
                assert "Hello!" in body



    @pytest.mark.asyncio
    async def test_approval_response_without_pending_returns_409(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                data = await resp.json()
                run_id = data["run_id"]

                approval_resp = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={"choice": "once"},
                )
                assert approval_resp.status == 409
                approval_data = await approval_resp.json()
                assert approval_data["error"]["code"] in {
                    "approval_not_active",
                    "approval_not_pending",
                }

    @pytest.mark.asyncio
    async def test_approval_string_false_does_not_resolve_all(self, adapter):
        """Quoted false must not fan out approval resolution across the queue."""
        app = _create_runs_app(adapter)
        run_id = "run_bool_parse"
        adapter._run_statuses[run_id] = {"run_id": run_id, "status": "running"}
        adapter._run_approval_sessions[run_id] = "session-123"

        async with TestClient(TestServer(app)) as cli:
            with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
                approval_resp = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={"choice": "once", "all": "false"},
                )

        assert approval_resp.status == 200
        mock_resolve.assert_called_once_with(
            "session-123",
            "once",
            resolve_all=False,
            reason=None,
            request_id=None,
        )

    @pytest.mark.asyncio
    async def test_approval_resolve_all_is_scoped_to_target_run(self, auth_adapter):
        """Same client session_id must not let one run approve another run's queue."""
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_create_agent") as mock_create:
                victim_agent, victim_ready, victim_interrupted = _make_slow_agent()
                attacker_agent, attacker_ready, attacker_interrupted = _make_slow_agent()
                mock_create.side_effect = [victim_agent, attacker_agent]

                victim_resp = await cli.post(
                    "/v1/runs",
                    json={"input": "victim", "session_id": "shared-project"},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                attacker_resp = await cli.post(
                    "/v1/runs",
                    json={"input": "attacker", "session_id": "shared-project"},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                assert victim_resp.status == 202
                assert attacker_resp.status == 202
                victim_run = (await victim_resp.json())["run_id"]
                attacker_run = (await attacker_resp.json())["run_id"]

                await asyncio.to_thread(victim_ready.wait, 3.0)
                await asyncio.to_thread(attacker_ready.wait, 3.0)
                assert auth_adapter._run_approval_sessions[victim_run] == victim_run
                assert auth_adapter._run_approval_sessions[attacker_run] == attacker_run
                assert auth_adapter._run_approval_sessions[victim_run] != auth_adapter._run_approval_sessions[attacker_run]

                victim_entry = approval_mod._ApprovalEntry({
                    "command": "bash -c victim-danger",
                    "description": "victim approval",
                    "pattern_keys": ["shell-c"],
                })
                attacker_entry = approval_mod._ApprovalEntry({
                    "command": "bash -c attacker-danger",
                    "description": "attacker approval",
                    "pattern_keys": ["shell-c"],
                })
                with approval_mod._lock:
                    approval_mod._gateway_queues[victim_run] = [victim_entry]
                    approval_mod._gateway_queues[attacker_run] = [attacker_entry]

                approval_resp = await cli.post(
                    f"/v1/runs/{attacker_run}/approval",
                    json={"choice": "always", "resolve_all": True},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                approval_data = await approval_resp.json()

                assert approval_resp.status == 200
                assert approval_data["resolved"] == 1
                assert attacker_entry.result == "always"
                assert attacker_entry.event.is_set()
                assert victim_entry.result is None
                assert not victim_entry.event.is_set()
                with approval_mod._lock:
                    assert approval_mod._gateway_queues[victim_run] == [victim_entry]
                    assert victim_run in approval_mod._gateway_queues
                    assert attacker_run not in approval_mod._gateway_queues

                # Clean up the synthetic pending victim approval and unblock the
                # slow test agents so their background run tasks can finish.
                with approval_mod._lock:
                    approval_mod._gateway_queues.pop(victim_run, None)
                victim_interrupted.set()
                attacker_interrupted.set()


    @pytest.mark.asyncio
    async def test_events_not_found_returns_404(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/runs/run_nonexistent/events")
        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_events_requires_auth(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/runs/run_any/events")
        assert resp.status == 401


# ---------------------------------------------------------------------------
# POST /v1/runs/{run_id}/approval — correlated, retry-safe decisions
# ---------------------------------------------------------------------------


class TestRunApproval:
    @pytest.mark.asyncio
    async def test_request_event_and_deny_reason_are_correlated(self, adapter):
        app = _create_runs_app(adapter)
        decision_holder = {}

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0

                def _run_with_approval(**_kwargs):
                    session_key = approval_mod.get_current_session_key()
                    with approval_mod._lock:
                        notify = approval_mod._gateway_notify_cbs[session_key]
                    decision_holder["decision"] = approval_mod._await_gateway_decision(
                        session_key,
                        notify,
                        {
                            "command": "bash -c dangerous",
                            "description": "dangerous shell",
                            "pattern_key": "shell-c",
                            "pattern_keys": ["shell-c"],
                        },
                    )
                    return {"final_response": "blocked"}

                mock_agent.run_conversation.side_effect = _run_with_approval
                mock_create.return_value = mock_agent

                start_resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert start_resp.status == 202
                run_id = (await start_resp.json())["run_id"]
                run_task = adapter._active_run_tasks[run_id]

                request_event = await asyncio.wait_for(
                    adapter._run_streams[run_id].get(), timeout=3
                )
                assert request_event["event"] == "approval.request"
                request_id = request_event["request_id"]
                assert isinstance(request_id, str)
                assert request_id

                deny_resp = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={
                        "choice": "deny",
                        "request_id": request_id,
                        "reason": "production policy forbids this action",
                    },
                )
                assert deny_resp.status == 200
                assert (await deny_resp.json())["resolved"] == 1

                await asyncio.wait_for(run_task, timeout=3)
                assert decision_holder["decision"] == {
                    "resolved": True,
                    "choice": "deny",
                    "reason": "production policy forbids this action",
                }

    @pytest.mark.asyncio
    async def test_targeted_retry_returns_success_without_resolving_fifo_head(
        self, adapter
    ):
        app = _create_runs_app(adapter)
        adapter._set_run_status("run_retry", "waiting_for_approval")
        adapter._run_approval_sessions["run_retry"] = "run_retry"
        first = approval_mod._ApprovalEntry({"command": "first"})
        second = approval_mod._ApprovalEntry({"command": "second"})
        with approval_mod._lock:
            approval_mod._gateway_queues["run_retry"] = [first, second]

        async with TestClient(TestServer(app)) as cli:
            first_resp = await cli.post(
                "/v1/runs/run_retry/approval",
                json={"choice": "once", "request_id": second.data["request_id"]},
            )
            retry_resp = await cli.post(
                "/v1/runs/run_retry/approval",
                json={"choice": "deny", "request_id": second.data["request_id"]},
            )
            retry_payload = await retry_resp.json()

        assert first_resp.status == 200
        assert retry_resp.status == 200
        assert retry_payload["resolved"] == 1
        assert first.result is None
        assert not first.event.is_set()
        assert second.result == "once"
        assert adapter._run_statuses["run_retry"]["status"] == "waiting_for_approval"

        approval_mod.unregister_gateway_notify("run_retry")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("request_id", ["", 123, {}, []])
    async def test_invalid_request_id_is_rejected(self, adapter, request_id):
        app = _create_runs_app(adapter)
        adapter._set_run_status("run_invalid", "waiting_for_approval")
        adapter._run_approval_sessions["run_invalid"] = "run_invalid"

        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(
                "/v1/runs/run_invalid/approval",
                json={"choice": "once", "request_id": request_id},
            )
            payload = await response.json()

        assert response.status == 400
        assert payload["error"]["code"] == "invalid_approval_request_id"


# ---------------------------------------------------------------------------
# POST /v1/runs/{run_id}/steer — steer a running agent
# ---------------------------------------------------------------------------


def _chat_completion(content=None, tool_name=None):
    message = {"role": "assistant", "content": content}
    finish = "stop"
    if tool_name:
        message["tool_calls"] = [{
            "id": "call_1",
            "type": "function",
            "function": {"name": tool_name, "arguments": "{}"},
        }]
        finish = "tool_calls"
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "fake-model",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


class TestSteerRun:
    @pytest.mark.asyncio
    async def test_steer_running_agent(self, adapter):
        app = _create_runs_app(adapter)
        agent = MagicMock()
        agent.steer.return_value = True
        queue = asyncio.Queue()
        adapter._active_run_agents["run_123"] = agent
        adapter._run_streams["run_123"] = queue
        adapter._set_run_status("run_123", "running")

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_123/steer", json={"input": "tighten the ending"})
            payload = await resp.json()

        assert resp.status == 200
        assert payload == {"object": "hermes.run.steer", "run_id": "run_123", "accepted": True}
        agent.steer.assert_called_once_with("tighten the ending")
        assert adapter._run_statuses["run_123"]["last_event"] == "run.steered"
        assert adapter._run_statuses["run_123"]["status"] == "running"
        event = queue.get_nowait()
        assert event["event"] == "run.steered"
        assert event["run_id"] == "run_123"
        assert event["accepted"] is True

    @pytest.mark.asyncio
    async def test_steer_accepts_message_and_text_aliases(self, adapter):
        app = _create_runs_app(adapter)
        agent = MagicMock()
        agent.steer.return_value = True
        adapter._active_run_agents["run_123"] = agent
        adapter._set_run_status("run_123", "running")
        async with TestClient(TestServer(app)) as cli:
            r1 = await cli.post("/v1/runs/run_123/steer", json={"message": "a"})
            r2 = await cli.post("/v1/runs/run_123/steer", json={"text": "b"})
        assert r1.status == 200 and r2.status == 200
        assert [c.args[0] for c in agent.steer.call_args_list] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_steer_nonexistent_run_returns_404(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_missing/steer", json={"input": "hello"})
            payload = await resp.json()

        assert resp.status == 404
        assert payload["error"]["code"] == "run_not_found"

    @pytest.mark.asyncio
    async def test_steer_finished_run_returns_409(self, adapter):
        app = _create_runs_app(adapter)
        agent = MagicMock()
        # Even a retained agent ref must not make a terminal run steerable.
        adapter._active_run_agents["run_done"] = agent
        adapter._set_run_status("run_done", "completed")

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_done/steer", json={"input": "hello"})
            payload = await resp.json()

        assert resp.status == 409
        assert payload["error"]["code"] == "run_not_accepting_steer"
        agent.steer.assert_not_called()

    @pytest.mark.asyncio
    async def test_steer_run_completing_during_body_read_returns_409(self, adapter):
        """A run that finishes while the body is read must not be resurrected."""
        app = _create_runs_app(adapter)
        agent = MagicMock()
        agent.steer.return_value = True
        adapter._active_run_agents["run_race"] = agent
        adapter._set_run_status("run_race", "running")
        real_read = adapter._read_json_body

        async def _read_then_complete(request):
            result = await real_read(request)
            adapter._set_run_status("run_race", "completed", last_event="run.completed")
            return result

        with patch.object(adapter, "_read_json_body", _read_then_complete):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post("/v1/runs/run_race/steer", json={"input": "late"})
                payload = await resp.json()

        assert resp.status == 409
        assert payload["error"]["code"] == "run_not_accepting_steer"
        agent.steer.assert_not_called()
        assert adapter._run_statuses["run_race"]["status"] == "completed"
        assert adapter._run_statuses["run_race"]["last_event"] == "run.completed"

    @pytest.mark.asyncio
    async def test_steer_keeps_stopping_status_after_stop(self, adapter):
        app = _create_runs_app(adapter)
        agent = MagicMock()
        agent.steer.return_value = True
        adapter._active_run_agents["run_stop"] = agent
        adapter._set_run_status("run_stop", "stopping", last_event="run.stopping")
        adapter._stopping_run_ids.add("run_stop")

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_stop/steer", json={"input": "late"})

        assert resp.status == 409
        agent.steer.assert_not_called()
        assert adapter._run_statuses["run_stop"]["status"] == "stopping"

    @pytest.mark.asyncio
    async def test_successful_steer_keeps_current_status(self, adapter):
        """Only last_event changes; the status is whatever it currently is."""
        app = _create_runs_app(adapter)
        agent = MagicMock()
        agent.steer.return_value = True
        adapter._active_run_agents["run_ok"] = agent
        adapter._set_run_status("run_ok", "running")

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_ok/steer", json={"input": "go"})

        assert resp.status == 200
        assert adapter._run_statuses["run_ok"]["status"] == "running"
        assert adapter._run_statuses["run_ok"]["last_event"] == "run.steered"

    @pytest.mark.asyncio
    async def test_steer_over_pending_cap_returns_413(self, adapter):
        from gateway.platforms.api_server import _STEER_MAX_PENDING_CHARS

        app = _create_runs_app(adapter)
        agent = MagicMock()
        agent.steer.return_value = True
        agent._pending_steer = "x" * (_STEER_MAX_PENDING_CHARS - 10)
        adapter._active_run_agents["run_big"] = agent
        adapter._set_run_status("run_big", "running")

        async with TestClient(TestServer(app)) as cli:
            too_big = await cli.post("/v1/runs/run_big/steer", json={"input": "y" * 50})
            payload = await too_big.json()
            fits_text = "z" * 8
            agent._pending_steer = "x" * (_STEER_MAX_PENDING_CHARS - 10)
            ok = await cli.post("/v1/runs/run_big/steer", json={"input": fits_text})

        assert too_big.status == 413
        assert payload["error"]["code"] == "steer_too_large"
        assert ok.status == 200
        agent.steer.assert_called_once_with(fits_text)
        assert adapter._run_statuses["run_big"]["status"] == "running"

    @pytest.mark.asyncio
    async def test_single_steer_over_cap_returns_413(self, adapter):
        from gateway.platforms.api_server import _STEER_MAX_PENDING_CHARS

        app = _create_runs_app(adapter)
        agent = MagicMock()
        agent._pending_steer = None
        adapter._active_run_agents["run_one"] = agent
        adapter._set_run_status("run_one", "running")
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs/run_one/steer", json={"input": "a" * (_STEER_MAX_PENDING_CHARS + 1)}
            )
        assert resp.status == 413
        agent.steer.assert_not_called()

    @pytest.mark.asyncio
    async def test_steer_queued_run_returns_409(self, adapter):
        app = _create_runs_app(adapter)
        adapter._set_run_status("run_q", "queued")
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_q/steer", json={"input": "hello"})
        assert resp.status == 409

    @pytest.mark.asyncio
    async def test_steer_missing_input_returns_400(self, adapter):
        app = _create_runs_app(adapter)
        agent = MagicMock()
        agent.steer.return_value = True
        adapter._active_run_agents["run_123"] = agent
        adapter._set_run_status("run_123", "running")

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_123/steer", json={"input": ""})
            payload = await resp.json()

        assert resp.status == 400
        assert payload["error"]["code"] == "invalid_steer_input"
        agent.steer.assert_not_called()

    @pytest.mark.asyncio
    async def test_steer_not_accepted_by_agent_returns_409(self, adapter):
        app = _create_runs_app(adapter)
        agent = MagicMock()
        agent.steer.return_value = False
        adapter._active_run_agents["run_123"] = agent
        adapter._set_run_status("run_123", "running")
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_123/steer", json={"input": "x"})
            payload = await resp.json()
        assert resp.status == 409
        assert payload["error"]["code"] == "steer_not_accepted"

    @pytest.mark.asyncio
    async def test_stop_then_steer_rejects_retained_agent_ref(self, adapter):
        """Steer must reject a stopping run even if the executor thread is still live."""
        app = _create_runs_app(adapter)
        run_can_finish = threading.Event()
        run_started = threading.Event()

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_agent.steer = MagicMock(return_value=True)

                def _run_conversation(*_args, **_kwargs):
                    run_started.set()
                    run_can_finish.wait(timeout=5)
                    return {"final_response": "late result"}

                mock_agent.interrupt = MagicMock(return_value=None)
                mock_agent.run_conversation.side_effect = _run_conversation
                mock_create.return_value = mock_agent

                start_resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await start_resp.json())["run_id"]
                assert await asyncio.to_thread(run_started.wait, 3.0)

                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                assert run_id in adapter._active_run_agents

                steer_resp = await cli.post(
                    f"/v1/runs/{run_id}/steer", json={"input": "tighten the ending"}
                )
                steer_data = await steer_resp.json()

                assert steer_resp.status == 409
                assert steer_data["error"]["code"] == "run_not_accepting_steer"
                mock_agent.steer.assert_not_called()

                run_can_finish.set()
                for _ in range(40):
                    if run_id not in adapter._active_run_tasks:
                        break
                    await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_pending_steer_preserved_on_run_completed(self, adapter):
        """A steer left over after the final response (drained by the turn
        finalizer into result["pending_steer"]) surfaces on the terminal run
        status and event instead of being dropped."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_agent.run_conversation.return_value = {
                    "final_response": "done",
                    "pending_steer": "tighten the ending",
                }
                mock_create.return_value = mock_agent

                start_resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await start_resp.json())["run_id"]
                events_resp = await cli.get(f"/v1/runs/{run_id}/events")
                body = await events_resp.text()

                for _ in range(40):
                    if adapter._run_statuses.get(run_id, {}).get("status") == "completed":
                        break
                    await asyncio.sleep(0.05)

                status_resp = await cli.get(f"/v1/runs/{run_id}")
                status = await status_resp.json()

        assert status["status"] == "completed"
        assert status["pending_steer"] == "tighten the ending"
        assert "run.completed" in body and "tighten the ending" in body

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "result, expected_status, expected_steer",
        [
            ({"final_response": "done"}, "completed", "late one"),
            ({"final_response": "done", "pending_steer": "early"}, "completed", "early\nlate one"),
            ({"failed": True, "error": "boom"}, "failed", "late one"),
            ({"failed": True, "error": "boom", "pending_steer": "early"}, "failed", "early\nlate one"),
        ],
    )
    async def test_steer_after_finalizer_drain_reaches_terminal_status(
        self, adapter, result, expected_status, expected_steer
    ):
        """A steer accepted after the finalizer's drain (the run is still
        'running') is drained on the loop and carried on the terminal state."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_agent.run_conversation.return_value = result
                mock_agent._drain_pending_steer.return_value = "late one"
                mock_create.return_value = mock_agent

                start_resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await start_resp.json())["run_id"]
                events_resp = await cli.get(f"/v1/runs/{run_id}/events")
                body = await events_resp.text()
                for _ in range(40):
                    if adapter._run_statuses.get(run_id, {}).get("status") == expected_status:
                        break
                    await asyncio.sleep(0.05)

        status = adapter._run_statuses[run_id]
        assert status["status"] == expected_status
        assert status["pending_steer"] == expected_steer
        assert "late one" in body

    @pytest.mark.asyncio
    async def test_runs_build_the_agent_off_the_event_loop(self, adapter):
        """Agent construction can block (tool-availability probes); it must not
        run on the API server's event loop thread."""
        app = _create_runs_app(adapter)
        loop_thread = threading.get_ident()
        build_threads = []

        def _fake_create(*_a, **_kw):
            build_threads.append(threading.get_ident())
            agent = MagicMock()
            agent.session_prompt_tokens = 0
            agent.session_completion_tokens = 0
            agent.session_total_tokens = 0
            agent.run_conversation.return_value = {"final_response": "ok"}
            return agent

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", side_effect=_fake_create):
                start_resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await start_resp.json())["run_id"]
                for _ in range(60):
                    if adapter._run_statuses.get(run_id, {}).get("status") == "completed":
                        break
                    await asyncio.sleep(0.05)

        assert adapter._run_statuses[run_id]["status"] == "completed"
        assert build_threads and build_threads[0] != loop_thread

    @pytest.mark.asyncio
    async def test_completed_run_without_leftover_has_no_pending_steer(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_create.return_value = mock_agent
                start_resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await start_resp.json())["run_id"]
                for _ in range(40):
                    if adapter._run_statuses.get(run_id, {}).get("status") == "completed":
                        break
                    await asyncio.sleep(0.05)
        assert "pending_steer" not in adapter._run_statuses[run_id]

    @pytest.mark.asyncio
    async def test_steer_requires_auth(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        agent = MagicMock()
        auth_adapter._active_run_agents["run_any"] = agent
        auth_adapter._set_run_status("run_any", "running")
        async with TestClient(TestServer(app)) as cli:
            no_key = await cli.post("/v1/runs/run_any/steer", json={"input": "hello"})
            wrong = await cli.post(
                "/v1/runs/run_any/steer",
                json={"input": "hello"},
                headers={"Authorization": "Bearer sk-wrong"},
            )
        assert no_key.status == 401
        assert wrong.status == 401
        agent.steer.assert_not_called()

    @pytest.mark.asyncio
    async def test_steer_via_profile_prefixed_route(self, auth_adapter):
        """The steer route is in the route table and mirrored under /p/<profile>/."""
        rows = auth_adapter._http_route_table()
        assert ("POST", "/v1/runs/{run_id}/steer", auth_adapter._handle_steer_run) in rows

        app = web.Application()
        for method, path, handler in rows:
            if path == "/v1/runs/{run_id}/steer":
                app.router.add_route(method, f"/p/{{profile}}{path}", handler)
        agent = MagicMock()
        agent.steer.return_value = True
        auth_adapter._active_run_agents["run_p"] = agent
        auth_adapter._set_run_status("run_p", "running")
        headers = {"Authorization": "Bearer sk-secret"}
        async with TestClient(TestServer(app)) as cli:
            bad = await cli.post("/p/coder/v1/runs/run_p/steer", json={"input": "x"})
            ok = await cli.post(
                "/p/coder/v1/runs/run_p/steer", json={"input": "go left"}, headers=headers
            )
        assert bad.status == 401
        assert ok.status == 200
        agent.steer.assert_called_once_with("go left")

    @pytest.mark.asyncio
    async def test_steer_reaches_next_model_request_end_to_end(self, adapter, monkeypatch):
        """Real AIAgent against a fake OpenAI-compatible server: a steer sent
        while the first model call is in flight lands in the tool result that
        the second model request carries."""
        from run_agent import AIAgent

        requests = []
        first_seen = threading.Event()
        release_first = threading.Event()

        async def completions(request):
            body = await request.json()
            requests.append(body)
            if len(requests) == 1:
                first_seen.set()
                while not release_first.is_set():
                    await asyncio.sleep(0.01)
                payload = _chat_completion(tool_name="no_such_tool")
            else:
                payload = _chat_completion(content="all done")
            if not body.get("stream"):
                return web.json_response(payload)
            # The agent streams by default: replay the message as SSE chunks.
            choice = payload["choices"][0]
            delta = dict(choice["message"])
            if delta.get("tool_calls"):
                delta["tool_calls"] = [dict(tc, index=0) for tc in delta["tool_calls"]]
            base = {"id": payload["id"], "object": "chat.completion.chunk",
                    "created": 0, "model": "fake-model"}
            frames = [
                dict(base, choices=[{"index": 0, "delta": delta, "finish_reason": None}]),
                dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}]),
                dict(base, choices=[], usage=payload["usage"]),
            ]
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            for frame in frames:
                await resp.write(f"data: {json.dumps(frame)}\n\n".encode())
            await resp.write(b"data: [DONE]\n\n")
            return resp

        # The fake model runs on its own thread and loop: AIAgent makes
        # blocking HTTP probes (model metadata) during init, which would
        # deadlock against a server sharing the test's event loop.
        server_ready = threading.Event()
        server_state = {}

        def _serve():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            model_app = web.Application()
            model_app.router.add_post("/v1/chat/completions", completions)
            runner = web.AppRunner(model_app)
            loop.run_until_complete(runner.setup())
            site = web.TCPSite(runner, "127.0.0.1", 0)
            loop.run_until_complete(site.start())
            server_state["port"] = site._server.sockets[0].getsockname()[1]
            server_state["loop"] = loop
            server_ready.set()
            loop.run_forever()
            loop.run_until_complete(runner.cleanup())
            loop.close()

        server_thread = threading.Thread(target=_serve, daemon=True)
        server_thread.start()
        assert server_ready.wait(timeout=10)

        app = _create_runs_app(adapter)
        try:
            async with TestClient(TestServer(app)) as cli:
                base_url = f"http://127.0.0.1:{server_state['port']}/v1"

                def _create_agent(**_kwargs):
                    return AIAgent(
                        base_url=base_url,
                        api_key="sk-test",
                        model="fake-model",
                        provider="custom",
                        quiet_mode=True,
                        skip_context_files=True,
                        skip_memory=True,
                        enabled_toolsets=[],
                        max_iterations=5,
                    )

                with patch.object(adapter, "_create_agent", side_effect=_create_agent):
                    start = await cli.post("/v1/runs", json={"input": "hello"})
                    run_id = (await start.json())["run_id"]
                    assert await asyncio.get_running_loop().run_in_executor(None, first_seen.wait, 20)
                    for _ in range(100):
                        if run_id in adapter._active_run_agents:
                            break
                        await asyncio.sleep(0.02)

                    steer = await cli.post(
                        f"/v1/runs/{run_id}/steer", json={"input": "prefer the blue option"}
                    )
                    assert steer.status == 200
                    release_first.set()

                    for _ in range(200):
                        if adapter._run_statuses[run_id]["status"] in {"completed", "failed"}:
                            break
                        await asyncio.sleep(0.05)
        finally:
            release_first.set()
            server_state["loop"].call_soon_threadsafe(server_state["loop"].stop)
            server_thread.join(timeout=5)

        status = adapter._run_statuses[run_id]
        assert status["status"] == "completed", status
        assert len(requests) == 2
        assert "prefer the blue option" not in json.dumps(requests[0])
        second = requests[1]["messages"]
        tool_msgs = [m for m in second if m.get("role") == "tool"]
        assert tool_msgs and "prefer the blue option" in tool_msgs[-1]["content"]
        assert "pending_steer" not in status


# ---------------------------------------------------------------------------
# Run lifecycle TTL sweeping
# ---------------------------------------------------------------------------


class TestRunLifecycleSweep:
    def test_sweep_keeps_transport_with_active_subscriber(self, adapter):
        run_id = "run_subscribed"
        queue = asyncio.Queue()
        adapter._run_streams[run_id] = queue
        adapter._run_streams_created[run_id] = 0
        adapter._run_stream_subscribers.add(run_id)

        adapter._sweep_orphaned_runs_once(time.time())

        assert adapter._run_streams[run_id] is queue
        assert run_id in adapter._run_streams_created

    @pytest.mark.asyncio
    async def test_expired_live_run_drops_transport_but_keeps_control_state(self, adapter):
        """Stream TTL bounds buffering without detaching a live run."""
        app = _create_runs_app(adapter)
        adapter._max_concurrent_runs = 1

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, _ = _make_slow_agent()
                mock_create.return_value = mock_agent

                start_resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert start_resp.status == 202
                run_id = (await start_resp.json())["run_id"]
                assert await asyncio.to_thread(agent_ready.wait, 3.0)

                task = adapter._active_run_tasks[run_id]
                assert isinstance(task, asyncio.Task)
                assert not task.done()

                pending = approval_mod._ApprovalEntry({
                    "command": "bash -c long-running",
                    "description": "approval after stream TTL",
                    "pattern_keys": ["shell-c"],
                })
                with approval_mod._lock:
                    approval_mod._gateway_queues[run_id] = [pending]

                adapter._run_streams_created[run_id] -= adapter._RUN_STREAM_TTL + 1
                # Exercise one real sweeper iteration without waiting 60 seconds.
                with patch(
                    "gateway.platforms.api_server.asyncio.sleep",
                    side_effect=[None, asyncio.CancelledError()],
                ):
                    with pytest.raises(asyncio.CancelledError):
                        await adapter._sweep_orphaned_runs()

                assert adapter._active_run_tasks[run_id] is task
                assert adapter._active_run_agents[run_id] is mock_agent
                assert run_id not in adapter._run_streams
                assert run_id not in adapter._run_streams_created
                assert adapter._run_approval_sessions[run_id] == run_id

                limited = adapter._concurrency_limited_response()
                assert limited is not None
                assert limited.status == 429

                approval_resp = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={"choice": "once"},
                )
                assert approval_resp.status == 200
                assert pending.event.is_set()
                assert pending.result == "once"

                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                mock_agent.interrupt.assert_called_once_with("Stop requested via API")

    @pytest.mark.asyncio
    async def test_expired_transport_stops_buffering_new_deltas(self, adapter):
        """An unconsumed expired queue must not grow for the rest of a live run."""
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, _ = _make_slow_agent()
                mock_create.return_value = mock_agent

                start_resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await start_resp.json())["run_id"]
                assert await asyncio.to_thread(agent_ready.wait, 3.0)
                expired_queue = adapter._run_streams[run_id]
                stream_delta = mock_create.call_args.kwargs["stream_delta_callback"]

                adapter._run_streams_created[run_id] -= adapter._RUN_STREAM_TTL + 1
                adapter._sweep_orphaned_runs_once(time.time())
                before = expired_queue.qsize()
                stream_delta("must-not-buffer")
                mock_agent.interrupt("finish test")
                for _ in range(40):
                    if run_id not in adapter._active_run_tasks:
                        break
                    await asyncio.sleep(0.05)

                assert expired_queue.qsize() == before

    @pytest.mark.asyncio
    async def test_expired_orphan_run_state_is_reaped(self, adapter):
        run_id = "run_expired_orphan"
        adapter._run_streams[run_id] = asyncio.Queue()
        adapter._run_streams_created[run_id] = 0
        adapter._run_approval_sessions[run_id] = run_id

        pending = approval_mod._ApprovalEntry({
            "command": "bash -c orphaned",
            "description": "orphaned approval",
            "pattern_keys": ["shell-c"],
        })
        with approval_mod._lock:
            approval_mod._gateway_queues[run_id] = [pending]

        with patch(
            "gateway.platforms.api_server.asyncio.sleep",
            side_effect=[None, asyncio.CancelledError()],
        ):
            with pytest.raises(asyncio.CancelledError):
                await adapter._sweep_orphaned_runs()

        assert run_id not in adapter._run_streams
        assert run_id not in adapter._run_streams_created
        assert run_id not in adapter._run_approval_sessions
        assert pending.event.is_set()
        with approval_mod._lock:
            assert run_id not in approval_mod._gateway_queues


# ---------------------------------------------------------------------------
# POST /v1/runs/{run_id}/stop — interrupt a running agent
# ---------------------------------------------------------------------------


class TestStopRun:
    @pytest.mark.asyncio
    async def test_stop_before_agent_creation_prevents_run_start(self, adapter):
        """A stop accepted while queued must prevent agent construction."""
        app = _create_runs_app(adapter)
        original_create_task = asyncio.create_task
        task_started = asyncio.Event()
        allow_task = asyncio.Event()

        def _delayed_create_task(coro):
            async def _delayed():
                task_started.set()
                await allow_task.wait()
                return await coro

            return original_create_task(_delayed())

        with patch("gateway.platforms.api_server.asyncio.create_task", side_effect=_delayed_create_task), \
             patch.object(adapter, "_create_agent") as mock_create:
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await resp.json())["run_id"]
                await task_started.wait()

                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                allow_task.set()

                for _ in range(20):
                    if run_id not in adapter._active_run_tasks:
                        break
                    await asyncio.sleep(0.05)

                mock_create.assert_not_called()
                assert adapter._run_statuses[run_id]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_stop_keeps_uncooperative_executor_tracked_until_exit(self, adapter):
        """Cancelling an asyncio wrapper must not hide its live executor thread."""
        app = _create_runs_app(adapter)
        run_can_finish = threading.Event()
        run_finished = threading.Event()

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                started = threading.Event()

                def _run_conversation(*_args, **_kwargs):
                    started.set()
                    run_can_finish.wait(timeout=5)
                    run_finished.set()
                    return {"final_response": "late result"}

                mock_agent.run_conversation.side_effect = _run_conversation
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await resp.json())["run_id"]
                assert await asyncio.to_thread(started.wait, 3)

                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                await asyncio.sleep(0.1)

                assert not run_finished.is_set()
                assert run_id in adapter._active_run_agents
                assert run_id in adapter._active_run_tasks
                assert adapter._run_statuses[run_id]["status"] == "stopping"

                run_can_finish.set()
                for _ in range(40):
                    if run_id not in adapter._active_run_tasks:
                        break
                    await asyncio.sleep(0.05)

                assert run_id not in adapter._active_run_agents
                assert run_id not in adapter._active_run_tasks
                assert adapter._run_statuses[run_id]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_stop_running_agent(self, adapter):
        """Stop should interrupt the agent and cancel the task."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, _ = _make_slow_agent()
                mock_create.return_value = mock_agent

                # Start run
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                # Wait for agent to start running in the thread
                await asyncio.to_thread(agent_ready.wait, 3.0)
                await asyncio.sleep(0.1)

                # Verify agent ref is stored
                assert run_id in adapter._active_run_agents

                # Stop the run
                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                stop_data = await stop_resp.json()
                assert stop_data["run_id"] == run_id
                assert stop_data["status"] == "stopping"

                # Agent interrupt should have been called
                mock_agent.interrupt.assert_called_once_with("Stop requested via API")

                status_resp = await cli.get(f"/v1/runs/{run_id}")
                assert status_resp.status == 200
                status_data = await status_resp.json()
                assert status_data["status"] in {"stopping", "cancelled"}

                # Refs should be cleaned up
                await asyncio.sleep(0.5)
                assert run_id not in adapter._active_run_agents
                assert run_id not in adapter._active_run_tasks

    @pytest.mark.asyncio
    async def test_stop_nonexistent_run_returns_404(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_nonexistent/stop")
        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_stop_requires_auth(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_any/stop")
        assert resp.status == 401

    @pytest.mark.asyncio
    async def test_stop_already_completed_run_returns_404(self, adapter):
        """Stopping a run that already finished should return 404 (refs cleaned up)."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                # Start and wait for completion
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                await asyncio.sleep(0.3)

                # Run should be done, refs cleaned up
                assert run_id not in adapter._active_run_agents

                # Stop should return 404
                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 404

    @pytest.mark.asyncio
    async def test_stop_interrupt_exception_does_not_crash(self, adapter):
        """If agent.interrupt() raises, stop should still succeed."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, interrupted = _make_slow_agent()

                # Override the interrupt side_effect to raise. Still trip
                # ``interrupted`` so the slow_run thread unblocks at teardown
                # — without this the agent thread blocks the full 10s
                # timeout and the test teardown waits the same amount.
                def _raising_interrupt(message=None):
                    interrupted.set()
                    raise RuntimeError("interrupt failed")

                mock_agent.interrupt = MagicMock(side_effect=_raising_interrupt)
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                await asyncio.to_thread(agent_ready.wait, 3.0)
                await asyncio.sleep(0.1)

                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                stop_data = await stop_resp.json()
                assert stop_data["status"] == "stopping"

    @pytest.mark.asyncio
    async def test_stop_sends_sentinel_to_events_stream(self, adapter):
        """After stop, the events stream should close."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, _ = _make_slow_agent()
                mock_create.return_value = mock_agent

                # Start run
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                await asyncio.to_thread(agent_ready.wait, 3.0)
                await asyncio.sleep(0.1)

                # Subscribe to events in background
                events_task = asyncio.ensure_future(
                    cli.get(f"/v1/runs/{run_id}/events")
                )

                await asyncio.sleep(0.1)

                # Stop the run
                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200

                # Events stream should close
                events_resp = await asyncio.wait_for(events_task, timeout=5.0)
                assert events_resp.status == 200
                body = await events_resp.text()
                # Stream should have received run.failed and closed
                assert "run.failed" in body or "stream closed" in body


# ---------------------------------------------------------------------------
# Turn liveness watchdog (upstream #99758) — stalled run ends as run.failed
# ---------------------------------------------------------------------------


class TestTurnLivenessRun:
    @pytest.mark.asyncio
    async def test_stalled_run_fails_and_next_run_completes(self, adapter):
        from run_agent import AIAgent

        agent = AIAgent.__new__(AIAgent)
        agent.session_id = "sess-liveness"
        agent._interrupt_requested = False
        agent._interrupt_message = None
        agent._pending_redirect = None
        agent._last_activity_ts = time.time()
        agent._last_activity_desc = "idle"
        agent.api_mode = "chat_completions"
        agent._emit_warning = lambda *_a, **_k: None
        agent._conversation_root_id = lambda: "root"
        agent.session_prompt_tokens = 0
        agent.session_completion_tokens = 0
        agent.session_total_tokens = 0

        turns = {"n": 0}

        def _loop(agent_, user_message, *a, **k):
            turns["n"] += 1
            agent_._interrupt_requested = False
            if turns["n"] == 1:
                deadline = time.time() + 10
                while not agent_._interrupt_requested and time.time() < deadline:
                    time.sleep(0.01)
                return {"final_response": "Operation interrupted.", "interrupted": True}
            return {"final_response": "second ok", "completed": True}

        cfg = {"agent": {"turn_liveness": {"timeout_s": 0.2, "poll_s": 0.02}}}
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", return_value=agent), patch(
                "agent.conversation_loop.run_conversation", _loop
            ), patch("hermes_cli.config.load_config", return_value=cfg):
                bodies = []
                for text in ("stall", "again"):
                    resp = await cli.post("/v1/runs", json={"input": text})
                    assert resp.status == 202
                    run_id = (await resp.json())["run_id"]
                    events = await cli.get(f"/v1/runs/{run_id}/events")
                    bodies.append(await events.text())
                    status = await (await cli.get(f"/v1/runs/{run_id}")).json()
                    bodies[-1] = (bodies[-1], status)

        (first_body, first_status), (second_body, second_status) = bodies
        assert "run.failed" in first_body and "run.completed" not in first_body
        assert "no progress" in first_body
        assert first_status["status"] == "failed"
        assert "run.completed" in second_body and "run.failed" not in second_body
        assert second_status["status"] == "completed"

    @pytest.mark.asyncio
    async def test_unwindable_run_fails_and_abandons_the_agent(self, adapter):
        import threading
        from agent import turn_liveness
        from run_agent import AIAgent
        import gateway.platforms.api_server as api_server

        agent = AIAgent.__new__(AIAgent)
        agent.session_id = "sess-wedged"
        agent._interrupt_requested = False
        agent._interrupt_message = None
        agent._pending_redirect = None
        agent._last_activity_ts = time.time()
        agent._last_activity_desc = "idle"
        agent.api_mode = "chat_completions"
        agent._emit_warning = lambda *_a, **_k: None
        agent._conversation_root_id = lambda: "root"
        agent.session_prompt_tokens = 0
        agent.session_completion_tokens = 0
        agent.session_total_tokens = 0

        release = threading.Event()

        def _wedged(agent_, user_message, *a, **k):
            release.wait(10)  # ignores the interrupt entirely
            return {"final_response": "late", "completed": True}

        before = turn_liveness.abandoned_thread_count()
        cfg = {"agent": {"turn_liveness": {"timeout_s": 0.2, "poll_s": 0.02}}}
        app = _create_runs_app(adapter)
        try:
            async with TestClient(TestServer(app)) as cli:
                with patch.object(adapter, "_create_agent", return_value=agent), patch(
                    "agent.conversation_loop.run_conversation", _wedged
                ), patch("hermes_cli.config.load_config", return_value=cfg), patch.object(
                    api_server, "_TURN_LIVENESS_UNWIND_GRACE_S", 0.3
                ):
                    resp = await cli.post("/v1/runs", json={"input": "wedge"})
                    run_id = (await resp.json())["run_id"]
                    body = await (await cli.get(f"/v1/runs/{run_id}/events")).text()
                    status = await (await cli.get(f"/v1/runs/{run_id}")).json()
                    assert "run.failed" in body and status["status"] == "failed"
                    assert agent._abandoned is True
                    assert turn_liveness.abandoned_thread_count() == before + 1
        finally:
            release.set()
        for _ in range(100):
            if turn_liveness.abandoned_thread_count() == before:
                break
            await asyncio.sleep(0.05)
        assert turn_liveness.abandoned_thread_count() == before


def test_run_event_callback_forwards_lifecycle_status_as_run_status():
    """Auto-recovery waits reach /v1/runs consumers as ``run.status`` (unknown events are ignored)."""
    import asyncio
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter

    async def _go():
        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        loop = asyncio.get_running_loop()
        q = asyncio.Queue()
        adapter._run_streams["r1"] = q
        cb = adapter._make_run_event_callback("r1", loop)
        cb("run.status", preview="⏳ Provider temporarily unavailable — retrying automatically in 15s (cycle 1/5)", kind="lifecycle")
        event = await asyncio.wait_for(q.get(), 1)
        assert event["event"] == "run.status" and event["kind"] == "lifecycle"
        assert "cycle 1/5" in event["text"] and event["run_id"] == "r1"

    asyncio.run(_go())
