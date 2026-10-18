"""Full-loop checks for the bounded auto-recovery ladder (upstream #116353)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent import turn_recovery_autorecover as ar
from run_agent import AIAgent


class _HttpErr(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status_code = status
        self.message = message
        self.response = SimpleNamespace(headers={})
        self.body = None


def _agent(cycles=5, retries=2):
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI", return_value=MagicMock()),
    ):
        agent = AIAgent(
            api_key="k-abcdef123456", base_url="http://proxy.invalid/v1", provider="custom",
            model="m", quiet_mode=True, skip_context_files=True, skip_memory=True,
        )
    agent.client = MagicMock()
    agent._api_max_retries = retries
    agent._auto_recovery_cycles = cycles
    return agent


def _ok(text):
    msg = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], model="m", usage=None)


def _run(agent, fake_call, waits):
    with (
        patch.object(agent, "_interruptible_api_call", side_effect=fake_call),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("agent.conversation_loop.time.sleep"),
        patch("agent.agent_runtime_helpers.time.sleep"),
        patch.object(ar, "_interruptible_wait", lambda a, w, label: waits.append(w) or False),
    ):
        return agent.run_conversation("hello")


def test_litellm_no_healthy_deployments_recovers_after_ladder_wait():
    agent = _agent(retries=2)
    calls = []

    def fake(api_kwargs):
        calls.append(1)
        if len(calls) <= 2:  # initial attempts exhaust max_retries=2
            raise _HttpErr(400, "litellm.BadRequestError: There are no healthy deployments available")
        return _ok("back online")

    waits = []
    result = _run(agent, fake, waits)
    assert result["completed"] is True and result["final_response"] == "back online"
    assert len(waits) == 1 and 15 <= waits[0] <= 18


def test_ladder_gives_up_after_configured_cycles():
    agent = _agent(cycles=2, retries=1)
    waits = []

    def fake(api_kwargs):
        raise _HttpErr(503, "Service temporarily unavailable")

    result = _run(agent, fake, waits)
    assert result.get("completed") is False and result.get("failed") is True
    assert len(waits) == 2


def test_auth_error_fails_fast_without_waiting():
    agent = _agent()
    waits = []

    def fake(api_kwargs):
        raise _HttpErr(401, "Incorrect API key provided")

    result = _run(agent, fake, waits)
    assert result.get("completed") is False and waits == []
