"""A deferred tool called by its own name goes through the bridge, no strike."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent
from tests.run_agent.test_run_agent import (
    _make_tool_defs,
    _mock_response,
    _mock_tool_call,
)


@pytest.fixture()
def agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    a.client = MagicMock()
    a._cached_system_prompt = "You are helpful."
    a._use_prompt_caching = False
    a.tool_delay = 0
    a.compression_enabled = False
    a.save_trajectories = False
    # Tool search active: bridge tools listed, session_search deferred.
    a.valid_tool_names = set(a.valid_tool_names) | {"tool_search", "tool_describe", "tool_call"}
    return a


def _run(agent, responses):
    agent.client.chat.completions.create.side_effect = responses
    seen = []

    def fake_search(**kw):
        seen.append(("session_search", {"query": kw.get("query")}))
        return json.dumps({"success": True})

    with (
        patch("agent.tool_executor._tool_search_scoped_names", return_value=frozenset({"session_search"})),
        patch.object(agent, "_get_session_db_for_recall", return_value=object()),
        patch("tools.session_search_tool.session_search", side_effect=fake_search),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("find the thing")
    return result, seen


def test_direct_call_to_deferred_tool_dispatches_through_bridge(agent):
    call = _mock_tool_call(name="session_search", arguments='{"query": "x"}', call_id="c1")
    result, seen = _run(agent, [
        _mock_response(content="", finish_reason="tool_calls", tool_calls=[call]),
        _mock_response(content="done", finish_reason="stop"),
    ])
    assert result["completed"] is True and result["final_response"] == "done"
    assert seen and seen[0][0] == "session_search"
    assert seen[0][1] == {"query": "x"}  # arguments survive the rewrite
    assert agent._invalid_tool_retries == 0


def test_three_direct_deferred_calls_do_not_end_the_turn(agent):
    responses = [
        _mock_response(
            content="", finish_reason="tool_calls",
            tool_calls=[_mock_tool_call(name="session_search", arguments='{"query": "q"}', call_id=f"c{i}")],
        )
        for i in range(4)
    ] + [_mock_response(content="finished", finish_reason="stop")]
    result, seen = _run(agent, responses)
    assert result["completed"] is True and result["final_response"] == "finished"
    assert len(seen) == 4
    assert "invalid tool call" not in (result.get("error") or "")


def test_unknown_name_still_counts_as_invalid(agent):
    bad = _mock_tool_call(name="nonexistent_tool", arguments="{}", call_id="c1")
    result, seen = _run(agent, [
        _mock_response(content="", finish_reason="tool_calls", tool_calls=[bad]),
        _mock_response(content="ok", finish_reason="stop"),
    ])
    assert result["final_response"] == "ok"
    assert seen == []


def test_no_rewrite_when_bridge_inactive(agent):
    from agent.conversation_loop import _reroute_deferred_direct_calls

    agent.valid_tool_names = {"web_search"}
    tc = _mock_tool_call(name="session_search", arguments="{}")
    with patch("agent.tool_executor._tool_search_scoped_names", return_value=frozenset({"session_search"})):
        assert _reroute_deferred_direct_calls(agent, [tc]) == 0
    assert tc.function.name == "session_search"
