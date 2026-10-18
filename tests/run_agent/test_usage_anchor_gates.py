"""The post-tool compaction gate prices the transcript with the provider's real usage plus ONLY the
tool results appended since (upstream NousResearch/hermes-agent#104555)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _tool_defs():
    return [{"type": "function", "function": {
        "name": "web_search", "description": "x",
        "parameters": {"type": "object", "properties": {}}}}]


def _resp(tool: bool, prompt_tokens: int, completion_tokens: int = 10):
    calls = [SimpleNamespace(id="call_1", type="function",
                             function=SimpleNamespace(name="web_search", arguments="{}"))] if tool else None
    msg = SimpleNamespace(content=None if tool else "done", reasoning_content=None,
                          reasoning=None, tool_calls=calls)
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens,
                            prompt_tokens_details=None, completion_tokens_details=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="tool_calls" if tool else "stop")],
                           model="test/model", usage=usage)


@pytest.fixture()
def agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs()),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1",
                    quiet_mode=True, skip_context_files=True, skip_memory=True, max_iterations=5)
    a.client = MagicMock()
    a._cached_system_prompt = "You are helpful."
    a._use_prompt_caching = False
    a._disable_streaming = True
    a.tool_delay = 0
    a.save_trajectories = False
    a.compression_enabled = True
    return a


def _recording_compressor():
    c = MagicMock()
    c.protect_first_n = 3
    c.protect_last_n = 20
    c.threshold_tokens = 50_000
    c.context_length = 200_000
    c.last_prompt_tokens = 0
    c.should_compress.return_value = False
    c.should_defer_preflight_to_real_usage.return_value = False
    c.get_active_compression_failure_cooldown.return_value = None
    return c


def _drive(agent, big_tool_result: str):
    agent.client.chat.completions.create.side_effect = [_resp(True, 12_000), _resp(False, 12_500)]
    with (
        patch.object(agent, "_compress_context", side_effect=lambda m, s, **k: (m, "p")),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("run_agent.handle_function_call", lambda name, args, task_id=None, **kw: big_tool_result),
    ):
        agent.run_conversation("go")


def test_post_tool_gate_charges_the_appended_tool_result(agent):
    agent.context_compressor = _recording_compressor()
    _drive(agent, json.dumps({"blob": "x" * 40_000}))  # ~10K tokens by bytes/4
    figures = [c.args[0] for c in agent.context_compressor.should_compress.call_args_list]
    # real 12_000 + 10 completion + the ~10K-token tool result; last_prompt_tokens alone (12_000)
    # would have hidden it.
    assert any(f >= 12_010 + 8_000 for f in figures), figures


def test_post_tool_gate_uses_real_usage_when_result_is_small(agent):
    agent.context_compressor = _recording_compressor()
    _drive(agent, json.dumps({"ok": True}))
    figures = [c.args[0] for c in agent.context_compressor.should_compress.call_args_list]
    assert figures and all(f < 20_000 for f in figures), figures
