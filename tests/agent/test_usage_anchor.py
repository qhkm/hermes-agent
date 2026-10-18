"""Usage-anchored token accounting (upstream NousResearch/hermes-agent#104555, in-process part)."""

import copy
import json
from types import SimpleNamespace

from agent.model_metadata import estimate_messages_tokens_rough
from agent.usage_anchor import (
    agent_anchored_tokens,
    anchored_context_tokens,
    capture_usage_anchor,
    message_fingerprint,
    set_usage_anchor,
)


def _history():
    return [
        {"role": "user", "content": "hello " * 50},
        {"role": "assistant", "content": "hi " * 50},
        {"role": "user", "content": "do the thing"},
    ]


def test_anchor_prices_prefix_with_real_usage_only():
    msgs = _history()
    anchor = capture_usage_anchor(9_000, 100, msgs)
    # Same transcript, reply appended: real usage + completion, nothing re-estimated.
    msgs.append({"role": "assistant", "content": "done " * 20})
    assert anchored_context_tokens(msgs, anchor) == 9_100


def test_only_rows_appended_since_are_estimated():
    msgs = _history()
    anchor = capture_usage_anchor(9_000, 100, msgs)
    reply = {"role": "assistant", "content": "ok", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}]}
    tool = {"role": "tool", "tool_call_id": "c1", "content": "x" * 8_000}
    msgs += [reply, tool]
    expected = 9_100 + estimate_messages_tokens_rough([tool])
    assert anchored_context_tokens(msgs, anchor) == expected
    assert expected > 9_100 + 1_500  # the big tool result is charged


def test_anchor_survives_json_reload_of_history():
    """The gateway re-reads history from the DB every turn: identity must be content-based."""
    msgs = _history()
    anchor = capture_usage_anchor(5_000, 50, msgs)
    reloaded = json.loads(json.dumps(msgs))
    reloaded.append({"role": "assistant", "content": "reply"})
    assert anchored_context_tokens(reloaded, anchor) == 5_050


def test_anchor_fails_closed_on_edit_compaction_or_rewind():
    msgs = _history()
    anchor = capture_usage_anchor(5_000, 50, msgs)

    edited = copy.deepcopy(msgs)
    edited[-1]["content"] = "something else"
    assert anchored_context_tokens(edited, anchor) is None

    compacted = [{"role": "user", "content": "summary"}, msgs[-1]]
    assert anchored_context_tokens(compacted, anchor) is None

    assert anchored_context_tokens(msgs[:2], anchor) is None
    assert anchored_context_tokens(msgs, None) is None


def test_unusable_usage_yields_no_anchor():
    msgs = _history()
    assert capture_usage_anchor(0, 10, msgs) is None
    assert capture_usage_anchor(None, None, msgs) is None
    assert capture_usage_anchor(100, 10, []) is None
    assert message_fingerprint("not a message") is None


def test_agent_helpers_and_clearing():
    agent = SimpleNamespace()
    msgs = _history()
    assert agent_anchored_tokens(agent, msgs) is None
    set_usage_anchor(agent, capture_usage_anchor(7_000, 0, msgs))
    assert agent_anchored_tokens(agent, msgs) == 7_000
    set_usage_anchor(agent, None)
    assert agent_anchored_tokens(agent, msgs) is None


def _anchor_agent(msgs, prompt="SYSTEM", tools=None):
    from agent.usage_anchor import agent_context_fingerprint

    agent = SimpleNamespace(
        _cached_system_prompt=prompt, ephemeral_system_prompt=None,
        tools=tools if tools is not None else [{"type": "function", "function": {"name": "a"}}],
        _usage_anchor=None,
    )
    set_usage_anchor(agent, capture_usage_anchor(9_000, 100, msgs, context_fp=agent_context_fingerprint(agent)))
    return agent


def test_system_prompt_change_invalidates_anchor():
    msgs = _history()
    agent = _anchor_agent(msgs)
    assert agent_anchored_tokens(agent, msgs) == 9_100
    agent._cached_system_prompt = "SYSTEM rewritten for fallback"
    assert agent_anchored_tokens(agent, msgs) is None


def test_tool_schema_change_invalidates_anchor():
    msgs = _history()
    agent = _anchor_agent(msgs)
    agent.tools = agent.tools + [{"type": "function", "function": {"name": "b"}}]
    assert agent_anchored_tokens(agent, msgs) is None


def test_unchanged_prompt_and_tools_keep_anchor():
    msgs = _history()
    agent = _anchor_agent(msgs)
    agent.tools = [dict(t) for t in agent.tools]  # equal content, new objects
    assert agent_anchored_tokens(agent, msgs) == 9_100


def _fallback_agent():
    from unittest.mock import MagicMock
    from agent.context_compressor import ContextCompressor
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent.model, agent.provider = "primary-model", "openrouter"
    agent.base_url = "https://openrouter.ai/api/v1"
    agent.api_key = "sk-primary"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock()
    agent.quiet_mode = True
    agent._fallback_activated = False
    agent._fallback_chain = [{"provider": "openai", "model": "gpt-4o"}]
    agent._fallback_index = 0
    agent.context_compressor = ContextCompressor(
        model="primary-model", threshold_percent=0.50, base_url=agent.base_url,
        api_key="sk-primary", provider="openrouter", quiet_mode=True,
    )
    agent._is_direct_openai_url = lambda url: "api.openai.com" in url
    agent._emit_status = lambda msg: None
    agent._usage_anchor = {"prompt_tokens": 9_000, "completion_tokens": 1, "base_count": 1,
                           "base_last_role": "user", "base_last_fp": "x", "context_fp": "y"}
    return agent


def test_fallback_activation_clears_anchor():
    from unittest.mock import MagicMock, patch

    agent = _fallback_agent()
    fb_client = MagicMock()
    fb_client.base_url = "https://api.openai.com/v1"
    fb_client.api_key = "sk-fallback"
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(fb_client, None)), \
         patch("agent.model_metadata.get_model_context_length", return_value=128_000):
        assert agent._try_activate_fallback() is True
    assert agent._usage_anchor is None


def test_restore_primary_clears_anchor():
    from unittest.mock import MagicMock, patch

    agent = _fallback_agent()
    agent._primary_runtime = {
        "model": "primary-model", "provider": "openrouter", "base_url": "https://openrouter.ai/api/v1",
        "api_mode": "chat_completions", "api_key": "sk-primary", "client_kwargs": {"api_key": "sk-primary"},
        "use_prompt_caching": False, "compressor_model": "primary-model",
        "compressor_context_length": 128_000, "compressor_base_url": "https://openrouter.ai/api/v1",
        "compressor_api_key": "sk-primary", "compressor_provider": "openrouter",
    }
    agent._fallback_activated = True
    agent._transport_cache = MagicMock()
    agent._create_openai_client = lambda *a, **k: MagicMock()
    agent._credential_pool = None
    with patch("agent.chat_completion_helpers.rewrite_prompt_model_identity"):
        assert agent._restore_primary_runtime() is True
    assert agent._usage_anchor is None
