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
