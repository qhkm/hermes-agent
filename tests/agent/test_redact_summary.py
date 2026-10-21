"""The iteration-limit summary request goes through the private-form scrubber too.

``handle_max_iterations`` builds its own ``api_messages`` and calls the
provider directly, bypassing the main loop. A tool result carrying a value the
Jentera runner typed for the owner must not reach the model on that request
either: every text and every tool-call argument is scrubbed exactly as in the
main loop (``agent.redact.scrub_outgoing_messages``), whichever API mode the
summary uses, and on its retry.
"""

import copy
import importlib
import json
import types
from unittest.mock import patch

import pytest

CARD = "4111 1111 1111 1111"
FAILED = "[hidden: could not check for private values]"


class _Redact:
    """``agent.redact`` as it is when used, not as it was when this file was imported.

    A few other test files drop the ``agent.*`` modules from ``sys.modules`` to
    import them afresh. Code that imports ``agent.redact`` lazily (the main
    loop, ``handle_max_iterations``) then sees a newer module than one bound at
    import time, and a scrubber installed on the old one is never consulted.
    """

    def __getattr__(self, name):
        return getattr(importlib.import_module("agent.redact"), name)


redact = _Redact()


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    assert redact.MODEL_TEXT_CHECK_FAILED == FAILED
    # The Jentera plugin installs its own scrubber whenever plugins are
    # discovered, and the first agent built in a process triggers that. Keep it
    # from replacing the one a test installs.
    monkeypatch.setattr("plugins.jentera.private_form.set_model_text_scrubber", lambda scrubber: None)
    redact.set_model_text_scrubber(None)
    yield
    redact.set_model_text_scrubber(None)


def _hide(texts):
    return [t.replace(CARD, "[hidden]") for t in texts]


def _raises(texts):
    raise RuntimeError("runner unreachable")


def _history():
    return [
        {"role": "user", "content": f"pay with {CARD}"},
        {"role": "assistant", "content": f"typing {CARD}", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "browser_type", "arguments": json.dumps({"text": CARD})}}]},
        {"role": "tool", "name": "browser_snapshot", "tool_call_id": "c1",
         "content": f"- textbox [ref=e3]: {CARD}"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": [{"type": "text", "text": f"receipt {CARD}"}]},
        {"role": "assistant", "content": f"noted {CARD}"},
    ]


def _agent(api_mode=None, streaming=False):
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._cached_system_prompt = "SYS"
    if api_mode:
        agent.api_mode = api_mode
    if streaming:
        agent.stream_delta_callback = lambda *_: None
    return agent


def _transport(replies, seen):
    """A transport that records what it is asked to send and answers in turn."""

    def build_kwargs(**kw):
        seen.append(copy.deepcopy(kw["messages"]))
        return {"messages": kw["messages"]}

    def normalize_response(_raw, **_kw):
        return types.SimpleNamespace(content=replies.pop(0))

    return types.SimpleNamespace(build_kwargs=build_kwargs, normalize_response=normalize_response)


def _summarise(messages, mode, replies=("SUMMARY",)):
    """Run the summary in ``mode``; return (final text, every request's messages)."""
    from agent.chat_completion_helpers import handle_max_iterations

    replies = list(replies)
    seen = []
    streaming = mode == "stream"
    scrubber = redact._MODEL_TEXT_SCRUBBER
    agent = _agent({"codex": "codex_responses", "anthropic": "anthropic_messages"}.get(mode), streaming)
    redact.set_model_text_scrubber(scrubber)
    transport = _transport(replies, seen)

    def completions_create(**kwargs):
        seen.append(copy.deepcopy(kwargs["messages"]))
        return "RAW"

    def streaming_call(kwargs):
        seen.append(copy.deepcopy(kwargs["messages"]))
        return "RAW"

    def build_api_kwargs(api_messages):
        seen.append(copy.deepcopy(api_messages))
        return {"input": api_messages, "tools": []}

    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=completions_create)))
    with (
        patch.object(agent, "_ensure_primary_openai_client", return_value=client),
        patch.object(agent, "_get_transport", return_value=transport),
        patch.object(agent, "_interruptible_streaming_api_call", side_effect=streaming_call),
        patch.object(agent, "_build_api_kwargs", side_effect=build_api_kwargs),
        patch.object(agent, "_run_codex_stream", return_value="RAW"),
        patch.object(agent, "_anthropic_messages_create", return_value="RAW"),
    ):
        out = handle_max_iterations(agent, messages, 5)
    return out, seen


MODES = ["plain", "stream", "codex", "anthropic"]


def _arguments(request):
    return [
        call["function"]["arguments"]
        for message in request
        for call in message.get("tool_calls") or []
    ]


@pytest.mark.parametrize("mode", MODES)
def test_the_summary_request_carries_the_replacement_not_the_value(mode):
    redact.set_model_text_scrubber(_hide)
    out, seen = _summarise(_history(), mode)
    assert out == "SUMMARY"
    assert len(seen) == 1
    blob = json.dumps(seen[0])
    assert CARD not in blob
    # Message content (string, list-of-parts, assistant text, tool result) ...
    assert [m["content"] for m in seen[0] if m["role"] == "tool"] == ["- textbox [ref=e3]: [hidden]"]
    assert "pay with [hidden]" in blob and "receipt [hidden]" in blob
    assert "typing [hidden]" in blob and "noted [hidden]" in blob
    # ... and tool-call arguments.
    assert [json.loads(a) for a in _arguments(seen[0])] == [{"text": "[hidden]"}]


@pytest.mark.parametrize("mode", MODES)
def test_the_retry_after_an_empty_summary_is_scrubbed_too(mode):
    redact.set_model_text_scrubber(_hide)
    out, seen = _summarise(_history(), mode, replies=["", "SUMMARY"])
    assert out == "SUMMARY"
    assert len(seen) == 2
    for request in seen:
        assert CARD not in json.dumps(request)
        assert [json.loads(a) for a in _arguments(request)] == [{"text": "[hidden]"}]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("scrubber", [_raises, lambda texts: [FAILED for _ in texts]], ids=["raises", "answers-failed"])
def test_a_check_that_fails_withholds_text_and_keeps_arguments_json(mode, scrubber):
    redact.set_model_text_scrubber(scrubber)
    out, seen = _summarise(_history(), mode, replies=["", "SUMMARY"])
    assert out == "SUMMARY"
    assert len(seen) == 2  # the retry fails closed as well
    for request in seen:
        assert CARD not in json.dumps(request)
        assert [m["content"] for m in request if m["role"] == "tool"] == [FAILED]
        assert [json.loads(a) for a in _arguments(request)] == [{"hidden": FAILED}]
        assert FAILED in json.dumps(request)


@pytest.mark.parametrize("mode", MODES)
def test_the_stored_messages_are_never_rewritten(mode):
    redact.set_model_text_scrubber(_hide)
    messages = _history()
    stored = copy.deepcopy(messages)
    originals = [m.get("tool_calls") for m in messages]
    _summarise(messages, mode)
    # The summary prompt is appended, nothing before it changes.
    assert messages[:len(stored)] == stored
    for before, after in zip(originals, [m.get("tool_calls") for m in messages]):
        assert before is after


@pytest.mark.parametrize("mode", MODES)
def test_with_no_scrubber_the_request_is_unchanged(mode):
    messages = _history()
    out, seen = _summarise(messages, mode)
    assert out == "SUMMARY"
    request = seen[0]
    assert request[0] == {"role": "system", "content": "SYS"}
    assert [m["content"] for m in request if m["role"] == "tool"] == [f"- textbox [ref=e3]: {CARD}"]
    assert [json.loads(a) for a in _arguments(request)] == [{"text": CARD}]
    assert f"pay with {CARD}" in json.dumps(request)
    assert request[-1]["role"] == "user" and "maximum number of tool-calling iterations" in json.dumps(request[-1])
    assert FAILED not in json.dumps(request)


def test_the_notes_are_added_before_the_scrub_so_a_value_in_them_is_blanked():
    # The turn notes go on the current user message first; the scrub then sees
    # the message as it will be sent.
    from agent.chat_completion_helpers import handle_max_iterations

    agent = _agent()
    redact.set_model_text_scrubber(_hide)
    agent.turn_notes = f"card on file {CARD}"
    agent._persist_user_message_idx = 0
    seen = []
    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(
        create=lambda **kw: (seen.append(copy.deepcopy(kw["messages"])), "RAW")[1])))
    transport = types.SimpleNamespace(normalize_response=lambda _r: types.SimpleNamespace(content="SUMMARY"))
    with patch.object(agent, "_ensure_primary_openai_client", return_value=client), \
            patch.object(agent, "_get_transport", return_value=transport):
        handle_max_iterations(agent, [{"role": "user", "content": "hello"}], 5)
    first_user = next(m for m in seen[0] if m["role"] == "user")["content"]
    assert "card on file [hidden]" in first_user and first_user.endswith("hello")
    assert CARD not in json.dumps(seen[0])


# ---------------------------------------------------------------------------
# The shared helper itself (the main loop calls the same function)
# ---------------------------------------------------------------------------


def _api_messages():
    # What both callers hand over: shallow per-call copies of the stored messages.
    return [dict(m) for m in _history()]


def test_the_helper_fails_closed_when_the_scrub_itself_cannot_run():
    redact.set_model_text_scrubber(_hide)
    stored = _history()
    api = [dict(m) for m in stored]
    before = copy.deepcopy(stored)
    with patch.object(importlib.import_module("agent.redact"), "scrub_model_texts", side_effect=RuntimeError("boom")):
        out = redact.scrub_outgoing_messages(api)
    assert out is api
    assert CARD not in json.dumps(api)
    assert [m["content"] for m in api if m["role"] == "tool"] == [FAILED]
    assert [json.loads(a) for a in _arguments(api)] == [{"hidden": FAILED}]
    assert [m["content"] for m in api if m["role"] in ("user", "assistant") and m["content"]] == [FAILED] * 5
    # Nothing the stored messages hold was touched, even on this path.
    assert stored == before
    assert stored[1]["tool_calls"][0]["function"]["arguments"] == json.dumps({"text": CARD})


def test_the_helper_leaves_messages_alone_when_the_scrub_cannot_run_and_no_scrubber_is_installed():
    api = _api_messages()
    before = copy.deepcopy(api)
    with patch.object(importlib.import_module("agent.redact"), "scrub_model_texts", side_effect=RuntimeError("boom")):
        redact.scrub_outgoing_messages(api)
    assert api == before


def test_the_helper_is_a_no_op_without_a_scrubber():
    api = _api_messages()
    before = copy.deepcopy(api)
    assert redact.scrub_outgoing_messages(api) is api
    assert api == before


def test_the_helper_copies_lists_and_tool_calls_before_replacing_them():
    redact.set_model_text_scrubber(_hide)
    stored = _history()
    before = copy.deepcopy(stored)
    api = [dict(m) for m in stored]
    redact.scrub_outgoing_messages(api)
    assert stored == before
    assert api[4]["content"] is not stored[4]["content"]
    assert api[1]["tool_calls"] is not stored[1]["tool_calls"]
    assert api[1]["tool_calls"][0] is not stored[1]["tool_calls"][0]
    assert api[1]["tool_calls"][0]["function"] is not stored[1]["tool_calls"][0]["function"]
    assert json.loads(api[1]["tool_calls"][0]["function"]["arguments"]) == {"text": "[hidden]"}


def _nested_history():
    # Content nested more than one level deep: parts inside a part's text.
    return [
        {"role": "user", "content": [{"type": "text", "text": [{"type": "text", "text": CARD}]}]},
        {"role": "assistant", "content": {"type": "text", "text": [f"noted {CARD}", {"content": CARD}]}},
    ]


def test_deeply_nested_content_is_copied_not_edited_in_place():
    redact.set_model_text_scrubber(_hide)
    stored = _nested_history()
    before = copy.deepcopy(stored)
    api = [dict(m) for m in stored]
    redact.scrub_outgoing_messages(api)
    assert CARD not in json.dumps(api)
    assert api[0]["content"][0]["text"][0]["text"] == "[hidden]"
    assert api[1]["content"]["text"] == ["noted [hidden]", {"content": "[hidden]"}]
    # The stored messages share every nested list and dict with the copies; none of them was touched.
    assert stored == before


@pytest.mark.parametrize("mode", MODES)
def test_the_stored_messages_with_nested_content_are_never_rewritten_by_the_summary(mode):
    redact.set_model_text_scrubber(_hide)
    messages = _nested_history() + _history()
    stored = copy.deepcopy(messages)
    _, seen = _summarise(messages, mode)
    assert CARD not in json.dumps(seen[0])
    assert messages[:len(stored)] == stored


def test_a_failure_before_the_tool_calls_are_reached_still_leaves_the_stored_ones_alone():
    # A message whose content cannot be walked (a cycle) fails the scrub part-way:
    # the messages after it have not had their tool calls copied yet.
    redact.set_model_text_scrubber(_hide)
    cycle = []
    cycle.append(cycle)
    stored = [
        {"role": "user", "content": cycle},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "t", "arguments": json.dumps({"text": CARD})}}]},
    ]
    api = [dict(m) for m in stored]
    redact.scrub_outgoing_messages(api)
    assert api[0]["content"] == FAILED
    assert [json.loads(a) for a in _arguments(api)] == [{"hidden": FAILED}]
    assert json.loads(stored[1]["tool_calls"][0]["function"]["arguments"]) == {"text": CARD}
