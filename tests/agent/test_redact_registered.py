"""Model-bound text goes through the private-form scrubber a plugin installs."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import agent.redact as redact
from run_agent import AIAgent

CARD = "4111 1111 1111 1111"


@pytest.fixture(autouse=True)
def _clean():
    redact.set_model_text_scrubber(None)
    yield
    redact.set_model_text_scrubber(None)


def _hide(texts):
    return [t.replace(CARD, "[hidden]") for t in texts]


def test_without_a_scrubber_nothing_changes():
    assert redact.scrub_model_texts([CARD, ""]) == [CARD, ""]


def test_the_scrubber_sees_every_text_and_its_answer_is_used():
    seen = []
    redact.set_model_text_scrubber(lambda texts: (seen.append(list(texts)), _hide(texts))[1])
    assert redact.scrub_model_texts([f"a {CARD}", "b"]) == ["a [hidden]", "b"]
    assert seen == [[f"a {CARD}", "b"]]


@pytest.mark.parametrize("broken", [
    lambda texts: (_ for _ in ()).throw(RuntimeError("boom")),
    lambda texts: texts[:-1],
    lambda texts: None,
    lambda texts: [1 for _ in texts],
])
def test_a_broken_scrubber_withholds_the_text(broken):
    redact.set_model_text_scrubber(broken)
    assert redact.scrub_model_texts([CARD, ""]) == [redact.MODEL_TEXT_CHECK_FAILED, ""]


def test_redact_sensitive_text_no_longer_holds_values(monkeypatch):
    monkeypatch.setattr(redact, "_REDACT_ENABLED", False)
    assert not hasattr(redact, "register_vault_redaction_value")
    assert redact.redact_sensitive_text(f"receipt {CARD}") == f"receipt {CARD}"


def _resp(content="done", tool_calls=None, finish="stop"):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls, reasoning_content=None, reasoning=None)
    r = SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish)], model="test/model")
    r.usage = None
    return r


def _run(history, message="continue"):
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1",
                        quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.client.chat.completions.create.return_value = _resp()
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        agent.run_conversation(message, conversation_history=history)
    return agent.client.chat.completions.create.call_args.kwargs["messages"]


def _history():
    return [
        {"role": "user", "content": [{"type": "text", "text": [{"type": "text", "text": CARD}]}]},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "browser_type", "arguments": json.dumps({"text": CARD})}}]},
        {"role": "tool", "name": "browser_snapshot", "tool_call_id": "c1",
         "content": f"- textbox [ref=e3]: {CARD}"},
    ]


def test_every_outgoing_text_and_tool_argument_is_scrubbed_in_the_request():
    redact.set_model_text_scrubber(_hide)
    history = _history()
    blob = repr(_run(history))
    assert CARD not in blob
    assert "[hidden]" in blob
    # The stored history is not rewritten.
    assert CARD in history[2]["content"]
    assert CARD in history[1]["tool_calls"][0]["function"]["arguments"]


def test_a_check_that_fails_withholds_text_and_keeps_arguments_json():
    redact.set_model_text_scrubber(lambda texts: [redact.MODEL_TEXT_CHECK_FAILED for _ in texts])
    sent = _run(_history())
    blob = repr(sent)
    assert CARD not in blob
    assert redact.MODEL_TEXT_CHECK_FAILED in blob
    for message in sent:
        for call in message.get("tool_calls") or []:
            assert json.loads(call["function"]["arguments"]) == {"hidden": redact.MODEL_TEXT_CHECK_FAILED}


def test_with_no_scrubber_the_request_is_unchanged():
    assert CARD in repr(_run(_history()))
