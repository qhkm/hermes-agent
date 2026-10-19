"""Registered private-form values are blanked from everything the model sees."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import agent.redact as redact
from run_agent import AIAgent

CARD = "4111 1111 1111 1111"


@pytest.fixture(autouse=True)
def _clean():
    redact.clear_vault_redaction_values()
    yield
    redact.clear_vault_redaction_values()


def test_registered_value_is_blanked_with_redaction_off(monkeypatch):
    monkeypatch.setattr(redact, "_REDACT_ENABLED", False)
    redact.register_vault_redaction_value(CARD)
    out = redact.redact_sensitive_text(f"receipt {CARD}")
    assert out == "receipt [hidden]"


def test_longest_first_and_cr_lf_form():
    redact.register_vault_redaction_value("1234")
    redact.register_vault_redaction_value("1234-5678\n")
    assert redact.redact_registered_vault_values("x 1234-5678 y 1234") == "x [hidden] y [hidden]"


def test_clear_drops_values():
    redact.register_vault_redaction_value(CARD)
    redact.clear_vault_redaction_values()
    assert redact.redact_registered_vault_values(CARD) == CARD


def _resp(content="done", tool_calls=None, finish="stop"):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls, reasoning_content=None, reasoning=None)
    r = SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish)], model="test/model")
    r.usage = None
    return r


def test_snapshot_tool_result_is_blanked_in_the_request(monkeypatch):
    monkeypatch.setattr(redact, "_REDACT_ENABLED", False)
    redact.register_vault_redaction_value(CARD)
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
    history = [
        {"role": "user", "content": "pay the bill"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "browser_snapshot", "arguments": "{}"}}]},
        {"role": "tool", "name": "browser_snapshot", "tool_call_id": "c1",
         "content": f"- textbox [ref=e3]: {CARD}"},
    ]
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        agent.run_conversation("continue", conversation_history=history)
    sent = agent.client.chat.completions.create.call_args.kwargs["messages"]
    blob = repr(sent)
    assert CARD not in blob
    assert "[hidden]" in blob


def test_values_of_another_profile_are_not_scrubbed(monkeypatch, tmp_path):
    import os
    original = os.environ.get("HERMES_HOME")
    redact.register_vault_redaction_value(CARD)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "other"))
    assert redact.redact_registered_vault_values(CARD) == CARD
    if original is None:
        monkeypatch.delenv("HERMES_HOME")
    else:
        monkeypatch.setenv("HERMES_HOME", original)
    assert redact.redact_registered_vault_values(CARD) == "[hidden]"


def test_nested_list_content_is_scrubbed_in_the_request(monkeypatch):
    redact.register_vault_redaction_value(CARD)
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1",
                        quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent.client = MagicMock()
    agent._cached_system_prompt = "x"
    agent._use_prompt_caching = False
    agent.client.chat.completions.create.return_value = _resp()
    history = [{"role": "user", "content": [{"type": "text", "text": [{"type": "text", "text": CARD}]}]}]
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        agent.run_conversation("go", conversation_history=history)
    assert CARD not in repr(agent.client.chat.completions.create.call_args.kwargs["messages"])
