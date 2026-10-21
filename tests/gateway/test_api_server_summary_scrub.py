"""The iteration-limit summary is scrubbed on the wire, not only in unit tests.

A real adapter and ``AIAgent`` against the in-process mock provider of
``test_api_server_turn_notes``: one iteration and a tool call whose argument
and result both carry a value the runner typed for the owner, so the turn ends
by asking the model for a toolless summary. With the Jentera scrubber
installed, that second request must carry the replacement, never the value.
"""

import importlib
import json

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway.test_api_server_turn_notes import (  # noqa: F401  (provider is a fixture)
    _app,
    _chat,
    _run,
    _text,
    _tool_call,
    _wire_adapter,
    provider,
)

SECRET = "s3cret-card-4111111111111111"


class _Redact:
    """``agent.redact`` as it is when used (see tests/agent/test_redact_summary.py)."""

    def __getattr__(self, name):
        return getattr(importlib.import_module("agent.redact"), name)


redact = _Redact()


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    # The Jentera plugin installs its own scrubber whenever plugins are
    # discovered, and the first agent built in a process triggers that. Keep it
    # from replacing the one a test installs.
    monkeypatch.setattr("plugins.jentera.private_form.set_model_text_scrubber", lambda scrubber: None)
    redact.set_model_text_scrubber(None)
    yield
    redact.set_model_text_scrubber(None)


def _hide(texts):
    return [t.replace(SECRET, "[hidden]") for t in texts]


async def _force_summary(monkeypatch, tmp_path, provider):
    """Run a turn that hits the iteration limit; return (status, chat requests)."""
    note = tmp_path / f"{SECRET}.txt"
    note.write_text(f"paid with {SECRET}\n")
    adapter, db = _wire_adapter(monkeypatch, tmp_path, provider, toolsets={"file"}, max_iterations=1)
    provider.queue = [
        _tool_call("read_file", json.dumps({"path": str(note)})),
        _text("summary"),
    ]
    try:
        async with TestClient(TestServer(_app(adapter))) as cli:
            status = await _run(cli, input="hello", instructions="RULES", session_id="sess-scrub")
    finally:
        db.close()
    return status, _chat(provider)


def _tool_text(request):
    return "".join(m["content"] for m in request["messages"] if m["role"] == "tool")


def _arguments(request):
    return [c["function"]["arguments"] for m in request["messages"] for c in m.get("tool_calls") or []]


@pytest.mark.asyncio
async def test_the_summary_request_on_the_wire_carries_the_replacement_not_the_value(monkeypatch, tmp_path, provider):
    redact.set_model_text_scrubber(_hide)
    status, reqs = await _force_summary(monkeypatch, tmp_path, provider)
    assert status["status"] == "completed" and status["output"].startswith("summary")
    assert len(reqs) == 2
    summary = reqs[1]
    assert "maximum number of tool-calling iterations" in summary["messages"][-1]["content"]
    assert SECRET not in json.dumps(summary)
    # The tool result was blanked ...
    assert "[hidden]" in _tool_text(summary)
    # ... and so was the argument the model itself wrote.
    assert [json.loads(a)["path"].endswith("[hidden].txt") for a in _arguments(summary)] == [True]


@pytest.mark.asyncio
async def test_a_scrubber_that_breaks_withholds_the_summary_request_text(monkeypatch, tmp_path, provider):
    def broken(_texts):
        raise RuntimeError("runner unreachable")

    redact.set_model_text_scrubber(broken)
    status, reqs = await _force_summary(monkeypatch, tmp_path, provider)
    assert status["status"] == "completed"
    summary = reqs[1]
    assert SECRET not in json.dumps(summary)
    assert _tool_text(summary) == redact.MODEL_TEXT_CHECK_FAILED
    assert [json.loads(a) for a in _arguments(summary)] == [{"hidden": redact.MODEL_TEXT_CHECK_FAILED}]


@pytest.mark.asyncio
async def test_without_a_scrubber_the_summary_request_is_unchanged(monkeypatch, tmp_path, provider):
    status, reqs = await _force_summary(monkeypatch, tmp_path, provider)
    assert status["status"] == "completed"
    summary = reqs[1]
    assert SECRET in _tool_text(summary)
    assert redact.MODEL_TEXT_CHECK_FAILED not in json.dumps(summary)
    assert "[hidden]" not in json.dumps(summary)
    assert [SECRET in json.loads(a)["path"] for a in _arguments(summary)] == [True]
