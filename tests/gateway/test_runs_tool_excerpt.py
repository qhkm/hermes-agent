"""Safe result excerpts on the real /v1/runs event callback."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from agent import redact
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from plugins.jentera import private_form


@pytest.fixture(autouse=True)
def isolate_scrubber(monkeypatch):
    monkeypatch.setattr(redact, "_MODEL_TEXT_SCRUBBER", None)


def run_events(*calls):
    async def collect():
        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        queue = asyncio.Queue()
        adapter._run_streams["r1"] = queue
        callback = adapter._make_run_event_callback("r1", asyncio.get_running_loop())
        for positional, kwargs in calls:
            callback(*positional, **kwargs)
        await asyncio.sleep(0)  # drain call_soon_threadsafe deliveries
        return [queue.get_nowait() for _ in range(queue.qsize())]

    return asyncio.run(collect())


def completed(tool, **kwargs):
    return ("tool.completed", tool, None, None), kwargs


def started(tool, path, **kwargs):
    return ("tool.started", tool, None, {"path": path}), kwargs


def test_web_extract_completed_carries_excerpt():
    event, = run_events(completed("web_extract", result="x" * 3000, is_error=False))
    assert event["result_excerpt"] == "x" * 1500


@pytest.mark.parametrize("tool", [
    "terminal", "execute_code", "browser_navigate", "browser_snapshot", "browser_type",
    "browser_console", "browser_vision", "vision_analyze", "business_records",
    "connect_service", "site_read", "ask_specialist", "delegate_task", "private_form",
])
def test_terminal_completed_has_no_excerpt(tool):
    event, = run_events(completed(tool, result="private", is_error=False))
    assert "result_excerpt" not in event
    assert "result_path" not in event


def test_error_result_has_no_excerpt():
    event, = run_events(completed("web_search", result="private error", is_error=True))
    assert "result_excerpt" not in event


@pytest.mark.parametrize("tool", ["write_file", "patch"])
def test_write_file_carries_path(tool):
    path = "/home/sprite/aisar/outputs/t/q.csv"
    events = run_events(started(tool, "/old.csv"), started(tool, path),
                        completed(tool, result="Added item, quantity, price and total."))
    assert events[-1]["result_path"] == path
    assert events[-1]["result_excerpt"] == "Added item, quantity, price and total."
    assert "result_excerpt" not in events[0]


def test_excerpt_is_blanked_while_private_form_live(monkeypatch):
    monkeypatch.setattr(private_form, "is_active", lambda: (True, True))
    monkeypatch.setattr(private_form, "_blank", lambda texts: [
        text.replace("4111111111111111", "[hidden]") for text in texts
    ])
    private_form.register_hooks(SimpleNamespace(register_hook=lambda *args: None))
    event, = run_events(completed("web_extract", result="4111111111111111"))
    assert event["result_excerpt"] == "[hidden]"


@pytest.mark.parametrize("failure", ["raises", "invalid", "plugin_failure"])
def test_scrubber_failure_omits_excerpt(monkeypatch, failure):
    def broken(texts):
        raise RuntimeError("blanker unavailable")

    if failure == "plugin_failure":
        monkeypatch.setattr(private_form, "is_active", lambda: (True, True))
        monkeypatch.setattr(private_form, "_blank", broken)
        private_form.register_hooks(SimpleNamespace(register_hook=lambda *args: None))
    else:
        redact.set_model_text_scrubber(broken if failure == "raises" else lambda texts: None)
    event, = run_events(completed("web_search", result="4111111111111111"))
    assert "result_excerpt" not in event


def test_scrubs_before_cap_and_caps_after_scrubbing():
    redact.set_model_text_scrubber(lambda texts: [
        text.replace("4111111111111111", "[hidden]") for text in texts
    ])
    event, = run_events(completed("web_extract", result="x" * 1495 + "4111111111111111"))
    assert event["result_excerpt"] == "x" * 1495 + "[hidd"
    redact.set_model_text_scrubber(lambda texts: ["[hidden]" * 300])
    event, = run_events(completed("web_extract", result="x"))
    assert len(event["result_excerpt"]) == 1500


def test_non_string_result_is_json():
    event, = run_events(completed("web_search", result={"found": 2, "items": {"one", "two"}}))
    assert json.loads(event["result_excerpt"])["found"] == 2


def test_missing_result_is_omitted():
    event, = run_events(completed("web_search"))
    assert "result_excerpt" not in event


def test_tool_call_id_matches_out_of_order_file_completions():
    events = run_events(
        started("write_file", "/first.csv", tool_call_id="c1"),
        started("write_file", "/second.csv", tool_call_id="c2"),
        completed("write_file", result="second", tool_call_id="c2"),
        completed("write_file", result="first", tool_call_id="c1"),
    )
    assert events[0]["tool_call_id"] == "c1"
    assert events[-2]["result_path"] == "/second.csv"
    assert events[-1]["result_path"] == "/first.csv"
    assert events[-1]["tool_call_id"] == "c1"


def test_unknown_call_id_does_not_borrow_another_files_path():
    events = run_events(started("write_file", "/other.csv", tool_call_id="c1"),
                        completed("write_file", result="ok", tool_call_id="unknown"))
    assert "result_path" not in events[-1]
