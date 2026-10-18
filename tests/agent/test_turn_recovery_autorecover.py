"""Bounded post-exhaustion auto-recovery ladder (upstream #116353): transient outages only,
visible via the status rails, interrupt cancels. Our retry loop is inline in conversation_loop, so
these exercise ``auto_recover_after_exhaustion`` and the classifier verdicts that feed it."""
import time
from types import SimpleNamespace

import pytest

from agent import turn_recovery_autorecover as ar
from agent.error_classifier import FailoverReason, classify_api_error
from agent.turn_recovery_autorecover import auto_recover_after_exhaustion, ladder_wait_seconds
from agent.turn_retry_state import TurnRetryState


class _Err(Exception):
    def __init__(self, status_code, message, headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.response = SimpleNamespace(headers=headers or {})
        self.body = None


class _Agent:
    log_prefix = ""
    provider = "custom"
    _current_streamed_assistant_text = ""
    _interrupt_requested = False

    def __init__(self, cycles=5, platform="cli"):
        self._auto_recovery_cycles = cycles
        self.platform = platform
        self.statuses, self.waits, self.persisted, self.cleared = [], [], [], []

    def _has_content_after_think_block(self, text):
        return bool(text.strip())

    def _emit_status(self, text):
        self.statuses.append(str(text))

    def _emit_wait_notice(self, text):
        self.waits.append(str(text))

    def _client_log_context(self):
        return ""

    def _vprint(self, *a, **k):
        pass

    def _touch_activity(self, desc):
        pass

    def _persist_session(self, messages, history):
        self.persisted.append(True)

    def clear_interrupt(self, **kw):
        self.cleared.append(True)


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    monkeypatch.setattr(ar, "_interruptible_wait", lambda agent, wait_s, label: bool(agent._interrupt_requested))


def _run(agent, err, retry=None):
    retry = retry or TurnRetryState()
    classified = classify_api_error(err, provider=agent.provider)
    out = auto_recover_after_exhaustion(
        agent, err, classified, retry, messages=[], conversation_history=None, api_call_count=1,
    )
    return out, retry


@pytest.mark.parametrize("status,message", [
    (503, "Service temporarily unavailable"),
    (529, "Overloaded"),
    (500, "Internal server error"),
])
def test_transient_exhaustion_enters_ladder(status, message):
    agent = _Agent()
    out, retry = _run(agent, _Err(status, message))
    assert out == {"action": "continue"} and retry.auto_recovery_cycles_used == 1
    assert agent.statuses == agent.waits and len(agent.statuses) == 1
    line = agent.statuses[0]
    assert "retrying automatically in" in line and "(cycle 1/5)" in line and "press Esc to stop" in line


def test_litellm_no_healthy_deployments_400_is_transient():
    """LiteLLM raises this as a 400 BadRequestError; it is a down backend, not a bad request."""
    err = _Err(400, "litellm.BadRequestError: OpenAIException - There are no healthy deployments "
                    "available for this model. Try again later.")
    classified = classify_api_error(err, provider="custom")
    assert classified.reason == FailoverReason.overloaded and classified.retryable
    out, retry = _run(_Agent(), err)
    assert out == {"action": "continue"} and retry.auto_recovery_cycles_used == 1


@pytest.mark.parametrize("status,message", [
    (401, "Incorrect API key provided"),
    (403, "forbidden"),
    (400, "Invalid request: messages must be non-empty"),
    (400, "This model's maximum context length is 8192 tokens, you requested 9000 tokens"),
    (404, "model not found"),
])
def test_deterministic_errors_never_enter_ladder(status, message):
    agent = _Agent()
    out, retry = _run(agent, _Err(status, message))
    assert out is None and retry.auto_recovery_cycles_used == 0 and agent.statuses == []


def test_no_healthy_deployments_does_not_mask_auth_status():
    classified = classify_api_error(_Err(401, "no healthy deployments (bad key)"), provider="custom")
    assert classified.reason != FailoverReason.overloaded


def test_ladder_is_bounded_and_gives_up():
    agent = _Agent()
    out, _ = _run(agent, _Err(503, "down"), retry=TurnRetryState(auto_recovery_cycles_used=5))
    assert out is None
    assert any("gave up after 5 cycles" in s for s in agent.statuses)


def test_streamed_text_and_disabled_config_never_enter():
    streamed = _Agent()
    streamed._current_streamed_assistant_text = "Here is the first half of the answer"
    out, retry = _run(streamed, _Err(503, "down"))
    assert out is None and retry.auto_recovery_cycles_used == 0
    off = _Agent(cycles=0)
    out, retry = _run(off, _Err(503, "down"))
    assert out is None and off.statuses == []


def test_schedule_honours_retry_after_and_platform_stop_hint(monkeypatch):
    import agent.retry_utils as retry_utils
    monkeypatch.setattr(
        retry_utils, "jittered_backoff",
        lambda attempt, *, base_delay, max_delay, jitter_ratio: min(base_delay * 2 ** (attempt - 1), max_delay),
    )
    plain = _Err(503, "down")
    assert [ladder_wait_seconds(c, plain) for c in (1, 2, 3, 4, 5)] == [15, 30, 60, 60, 60]
    assert ladder_wait_seconds(1, _Err(529, "Overloaded", {"Retry-After": "4"})) == 4
    assert ladder_wait_seconds(1, _Err(529, "Overloaded", {"Retry-After": "90"})) == 90
    assert ladder_wait_seconds(1, _Err(529, "Overloaded", {"Retry-After": "900"})) == 120

    tg = _Agent(platform="telegram")
    _run(tg, plain)
    assert "send /stop to cancel" in tg.statuses[0]
    cron = _Agent(platform="cron")
    _run(cron, plain)
    assert cron.statuses[0].endswith("(cycle 1/5)")


def test_worst_case_default_wait_is_bounded():
    # Real jitter: 5 cycles, each at most 60 s * 1.2 -> well under 6 minutes without Retry-After.
    total = sum(ladder_wait_seconds(c, _Err(503, "down")) for c in range(1, 6))
    assert total <= 5 * 60 * 1.2


def test_interrupt_during_wait_returns_interrupted_result():
    agent = _Agent()
    agent._interrupt_requested = True
    out, retry = _run(agent, _Err(503, "down"))
    assert out["action"] == "return" and out["result"]["interrupted"] is True
    assert out["result"]["completed"] is False
    assert agent.cleared and agent.persisted and retry.auto_recovery_cycles_used == 1


def test_real_wait_stops_within_a_slice_on_interrupt(monkeypatch):
    monkeypatch.undo()
    agent = _Agent()
    agent._interrupt_requested = True
    started = time.time()
    assert ar._interruptible_wait(agent, 60, "x") is True
    assert time.time() - started < 1


def test_max_seconds_cap_blocks_second_cycle(monkeypatch):
    monkeypatch.setattr(ar, "ladder_wait_seconds", lambda cycle, err: 15.0 if cycle == 1 else 30.0)
    agent = _Agent()
    agent._auto_recovery_max_seconds = 40.0
    agent._inflight_turn_started = time.time()
    retry = TurnRetryState()
    out, retry = _run(agent, _Err(503, "down"), retry=retry)
    assert out == {"action": "continue"} and retry.auto_recovery_cycles_used == 1
    agent._inflight_turn_started -= 15.0  # the first (mocked) wait elapsed
    out, retry = _run(agent, _Err(503, "down"), retry=retry)  # now + 30 > start + 40: refused
    assert out is None and retry.auto_recovery_cycles_used == 1
    assert any("gave up" in s and "recovery budget" in s for s in agent.statuses)


def test_absent_max_seconds_keeps_behaviour(monkeypatch):
    monkeypatch.setattr(ar, "ladder_wait_seconds", lambda cycle, err: 60.0)
    agent = _Agent()
    agent._inflight_turn_started = time.time() - 100000  # ancient turn, but no cap configured
    retry = TurnRetryState()
    for expected in (1, 2, 3):
        out, retry = _run(agent, _Err(503, "down"), retry=retry)
        assert out == {"action": "continue"} and retry.auto_recovery_cycles_used == expected
