"""Turn liveness watchdog (upstream #99758 / #95548), Jentera port.

A turn that makes no observable progress for ``agent.turn_liveness.timeout_s``
is interrupted, ends as a failure, and leaves the agent usable for the next
turn on the same session.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

from agent import turn_liveness
from run_agent import AIAgent


def _make_agent() -> AIAgent:
    """Bare AIAgent double: only what the turn wrapper and watchdog touch."""
    agent = AIAgent.__new__(AIAgent)
    agent.session_id = "sess-liveness"
    agent._interrupt_requested = False
    agent._interrupt_message = None
    agent._pending_redirect = None
    agent._last_activity_ts = time.time()
    agent._last_activity_desc = "idle"
    agent.api_mode = "chat_completions"
    agent._emit_warning = lambda *_a, **_k: None
    agent._conversation_root_id = lambda: "root"
    return agent


def _cfg(timeout_s, poll_s=0.02):
    return {"agent": {"turn_liveness": {"timeout_s": timeout_s, "poll_s": poll_s}}}


def _fake_loop(agent, stall_turns):
    """Conversation loop double: stalls (until interrupted) on listed turns."""
    calls = {"n": 0}

    def _loop(agent_, user_message, *a, **k):
        calls["n"] += 1
        agent_._interrupt_requested = False  # turn_context resets at turn start
        if calls["n"] in stall_turns:
            deadline = time.time() + 10
            while not agent_._interrupt_requested and time.time() < deadline:
                time.sleep(0.01)
            return {
                "final_response": "Operation interrupted.",
                "interrupted": True,
                "completed": False,
                "failed": False,
            }
        return {"final_response": f"ok:{user_message}", "completed": True}

    return _loop, calls


def _run(agent, loop, msg, cfg):
    with patch("agent.conversation_loop.run_conversation", loop), patch(
        "hermes_cli.config.load_config", return_value=cfg
    ):
        return agent.run_conversation(msg)


def test_stalled_turn_ends_as_failure_and_next_turn_works():
    agent = _make_agent()
    loop, calls = _fake_loop(agent, stall_turns={1})
    cfg = _cfg(0.15)

    started = time.time()
    first = _run(agent, loop, "stall me", cfg)
    assert time.time() - started < 5
    assert first["failed"] is True
    assert first["completed"] is False
    assert first["turn_liveness_aborted"] is True
    assert "no progress" in first["error"]

    # The session is free: the very next turn on the same agent succeeds.
    second = _run(agent, loop, "again", cfg)
    assert second["final_response"] == "ok:again"
    assert not second.get("failed")
    assert not second.get("turn_liveness_aborted")
    assert calls["n"] == 2
    assert agent._interrupt_requested is False


def test_progressing_turn_is_not_aborted():
    agent = _make_agent()

    def _loop(agent_, user_message, *a, **k):
        # Slower than the bound overall, but touches the clock regularly.
        for _ in range(15):
            time.sleep(0.04)
            agent_._touch_activity("working")
        return {"final_response": "done", "completed": True}

    result = _run(agent, _loop, "x", _cfg(0.2))
    assert result["final_response"] == "done"
    assert not result.get("failed")
    assert not agent._interrupt_requested


def test_disabled_when_timeout_non_positive():
    agent = _make_agent()
    loop, _ = _fake_loop(agent, stall_turns=set())
    with patch("hermes_cli.config.load_config", return_value=_cfg(0)):
        assert agent._start_turn_liveness_watchdog() == (None, None)
    result = _run(agent, loop, "x", _cfg(0))
    assert result["final_response"] == "ok:x"


def test_turn_entry_stamps_clock_so_prior_idle_is_not_charged():
    agent = _make_agent()
    agent._last_activity_ts = time.time() - 3600  # idle for an hour between turns

    def _loop(agent_, user_message, *a, **k):
        time.sleep(0.1)  # several polls, shorter than the bound
        return {"final_response": "fine", "completed": True}

    result = _run(agent, _loop, "x", _cfg(5.0))
    assert result["final_response"] == "fine"
    assert not result.get("failed")


def test_commit_declines_when_turn_resumed_after_sample():
    agent = _make_agent()
    stop, thread = agent._arm_turn_liveness_watchdog(3600.0, 3600.0)
    try:
        snapshot = turn_liveness.ActivitySnapshot(
            generation=agent._turn_liveness_activity_generation,
            activity_ts=agent._last_activity_ts,
            idle_seconds=9999.0,
        )
        agent._touch_activity("resumed")  # progress lands after the sample
        # Re-create the commit seam via a fresh watchdog sharing the lock.
        committed = []
        wd = turn_liveness.TurnLivenessWatchdog(
            agent,
            session_id="s",
            timeout_s=1.0,
            poll_s=0.01,
            stop_event=threading.Event(),
            activity_lock=agent._liveness_activity_lock(),
            is_turn_active=lambda: True,
            commit_abort=lambda snap, msg: committed.append(snap) or False,
        )
        assert wd._sample().generation == agent._turn_liveness_activity_generation
        assert snapshot.generation != agent._turn_liveness_activity_generation
    finally:
        agent._stop_turn_liveness_watchdog(stop, thread)
    assert agent._interrupt_requested is False
    assert getattr(agent, "_turn_liveness_aborted", None) is None
