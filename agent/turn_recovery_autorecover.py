"""Bounded post-exhaustion auto-recovery ladder for a pre-delivery turn (upstream #116353).

Runs from the max-retries branch of ``run_conversation`` once ``max_retries`` is spent AND the
fallback chain has nothing left to move to (fallback stays first). While the provider is only
*temporarily* away (5xx, overloaded/529, "no healthy deployments", connect/read timeouts) and no
answer text has reached the user yet, the turn parks with a visible countdown instead of ending in
"API failed after N retries", then re-enters the ordinary retry loop. Cycles:
``agent.auto_recovery_cycles`` (default 5) with a jittered 15/30/60/60/60 s schedule; a provider
``Retry-After`` wins up to 120 s. Non-retryable classes (auth, format, content policy, billing,
entitlement, context overflow) never enter, and an interrupt cancels the wait within 200 ms.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from agent.error_classifier import FailoverReason

logger = logging.getLogger("agent.conversation_loop")

# Transient transport verdicts: the provider is expected back. Everything else (auth, format,
# content policy, billing, entitlement, overflow, model-not-found ...) is deterministic for this
# request and stays out.
_LADDER_REASONS = frozenset({FailoverReason.overloaded, FailoverReason.server_error, FailoverReason.timeout})

_LADDER_BASE_DELAY_S = 15.0
_LADDER_CAP_S = 60.0
# A provider that names its own cooldown knows better than the schedule, within reason.
_RETRY_AFTER_CAP_S = 120.0

# How the user stops the wait on this surface; the ladder text must say so plainly.
_STOP_HINTS = {
    "cli": "press Esc to stop",
    "tui": "press Esc to stop",
    "desktop": "press Esc to stop",
    "api_server": "cancel the request to stop",
    "cron": "",
}
_DEFAULT_STOP_HINT = "send /stop to cancel"


def auto_recovery_cycles(agent: Any) -> int:
    """Configured ``agent.auto_recovery_cycles`` (0 disables the ladder)."""
    return max(int(getattr(agent, "_auto_recovery_cycles", 0) or 0), 0)


def _parse_retry_after(raw: Any) -> Optional[float]:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _retry_after_seconds(api_error: Any) -> Optional[float]:
    """Provider-declared cooldown from the ``Retry-After`` header or a ``retry_after`` body field."""
    headers = getattr(getattr(api_error, "response", None), "headers", None)
    value = None
    if headers is not None and hasattr(headers, "get"):
        value = _parse_retry_after(headers.get("retry-after") or headers.get("Retry-After"))
    if value is None:
        body = getattr(api_error, "body", None)
        if isinstance(body, dict):
            nested = body.get("error")
            value = _parse_retry_after((nested if isinstance(nested, dict) else body).get("retry_after"))
    return value


def ladder_wait_seconds(cycle: int, api_error: Any) -> float:
    """Wait before recovery ``cycle`` (1-based): jittered 15/30/60/60/60 s, or the provider's
    ``Retry-After`` when present (honoured past the 60 s cap, up to 120 s)."""
    from agent.retry_utils import jittered_backoff
    retry_after = _retry_after_seconds(api_error)
    if retry_after is not None:
        return min(retry_after, _RETRY_AFTER_CAP_S)
    return jittered_backoff(cycle, base_delay=_LADDER_BASE_DELAY_S, max_delay=_LADDER_CAP_S, jitter_ratio=0.2)


def ladder_eligible(agent: Any, classified: Any) -> bool:
    """True when the ladder may engage: cycles configured, a transient transport verdict, and no
    answer text streamed to the user yet (delivered text is never replayed by this path)."""
    if auto_recovery_cycles(agent) <= 0 or classified.reason not in _LADDER_REASONS:
        return False
    streamed = getattr(agent, "_current_streamed_assistant_text", "") or ""
    return not agent._has_content_after_think_block(streamed)


def ladder_notice(agent: Any, *, wait_s: float, cycle: int, total: int) -> str:
    hint = _STOP_HINTS.get(str(getattr(agent, "platform", "") or "").lower(), _DEFAULT_STOP_HINT)
    text = f"⏳ Provider temporarily unavailable — retrying automatically in {wait_s:.0f}s (cycle {cycle}/{total})"
    return f"{text}; {hint}" if hint else text


def _interruptible_wait(agent: Any, wait_s: float, label: str) -> bool:
    """Sleep ``wait_s`` in 200 ms slices. True when an interrupt cut the wait short."""
    end = time.time() + wait_s
    touch = 0
    while time.time() < end:
        if getattr(agent, "_interrupt_requested", False):
            return True
        time.sleep(0.2)
        touch += 1
        if touch % 150 == 0:  # ~30 s: keep the gateway inactivity monitor informed
            try:
                agent._touch_activity(f"{label}, {int(max(end - time.time(), 0))}s remaining")
            except Exception:
                pass
    return False


def auto_recover_after_exhaustion(
    agent: Any, api_error: Any, classified: Any, _retry: Any, *, messages: Any,
    conversation_history: Any, api_call_count: int,
) -> Optional[Dict[str, Any]]:
    """Run one recovery cycle after retries + fallback exhausted. Returns ``{"action": "continue"}``
    when the wait completed (caller zeroes ``retry_count`` and re-enters the retry loop),
    ``{"action": "return", "result": …}`` when the user interrupted, and ``None`` when the ladder
    does not apply or is spent."""
    if not ladder_eligible(agent, classified):
        return None
    total = auto_recovery_cycles(agent)
    used = int(getattr(_retry, "auto_recovery_cycles_used", 0) or 0)
    if used >= total:
        agent._emit_status(
            f"⏳ Automatic recovery gave up after {total} cycles — the provider is still unavailable."
        )
        return None
    cycle = used + 1
    _retry.auto_recovery_cycles_used = cycle
    wait_s = ladder_wait_seconds(cycle, api_error)
    notice = ladder_notice(agent, wait_s=wait_s, cycle=cycle, total=total)
    # Durable line on every surface (CLI print, gateway status_callback) plus the live wait line.
    agent._emit_status(notice)
    agent._emit_wait_notice(notice)
    logger.warning(
        "%sProvider unavailable (%s) — auto-recovery cycle %d/%d, retrying in %.0fs %s",
        agent.log_prefix, classified.reason.value, cycle, total, wait_s, agent._client_log_context(),
    )
    if _interruptible_wait(agent, wait_s, f"auto-recovery wait ({cycle}/{total})"):
        from agent.message_sanitization import close_interrupted_tool_sequence
        agent._vprint(
            f"{agent.log_prefix}⚡ Interrupt detected during automatic recovery wait, aborting.", force=True
        )
        interrupt_text = f"Operation interrupted: waiting for the provider to recover (cycle {cycle}/{total})."
        close_interrupted_tool_sequence(messages, interrupt_text)
        agent._persist_session(messages, conversation_history)
        agent.clear_interrupt()
        return {
            "action": "return",
            "result": {
                "final_response": interrupt_text,
                "messages": messages,
                "api_calls": api_call_count,
                "completed": False,
                "interrupted": True,
            },
        }
    return {"action": "continue"}
