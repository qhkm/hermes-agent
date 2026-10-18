"""Turn liveness watchdog (upstream #99758, #95548): abort turns that stall silently.

A conversation turn can stall mid-flight with no error and no further
progress, leaving the session "busy" until the process is killed. The
watchdog samples the agent's activity clock (``_last_activity_ts``, stamped
by ``AIAgent._touch_activity``) and, when a turn shows no observable
progress for ``agent.turn_liveness.timeout_s``, interrupts it so it unwinds
and frees its session.

Config surface (config.yaml)::

    agent:
      turn_liveness:
        timeout_s: 600.0   # idle bound; <= 0 disables the watchdog
        poll_s: 15.0       # sampling interval (seconds)

Invalid values (typo, NaN, Inf, non-positive poll) warn and fall back to the
defaults; they never crash startup or silently disable the watchdog.

Race safety: the abort is bound to the sampled ``(generation, timestamp)``
pair and revalidated at the commit point under the lock shared with
``_touch_activity``, so a turn that resumed while the stall was being
surfaced is never aborted.

Jentera port: this fork has no durable turn lease, so the upstream lease
deactivation and the ``require_generation`` interrupt claim are omitted; the
revalidation and the interrupt both happen inside the activity lock instead.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_TURN_LIVENESS_TIMEOUT_S = 600.0
DEFAULT_TURN_LIVENESS_POLL_S = 15.0
MIN_TURN_LIVENESS_POLL_S = 0.01

_CONFIG_TIMEOUT_KEY = "agent.turn_liveness.timeout_s"
_CONFIG_POLL_KEY = "agent.turn_liveness.poll_s"


class ActivitySnapshot(NamedTuple):
    """One observation of the activity clock, bound to an abort decision.

    ``generation`` + ``activity_ts`` uniquely identify the observed stamp.
    The commit callback must revalidate this pair under the lock shared
    with ``AIAgent._touch_activity``; if it no longer matches, the
    observation is stale and the abort must be declined.
    """

    generation: int
    activity_ts: Optional[float]
    idle_seconds: float


def _warn_invalid_value(key: str, raw: Any, default: float) -> None:
    logger.warning(
        "Invalid %s in config.yaml: %r — falling back to default %.1f.",
        key,
        raw,
        default,
    )


def _resolve_finite_seconds(raw: Any, *, default: float, key: str) -> float:
    """Coerce one duration knob, rejecting typos, NaN and Inf.

    A non-numeric value must not raise into durable-turn startup, and a
    non-finite value must not silently change behavior (``NaN`` would
    disable the timeout via the ``> 0`` comparison; ``Inf`` would freeze
    the poll loop in ``Event.wait``).
    """
    try:
        value = float(raw)
    except (TypeError, ValueError):
        _warn_invalid_value(key, raw, default)
        return default
    if not math.isfinite(value):
        _warn_invalid_value(key, raw, default)
        return default
    return value


def resolve_turn_liveness_settings(
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[float], float]:
    """Resolve ``(timeout_s, poll_s)`` from the ``agent.turn_liveness`` section.

    Precedence: explicit config.yaml value wins over the default; any
    invalid value (typo, NaN, Inf, non-positive poll) falls back to the
    default with a warning. ``timeout_s <= 0`` is the documented opt-out
    and yields ``(None, poll_s)`` — the caller then never arms the
    watchdog. The resolver never raises.
    """
    section: Dict[str, Any] = {}
    if isinstance(config, dict):
        agent_cfg = config.get("agent")
        if isinstance(agent_cfg, dict):
            raw_section = agent_cfg.get("turn_liveness")
            if isinstance(raw_section, dict):
                section = raw_section
            elif raw_section is not None:
                _warn_invalid_value(
                    "agent.turn_liveness", raw_section, DEFAULT_TURN_LIVENESS_TIMEOUT_S
                )

    timeout_s = _resolve_finite_seconds(
        section.get("timeout_s", DEFAULT_TURN_LIVENESS_TIMEOUT_S),
        default=DEFAULT_TURN_LIVENESS_TIMEOUT_S,
        key=_CONFIG_TIMEOUT_KEY,
    )
    poll_s = _resolve_finite_seconds(
        section.get("poll_s", DEFAULT_TURN_LIVENESS_POLL_S),
        default=DEFAULT_TURN_LIVENESS_POLL_S,
        key=_CONFIG_POLL_KEY,
    )
    if poll_s <= 0:
        _warn_invalid_value(_CONFIG_POLL_KEY, poll_s, DEFAULT_TURN_LIVENESS_POLL_S)
        poll_s = DEFAULT_TURN_LIVENESS_POLL_S
    # <= 0 is the documented opt-out, not an error.
    if timeout_s <= 0:
        timeout_s = None
    return timeout_s, poll_s


class TurnLivenessWatchdog:
    """Sampled-idle watchdog thread bound to one conversation turn.

    ``run_agent.py`` owns the turn state (stop event, turn-active flag,
    interrupt plumbing); this class only reads the activity clock and calls
    back at the stall commit point. All synchronization with
    ``AIAgent._touch_activity`` goes through ``activity_lock``, which must be
    the SAME lock the agent stamps its activity clock with.
    """

    def __init__(
        self,
        agent: Any,
        *,
        session_id: Optional[str],
        timeout_s: float,
        poll_s: float,
        stop_event: threading.Event,
        activity_lock: threading.Lock,
        is_turn_active: Callable[[], bool],
        commit_abort: Callable[[ActivitySnapshot, str], bool],
    ) -> None:
        self._agent = agent
        self._session_id = session_id
        self._timeout_s = float(timeout_s)
        self._poll_s = max(MIN_TURN_LIVENESS_POLL_S, float(poll_s))
        self._stop_event = stop_event
        self._activity_lock = activity_lock
        self._is_turn_active = is_turn_active
        self._commit_abort = commit_abort
        self._last_surfaced_generation: Optional[int] = None

    def make_thread(self) -> threading.Thread:
        """Build the (not yet started) watcher thread."""
        return threading.Thread(
            target=self._watch,
            name="turn-liveness-watchdog",
            daemon=True,
        )

    def _watch(self) -> None:
        while not self._stop_event.wait(self._poll_s):
            snapshot = self._sample()
            if snapshot is None:
                # Turn is no longer active; nothing to watch.
                return
            if snapshot.idle_seconds < self._timeout_s:
                continue
            self._surface_stall(snapshot)
            # Commit point: revalidate the sampled generation under the lock
            # shared with `_touch_activity`. If progress resumed meanwhile the
            # turn continues and sampling resumes.
            if not self._commit_abort(snapshot, self._abort_message(snapshot)):
                continue
            self._surface_committed_abort(snapshot)
            return

    def _sample(self) -> Optional[ActivitySnapshot]:
        with self._activity_lock:
            if not self._is_turn_active():
                return None
            generation = getattr(
                self._agent, "_turn_liveness_activity_generation", 0
            )
            activity_ts = getattr(self._agent, "_last_activity_ts", None)
        now = time.time()
        if activity_ts is None:
            idle_seconds = 0.0
        else:
            idle_seconds = max(0.0, now - activity_ts)
        return ActivitySnapshot(
            generation=generation,
            activity_ts=activity_ts,
            idle_seconds=idle_seconds,
        )

    def _abort_message(self, snapshot: ActivitySnapshot) -> str:
        return (
            f"Turn made no progress for {int(snapshot.idle_seconds)}s; "
            "aborting to release the session."
        )

    def _surface_stall(self, snapshot: ActivitySnapshot) -> None:
        """Log the stall loudly (observational; does not claim an abort).

        Rate-limited per activity generation so a turn whose aborts keep
        declining does not re-log every poll.
        """
        generation = snapshot.generation
        if self._last_surfaced_generation == generation:
            return
        self._last_surfaced_generation = generation
        session_id = getattr(self._agent, "session_id", None) or self._session_id
        logger.error(
            "Turn liveness watchdog fired for session %s: "
            "no progress for %.1fs (last activity: %r). "
            "Attempting recovery by interrupting the turn (#95548).",
            session_id,
            snapshot.idle_seconds,
            getattr(self._agent, "_last_activity_desc", None),
        )

    def _surface_committed_abort(self, snapshot: ActivitySnapshot) -> None:
        """Publish the definitive settlement after the interrupt committed."""
        session_id = getattr(self._agent, "session_id", None) or self._session_id
        logger.error(
            "Turn liveness watchdog aborted turn for session %s: "
            "no progress for %.1fs; turn interrupted (#95548).",
            session_id,
            snapshot.idle_seconds,
        )
        emit_warning = getattr(self._agent, "_emit_warning", None)
        if not callable(emit_warning):
            return
        try:
            emit_warning(
                "Turn aborted by the liveness watchdog "
                f"({int(snapshot.idle_seconds)}s without activity). "
                "You can retry your message."
            )
        except Exception:
            logger.debug("Failed to emit committed-abort warning", exc_info=True)
