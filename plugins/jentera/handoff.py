"""Ask another Jentera specialist to do part of this task.

The tool asks the Jentera runner on this machine. The runner runs the named
specialist's turn on its own profile while this turn waits, and returns its
answer. The runner holds every limit — depth, count, time, one at a time —
so this module only carries the request. The contract is aisar-site's
docs/plans/2026-09-26-specialist-handoff.md.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from tools.registry import tool_error, tool_result

HANDOFF_PATH = "/v1/handoff"
AVAILABLE_PATH = "/v1/handoff/available"
USER_AGENT = "Jentera-Agent/1.0"
BRIEF_MAX_CHARS = 2_000
# The runner gives a hand-off at most 390 s. A single tool call has no Hermes
# timeout on the sequential path, so this is the only bound on the wait.
HANDOFF_TIMEOUT_SECONDS = 420
AVAILABLE_TIMEOUT_SECONDS = 3

ASK_SPECIALIST_SCHEMA = {
    "name": "ask_specialist",
    "description": (
        "Hand part of this task to another specialist on this business's Jentera team and wait "
        "for their answer. Name the specialist by its profile key and say exactly what you need. "
        "They work with their own memory and skills and reply to you, not to the owner. Use it "
        "only when that part clearly sits in their remit."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "specialist": {"type": "string", "description": "The specialist's profile key, for example records."},
            "brief": {"type": "string", "description": "What you need from them, in at most 2000 characters."},
        },
        "required": ["specialist", "brief"],
    },
}


def _runner_url() -> str:
    return os.environ.get("JENTERA_RUNNER_URL", "http://127.0.0.1:8080").strip().rstrip("/")


def _runner_key() -> str:
    # The runner and Hermes share the Hermes API key. The default profile's
    # .env puts it in the process environment, which every profile's turn sees.
    return os.environ.get("API_SERVER_KEY", "").strip()


def _request(method: str, path: str, payload: dict | None, timeout: float) -> tuple[int, dict]:
    request = urllib.request.Request(
        _runner_url() + path,
        data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        method=method,
        headers={
            "Authorization": f"Bearer {_runner_key()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read() or b"{}")
        except ValueError:
            return exc.code, {}


def check_handoff_available() -> bool:
    """Offered only on a business whose hand-offs are switched on."""
    if not _runner_key():
        return False
    try:
        status, body = _request("GET", AVAILABLE_PATH, None, AVAILABLE_TIMEOUT_SECONDS)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False
    return status == 200 and body.get("available") is True


def _calling_run_id() -> str:
    # The API server binds the run id as the approval session key of the
    # thread running this turn (gateway/platforms/api_server.py). The runner
    # started that run, so the id identifies the caller exactly.
    from tools.approval import get_current_session_key

    return str(get_current_session_key() or "")


def handle_ask_specialist(args: dict, **_kw) -> str:
    specialist = str(args.get("specialist") or "").strip()
    brief = str(args.get("brief") or "").strip()
    if not specialist or not brief:
        return tool_error("Name the specialist and say what you need from them.", code="invalid")
    if len(brief) > BRIEF_MAX_CHARS:
        return tool_error(f"Keep the brief under {BRIEF_MAX_CHARS} characters.", code="invalid")
    run_id = _calling_run_id()
    if not run_id or not _runner_key():
        return tool_error("Hand-offs are not available here.", code="unavailable")
    try:
        status, body = _request(
            "POST", HANDOFF_PATH, {"runId": run_id, "specialist": specialist, "brief": brief},
            HANDOFF_TIMEOUT_SECONDS,
        )
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return tool_error("The specialist could not be reached. Say in your answer which part is missing.",
                          code="failed")
    if status == 200 and body.get("ok") is True:
        return tool_result(specialist=body.get("specialist", specialist), name=body.get("name", ""),
                           answer=body.get("answer", ""))
    code = body.get("code") if isinstance(body.get("code"), str) else "failed"
    message = body.get("message") if isinstance(body.get("message"), str) else "The specialist could not finish."
    return tool_error(f"{message} Say in your answer which part is missing.", code=code)
