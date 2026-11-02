"""Ask the loopback runner to end this turn with the owner's private sign-in card.

Only a runner scan can grant this request. Failure leaves the bot's existing
page-form path in place; no page text or private values are logged here.
"""

from __future__ import annotations

import json
import threading
import urllib.request

from plugins.jentera.handoff import _runner_key, _runner_url

STEP_IN_PATH = "/v1/signin/step-in"
STEP_IN_NOTE = "Jentera is showing the owner a private sign-in form for this page now. End your turn: write nothing more."
RESEARCH_INTRO = "This site has a sign-in (host {host}). Do not stop for it; see your rules on offering it."
RESPONSE_MAX_BYTES = 64_000
# Never send a loopback request through an environment proxy.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request_step_in(scan_id: str, timeout: float = 3.0) -> bool:
    """True only when the runner answers 200 with literal ``stepped: true``.

    The join bounds the entire operation, including a response that trickles
    bytes without ever reaching the socket's inactivity timeout.
    """
    key = _runner_key()
    if not key or not isinstance(scan_id, str) or not scan_id or timeout <= 0:
        return False
    accepted = {}

    def work():
        try:
            request = urllib.request.Request(
                _runner_url() + STEP_IN_PATH,
                data=json.dumps({"scanId": scan_id}).encode("utf-8"),
                method="POST",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "application/json"},
            )
            with _OPENER.open(request, timeout=timeout) as response:
                if response.status != 200:
                    return
                raw = response.read(RESPONSE_MAX_BYTES + 1)
            if len(raw) > RESPONSE_MAX_BYTES:
                return
            body = json.loads(raw)
            accepted["stepped"] = isinstance(body, dict) and body.get("stepped") is True
        except Exception:
            return

    thread = threading.Thread(target=work, name="jentera-signin-step-in", daemon=True)
    thread.start()
    thread.join(timeout)
    return not thread.is_alive() and accepted.get("stepped") is True
