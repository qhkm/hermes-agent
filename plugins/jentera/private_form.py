"""Blank the values the runner typed into a page for the owner.

The runner keeps recent filled values in memory and serves them on loopback.
Hermes fetches them and registers them with ``agent.redact`` so nothing the
model is sent can contain them. A failed fetch keeps the last list; values are
never logged.
"""

from __future__ import annotations

import json
import urllib.request

from agent.redact import redact_registered_vault_values, replace_vault_redaction_values
from plugins.jentera.handoff import _runner_key, _runner_url

REDACTIONS_PATH = "/v1/private-form/redactions"
TIMEOUT_SECONDS = 1.0
MAX_VALUES = 64
# Shorter strings (a 3-digit CVV) would blank ordinary text everywhere.
MIN_VALUE_LENGTH = 8

# Loopback only: never route through an environment proxy.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def refresh_values() -> bool:
    """Fetch the runner's list and replace this profile's registered values."""
    key = _runner_key()
    if not key:
        return False
    try:
        request = urllib.request.Request(
            _runner_url() + REDACTIONS_PATH,
            method="GET",
            headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
        )
        with _OPENER.open(request, timeout=TIMEOUT_SECONDS) as response:
            if response.status != 200:
                return False
            body = json.loads(response.read() or b"{}")
        if not isinstance(body, dict) or body.get("ok") is not True:
            return False
        values = body.get("values")
        if not isinstance(values, list):
            return False
        replace_vault_redaction_values(
            [v for v in values[:MAX_VALUES] if isinstance(v, str) and len(v) >= MIN_VALUE_LENGTH]
        )
        return True
    except Exception:
        return False


def _on_pre_llm_call(**_kwargs):
    refresh_values()
    return None


def _on_transform_tool_result(result=None, **_kwargs):
    # A value is typed during a tool call; fetch again so the next model call
    # sees it, and blank this very result with whatever list we hold now.
    refresh_values()
    if isinstance(result, str):
        scrubbed = redact_registered_vault_values(result)
        if scrubbed != result:
            return scrubbed
    return None


def register_hooks(ctx) -> None:
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    # First string returned wins across plugins, so a plugin ahead of this one
    # can bypass the blank here; the send-time scrub in conversation_loop is
    # the backstop.
    ctx.register_hook("transform_tool_result", _on_transform_tool_result)
