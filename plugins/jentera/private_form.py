"""Blank the values the runner typed into a page for the owner.

The runner keeps recent filled values in memory and serves them on loopback.
Hermes fetches them and registers them with ``agent.redact`` so nothing the
model is sent can contain them. A failed fetch keeps the last list; values are
never logged.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from agent.redact import (
    clear_vault_redaction_values,
    redact_registered_vault_values,
    register_vault_redaction_value,
)
from plugins.jentera.handoff import _runner_key, _runner_url

REDACTIONS_PATH = "/v1/private-form/redactions"
TIMEOUT_SECONDS = 1.0
MAX_VALUES = 64


def refresh_values() -> bool:
    """Fetch the runner's list and replace this profile's registered values."""
    key = _runner_key()
    if not key:
        return False
    request = urllib.request.Request(
        _runner_url() + REDACTIONS_PATH,
        method="GET",
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            if response.status != 200:
                return False
            body = json.loads(response.read() or b"{}")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False
    values = body.get("values") if isinstance(body, dict) else None
    if body.get("ok") is not True or not isinstance(values, list):
        return False
    clear_vault_redaction_values()
    for value in values[:MAX_VALUES]:
        register_vault_redaction_value(value)
    return True


def _on_pre_llm_call(**_kwargs):
    refresh_values()
    return None


def _on_pre_api_request(**_kwargs):
    refresh_values()


def _on_transform_tool_result(result=None, **_kwargs):
    # A value is typed during a tool call; fetch again before its result (and
    # the next model call) is built, and blank this very result.
    refresh_values()
    if isinstance(result, str):
        scrubbed = redact_registered_vault_values(result)
        if scrubbed != result:
            return scrubbed
    return None


def register_hooks(ctx) -> None:
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("pre_api_request", _on_pre_api_request)
    ctx.register_hook("transform_tool_result", _on_transform_tool_result)
