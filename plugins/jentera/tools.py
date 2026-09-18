"""Read the business's connected records through Jentera's control plane.

The point of this plugin is what it does *not* do. An accounting token
opens quotations, invoices, payments and every customer record, so it is
never delivered to this machine: the agent asks Jentera for a reading, and
Jentera — which holds the credential — makes the call and answers in
words. Since `web_extract` pulls arbitrary pages into this agent's
context, a credential here is one a stranger's page could ask it to
repeat. There is nothing to repeat.

The endpoint and the credential are already present for the model channel;
nothing new is transferred at bootstrap to make this work.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from urllib.parse import urlparse

from tools.registry import tool_error, tool_result

CONNECTOR_PATH = "/v1/runtime/connector"
REQUEST_TIMEOUT_SECONDS = 20
# Jentera answers in a sentence, not a dataset. A cap here stops an
# unexpected body from becoming the bulk of the model's context.
MAX_DETAIL_CHARS = 8_000


def _api_base() -> str:
    """Where Jentera's control plane is, from what the sprite already holds.

    `OPENROUTER_BASE_URL` points at the model proxy on the same origin, so
    the origin is taken from it rather than transferred separately — a new
    bootstrap field is the one change that strands a fleet when its arm and
    its sender ship out of order.
    """
    explicit = os.environ.get("JENTERA_API_BASE", "").strip()
    if explicit:
        return explicit.rstrip("/")
    model_base = os.environ.get("OPENROUTER_BASE_URL", "").strip()
    if not model_base:
        return ""
    parsed = urlparse(model_base)
    if parsed.scheme != "https" or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}"


def _credential() -> str:
    return os.environ.get("OPENROUTER_API_KEY", "").strip()


def check_available() -> bool:
    """Registered either way, so `hermes tools` lists it; dispatch is gated."""
    return bool(_api_base() and _credential())


BUSINESS_RECORDS_SCHEMA = {
    "name": "business_records",
    "description": (
        "Read the business's own records from the systems its owner has connected — "
        "unpaid or overdue invoices, and customer contacts. Use this instead of guessing "
        "or asking the owner to look something up. Returns a short summary, not a dataset. "
        "Reading only: it cannot create, send, or change anything."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "resource": {
                "type": "string",
                "enum": ["invoices", "contacts"],
                "description": "Which records to read.",
            },
            "payment_status": {
                "type": "string",
                "enum": ["PAID", "OUTSTANDING", "OVERDUE"],
                "description": (
                    "Invoices only. OVERDUE answers 'who hasn't paid me'. Omit for all invoices. "
                    "The accounting system decides this, so do not work it out from dates."
                ),
            },
            "search": {"type": "string", "description": "Contacts only. Narrows by name."},
            "limit": {"type": "integer", "description": "How many rows at most (default 20)."},
        },
        "required": ["resource"],
    },
}


def _post(payload: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"{_api_base()}{CONNECTOR_PATH}",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {_credential()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return response.status, json.loads(response.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") or "{}"
        try:
            return exc.code, json.loads(body)
        except ValueError:
            return exc.code, {}


def handle_business_records(args: dict, **_kw) -> str:
    if not check_available():
        return tool_error("Jentera's control plane is not reachable from here.")
    resource = str(args.get("resource") or "").strip().lower()
    if resource not in ("invoices", "contacts"):
        return tool_error("resource must be 'invoices' or 'contacts'")

    call_args: dict = {"resource": resource}
    if resource == "invoices" and args.get("payment_status"):
        call_args["paymentStatus"] = str(args["payment_status"]).strip().upper()
    if resource == "contacts" and args.get("search"):
        call_args["search"] = str(args["search"])[:200]
    if isinstance(args.get("limit"), int):
        call_args["limit"] = args["limit"]

    # No run id is sent, because none is available to be sent. Hermes
    # passes tool handlers `parent_agent` in CLI mode and nothing in
    # gateway mode, which is what a sprite runs, and no environment
    # variable carries one — a process-level variable would be worse than
    # none, since this gateway outlives every task and would stamp each
    # reading with the same stale run. The control plane accepts `runId`
    # for a caller that genuinely knows it; this one does not pretend to.
    payload = {
        # The connector is Jentera's business, not the model's: it knows
        # which accounting system this owner connected, and telling the
        # model would only invite it to name a different one.
        "connector": "Bukku",
        "op": "list",
        "args": call_args,
    }

    try:
        status, body = _post(payload)
    except (urllib.error.URLError, TimeoutError, OSError):
        return tool_error("Jentera did not respond. Try again, or tell the owner you could not check.")
    except ValueError:
        return tool_error("Jentera returned something unreadable.")

    if status == 200 and body.get("ok"):
        return tool_result(summary=str(body.get("detail", ""))[:MAX_DETAIL_CHARS], resource=resource)
    # Everything else is already phrased for a person, because the owner is
    # who ends up reading it. `needs_approval` is a refusal with an action.
    message = str(body.get("err") or f"Jentera refused that reading ({status}).")
    if body.get("code") == "needs_approval":
        return tool_error(message, needs_owner_approval=True)
    return tool_error(message)
