"""Build and change the owner's website through Jentera's control plane.

The agent edits a draft only. Publishing, going back to an earlier version and
changing the address are the owner's, in the Jentera app; this machine cannot
do them. Every change is checked by Jentera before it is saved, and a refusal
comes back as plain reasons the agent can fix and try again.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from tools.registry import tool_error, tool_result

from plugins.jentera import handoff
from plugins.jentera.handoff import _runner_key, _runner_url
from plugins.jentera.tools import _post, check_available

SITE_PATH = "/v1/runtime/site"
_RULES = (
    "Only write what the owner told you or what Jentera already knows: never invent reviews, prices, awards, "
    "opening hours or claims. The owner sees the draft in the Jentera app beside this chat; it is not public "
    "until they tap Publish there, so never say it is live. Reply in the owner's language, briefly, saying what changed."
)

_IMAGES = (
    "When the owner says they added a photo, it is in `images` (source 'owner_upload'): put its `id` in a "
    "section's `image_id` field; don't ask them for a link."
)
SITE_DRAFT_SCHEMA = {
    "name": "site_draft",
    "description": (
        "Start the owner's business website when they ask for one. Makes a first draft from what Jentera knows "
        "(name, confirmed facts, booking page, enquiry form) or returns the existing draft, plus the list of "
        "sections and fields you may use and the business's `images`. Call it before site_apply the first time. "
        "Follow the rules in site_apply. " + _IMAGES
    ),
    "parameters": {"type": "object", "properties": {}},
}
SITE_GET_SCHEMA = {
    "name": "site_get",
    "description": (
        "Read the current website draft, the sections and fields you may use, and `images`: the photos the "
        "business has added, newest first (`id`, `source`, `licence`). " + _IMAGES
    ),
    "parameters": {"type": "object", "properties": {}},
}
SITE_APPLY_SCHEMA = {
    "name": "site_apply",
    "description": (
        "Change the website draft. `ops` is a list applied together, or not at all: "
        "{op:'add_section', type, fields, after?: section id or null for the top}, "
        "{op:'update_section', id, fields} (only the fields you name change), "
        "{op:'move_section', id, after: section id or null}, {op:'remove_section', id}, "
        "{op:'set_theme', palette?, fonts?, corners?}, {op:'set_seo', title?, description?, image_id?}. "
        "If Jentera refuses, read its reasons, fix them and try again. " + _RULES
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "ops": {"type": "array", "items": {"type": "object"}, "description": "The changes, at most 30."},
            "note": {"type": "string", "description": "One short line the owner will see in the site's history, e.g. 'Added the menu'."},
        },
        "required": ["ops"],
    },
}
SITE_IMAGE_SCHEMA = {
    "name": "site_image",
    "description": (
        "Add a photo to the website from an https link to a PNG, JPEG or WebP image (up to 5 MB). The result's "
        "`imageId` is what goes into a section's `image_id` field (and `image_id` in set_seo). Only use images the business may use: the owner's own photos, or free "
        "licences such as Unsplash or Pexels. Say which in `licence`. Not for a photo the owner added in Jentera: "
        "that one is already in `images` from site_get."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "https link straight to the image file."},
            "licence": {"type": "string", "description": "Where it comes from and its licence."},
        },
        "required": ["url", "licence"],
    },
}


_AVAILABLE_TTL_SECONDS = 60
# Hermes runs check_fns while it builds a turn's tool list, so a slow probe
# delays the turn itself. Short, like handoff.py's AVAILABLE_TIMEOUT_SECONDS;
# the 20 s REQUEST_TIMEOUT_SECONDS is for the tool calls, not this question.
AVAILABLE_TIMEOUT_SECONDS = 3
_available_cache: tuple[float, bool, bool] | None = None


def site_available() -> bool:
    """Offered only on a business whose websites are switched on (cached 60 s).

    A failed probe is cached for the same minute as an answer: the last answer
    if there was one, otherwise hidden. Retrying on every call would make each
    turn wait out the timeout again while the control plane is down.
    """
    global _available_cache
    if not check_available():
        return False
    now = time.monotonic()
    if _available_cache and now - _available_cache[0] < _AVAILABLE_TTL_SECONDS:
        return _available_cache[1]
    try:
        status, body = _post({"op": "available"}, SITE_PATH, timeout=AVAILABLE_TIMEOUT_SECONDS)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        _available_cache = (now, _available_cache[1] if _available_cache else False, _available_cache[2] if _available_cache else False)
        return _available_cache[1]
    value = status == 200 and body.get("available") is True
    _available_cache = (now, value, value and body.get("code") is True)
    return value


def site_code_available() -> bool:
    """Websites are on and the owner's site is edited by Jentera's code builder."""
    return site_available() and bool(_available_cache and _available_cache[2])


def site_block_available() -> bool:
    """Websites are on and code mode is off, so edits go through site_apply."""
    return site_available() and not bool(_available_cache and _available_cache[2])


def _call(payload: dict) -> str:
    if not check_available():
        return tool_error("Jentera's control plane is not reachable from here.")
    try:
        status, body = _post(payload, SITE_PATH)
    except (urllib.error.URLError, TimeoutError, OSError):
        return tool_error("Jentera did not respond. Tell the owner the website could not be changed just now.")
    except ValueError:
        return tool_error("Jentera returned something unreadable.")
    if status == 200 and body.get("ok"):
        return tool_result({k: v for k, v in body.items() if k != "ok"})
    if body.get("problems"):
        if payload.get("op") == "draft":
            return tool_error("Jentera could not make the first draft:", problems=body["problems"])
        return tool_error("Jentera did not save that. Fix these and try again.", problems=body["problems"])
    return tool_error(str(body.get("err") or f"Jentera refused that ({status})."))


def handle_site_draft(args: dict, **_kw) -> str:
    return _call({"op": "draft"})


def handle_site_get(args: dict, **_kw) -> str:
    return _call({"op": "get"})


def handle_site_apply(args: dict, **_kw) -> str:
    ops = args.get("ops")
    if not isinstance(ops, list) or not ops:
        return tool_error("ops must be a list of changes, e.g. [{'op': 'set_theme', 'palette': 'rose'}].")
    payload: dict = {"op": "apply", "ops": ops}
    note = args.get("note")
    if isinstance(note, str) and note.strip():
        payload["note"] = note.strip()[:120]
    return _call(payload)


def handle_site_image(args: dict, **_kw) -> str:
    url = str(args.get("url") or "").strip()
    licence = str(args.get("licence") or "").strip()
    if not licence:
        return tool_error("Say where the image comes from and its licence before adding it.")
    if not url.startswith("https://"):
        return tool_error("url must be an https link straight to the image.")
    return _call({"op": "image", "url": url[:500], "licence": licence[:200]})


SITE_CODE_TIMEOUT_SECONDS = 300

SITE_CODE_SCHEMA = {
    "name": "site_code",
    "description": (
        "Change the owner's website. Describe the change in plain words, with the owner's own wording and any facts "
        "they gave (names, prices, hours). Jentera's website builder edits the real page and the owner sees it beside "
        "this chat. It writes HTML and CSS only (no scripts or forms). Where details are missing it leaves a marked gap "
        "like [Your price here]; then ask the owner for them. Photos the owner added are used automatically. "
        + _RULES
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "request": {"type": "string", "description": "The change to make, in plain words, with every fact needed. Up to 2000 characters."},
            "reference_image": {"type": "string", "description": "Optional: the file path of a picture the owner sent in this chat that shows the look they want (for example a site they like)."},
        },
        "required": ["request"],
    },
}


def handle_site_code(args: dict, **_kw) -> str:
    request_text = str(args.get("request") or "").strip()
    if not request_text or len(request_text) > 2000:
        return tool_error("request must say the change in 1 to 2000 characters.")
    run_id = handoff._calling_run_id()
    if not run_id or not _runner_key():
        return tool_error(
            "Website changes are not available here. Tell the owner to try again in a moment.",
            code="unavailable",
        )
    payload = {"runId": run_id, "request": request_text}
    image = str(args.get("reference_image") or "").strip()
    if image:
        payload["image"] = image
    req = urllib.request.Request(
        _runner_url() + "/v1/site/code",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {_runner_key()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=SITE_CODE_TIMEOUT_SECONDS) as resp:
            body = json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read() or b"{}")
        except ValueError:
            body = {}
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return tool_error("The website could not be changed just now. Tell the owner and offer to try again.")
    if body.get("ok"):
        return tool_result({"note": body.get("note", ""), "changed": bool(body.get("changed"))})
    return tool_error(str(body.get("error") or "The website could not be changed just now."))
