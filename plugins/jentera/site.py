"""Build and change the owner's website through Jentera's control plane.

The agent edits a draft only. Publishing, going back to an earlier version and
changing the address are the owner's, in the Jentera app; this machine cannot
do them. Every change is checked by Jentera before it is saved, and a refusal
comes back as plain reasons the agent can fix and try again.
"""

from __future__ import annotations

import urllib.error

from tools.registry import tool_error, tool_result

from plugins.jentera.tools import _post, check_available

SITE_PATH = "/v1/runtime/site"
_RULES = (
    "Only write what the owner told you or what Jentera already knows: never invent reviews, prices, awards, "
    "opening hours or claims. The owner sees the draft in the Jentera app beside this chat; it is not public "
    "until they tap Publish there, so never say it is live. Reply in the owner's language, briefly, saying what changed."
)

SITE_DRAFT_SCHEMA = {
    "name": "site_draft",
    "description": (
        "Start the owner's business website when they ask for one. Makes a first draft from what Jentera knows "
        "(name, confirmed facts, booking page, enquiry form) or returns the existing draft, plus the list of "
        "sections and fields you may use. Call it before site_apply the first time. " + _RULES
    ),
    "parameters": {"type": "object", "properties": {}},
}
SITE_GET_SCHEMA = {
    "name": "site_get",
    "description": "Read the current website draft and the sections and fields you may use. " + _RULES,
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
        "Add a photo to the website from an https link to a PNG, JPEG or WebP image (up to 5 MB). Returns an "
        "image_id to use in a section. Only use images the business may use: the owner's own photos, or free "
        "licences such as Unsplash or Pexels. Say which in `licence`."
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
