"""Blank the values the runner typed into a page for the owner.

The runner keeps recent filled values in memory; Hermes never holds them. It
asks the runner on loopback whether any value is live (``/v1/private-form/active``,
cached for two seconds) and, only while one is, sends the model-bound texts to
``/v1/private-form/blank`` and uses what comes back. Any failure while the last
known answer was "live" withholds the text instead of sending it unscrubbed;
while nothing is live, texts pass through. Nothing here logs a text.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request

from agent.redact import MODEL_TEXT_CHECK_FAILED, set_model_text_scrubber
from plugins.jentera.handoff import _runner_key, _runner_url

ACTIVE_PATH = "/v1/private-form/active"
BLANK_PATH = "/v1/private-form/blank"
TIMEOUT_SECONDS = 1.0
BLANK_TIMEOUT_SECONDS = 3.0
ACTIVE_CACHE_SECONDS = 2.0
# The runner's limits for one /blank request.
MAX_TEXTS = 64
MAX_CHARS = 512 * 1024
# A text longer than one piece is sent in pieces. Each cut is checked with a window
# around it: a value that crosses the cut changes the window but not its two halves.
PIECE_CHARS = 256 * 1024
WINDOW_CHARS = 600

# Loopback only: never route through an environment proxy.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_LOCK = threading.Lock()
# None until the runner has answered once; then its last answer.
_state = {"active": None, "checked_at": 0.0}


class _CheckFailed(Exception):
    pass


def _request(path: str, body=None, timeout: float = TIMEOUT_SECONDS):
    key = _runner_key()
    if not key:
        raise _CheckFailed()
    data = None if body is None else json.dumps(body).encode()
    headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        _runner_url() + path, data=data, method="GET" if data is None else "POST", headers=headers
    )
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            if response.status != 200:
                raise _CheckFailed()
            parsed = json.loads(response.read() or b"{}")
    except _CheckFailed:
        raise
    except Exception:
        raise _CheckFailed() from None
    if not isinstance(parsed, dict) or parsed.get("ok") is not True:
        raise _CheckFailed()
    return parsed


def is_active():
    """(active, known): whether a value is live, and whether the runner just said so.

    On a failed check ``active`` is the last answer the runner gave (False before any).
    """
    with _LOCK:
        if _state["active"] is not None and time.monotonic() - _state["checked_at"] < ACTIVE_CACHE_SECONDS:
            return _state["active"], True
    try:
        answer = _request(ACTIVE_PATH)
        active = answer.get("active")
        if not isinstance(active, bool):
            raise _CheckFailed()
    except _CheckFailed:
        with _LOCK:
            return bool(_state["active"]), False
    with _LOCK:
        _state["active"] = active
        _state["checked_at"] = time.monotonic()
    return active, True


def _blank_items(items):
    """The runner's answer for each item, in order, in as few requests as its limits allow."""
    out = []
    batch, size = [], 0
    for item in items + [None]:
        if item is None or len(batch) >= MAX_TEXTS or (batch and size + len(item) > MAX_CHARS):
            if batch:
                answer = _request(BLANK_PATH, {"texts": batch}, timeout=BLANK_TIMEOUT_SECONDS)
                texts = answer.get("texts")
                if not isinstance(texts, list) or len(texts) != len(batch) or not all(isinstance(t, str) for t in texts):
                    raise _CheckFailed()
                out.extend(texts)
            batch, size = [], 0
        if item is not None:
            batch.append(item)
            size += len(item)
    return out


def _blank(texts):
    items = []
    plans = []
    for text in texts:
        if not text:
            plans.append(None)
        elif len(text) <= PIECE_CHARS:
            plans.append(("whole", len(items)))
            items.append(text)
        else:
            cuts = list(range(PIECE_CHARS, len(text), PIECE_CHARS))
            pieces = []
            for start in [0] + cuts:
                pieces.append(len(items))
                items.append(text[start:start + PIECE_CHARS])
            windows = []
            for cut in cuts:
                windows.append((len(items), len(items) + 1, len(items) + 2))
                items.extend([text[cut - WINDOW_CHARS:cut + WINDOW_CHARS], text[cut - WINDOW_CHARS:cut], text[cut:cut + WINDOW_CHARS]])
            plans.append(("pieces", pieces, windows))
    answers = _blank_items(items)
    out = []
    for text, plan in zip(texts, plans):
        if plan is None:
            out.append(text)
        elif plan[0] == "whole":
            out.append(answers[plan[1]])
        else:
            _, pieces, windows = plan
            crossed = any(answers[w] != answers[left] + answers[right] for w, left, right in windows)
            out.append(MODEL_TEXT_CHECK_FAILED if crossed else "".join(answers[p] for p in pieces))
    return out


def scrub_texts(texts):
    """The texts as they may go to the model."""
    texts = list(texts)
    if not any(texts):
        return texts
    active, known = is_active()
    if not active:
        return texts
    if not known:
        return [MODEL_TEXT_CHECK_FAILED if t else t for t in texts]
    try:
        return _blank(texts)
    except Exception:
        return [MODEL_TEXT_CHECK_FAILED if t else t for t in texts]


def _on_transform_tool_result(result=None, **kwargs):
    # One hook for the plugin, because the first string returned wins: a browser result
    # first gains the page's form (PAGE_SCAN), then everything is scrubbed as before.
    original = result
    if isinstance(result, str) and result:
        try:
            from plugins.jentera.page_scan import augment

            augmented = augment(result=result, **kwargs)
        except Exception:
            augmented = None
        if isinstance(augmented, str):
            result = augmented
    # A value may have been typed during this tool call: the result is checked before the model reads it.
    if isinstance(result, str) and result:
        scrubbed = scrub_texts([result])[0]
        if scrubbed != original:
            return scrubbed
    return None


def register_hooks(ctx) -> None:
    # Every model request (conversation_loop) goes through the runner while a value is live.
    set_model_text_scrubber(scrub_texts)
    # First string returned wins across plugins, so a plugin ahead of this one
    # can bypass the blank here; the send-time scrub is the backstop.
    ctx.register_hook("transform_tool_result", _on_transform_tool_result)
