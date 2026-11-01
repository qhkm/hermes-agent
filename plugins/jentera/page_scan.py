"""Attach the open page's form to a browser result (Jentera PAGE_SCAN).

After ``browser_navigate`` or ``browser_click`` succeeds, this asks the
Jentera runner on loopback (``POST /v1/page/scan``) for the form on the page
the agent last used, and adds a short summary to the tool result: the host,
the kind of form, and one line per box (words, kind, options, pick), ready to
copy as a ``jentera-form`` field. The bot then sees the form without probing.

Bounds: at most once per page per task, three seconds in all, and nothing is
added when the runner answers anything but a scan with boxes (no page, busy,
404 because the business's switch is off, unreachable, or no boxes). The scan
reads no box values; nothing here logs a page, a box or a result.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict

from plugins.jentera.handoff import _runner_key, _runner_url

SCAN_PATH = "/v1/page/scan"
SCAN_BUDGET_SECONDS = 3.0
# A 404 means the switch is off for this business; ask again after this long.
UNAVAILABLE_RETRY_SECONDS = 60.0
SUMMARY_MAX_CHARS = 1200
BROWSER_TOOLS = frozenset({"browser_navigate", "browser_click"})
# Tasks remembered for the once-per-page rule.
MAX_TASKS = 64
MAX_PAGES_PER_TASK = 200
RESPONSE_MAX_BYTES = 64_000

# Loopback only: never route through an environment proxy.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_LOCK = threading.Lock()
_seen: "OrderedDict[str, set]" = OrderedDict()
_state = {"unavailable_until": 0.0}


def _request_scan(timeout: float):
    """The runner's scan as a dict, or None for anything but a 200 with ok: true."""
    key = _runner_key()
    if not key:
        return None
    request = urllib.request.Request(
        _runner_url() + SCAN_PATH,
        data=b"{}",
        method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            if response.status != 200:
                return None
            raw = response.read(RESPONSE_MAX_BYTES + 1)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            with _LOCK:
                _state["unavailable_until"] = time.monotonic() + UNAVAILABLE_RETRY_SECONDS
        return None
    except Exception:
        return None
    if len(raw) > RESPONSE_MAX_BYTES:
        return None
    try:
        parsed = json.loads(raw or b"{}")
    except Exception:
        return None
    if not isinstance(parsed, dict) or parsed.get("ok") is not True:
        return None
    return parsed


def scan_within(budget: float | None = None):
    """The scan, or None when it does not arrive within ``budget`` seconds in all."""
    budget = SCAN_BUDGET_SECONDS if budget is None else budget
    box = {}

    def work():
        box["scan"] = _request_scan(budget)

    thread = threading.Thread(target=work, name="jentera-page-scan", daemon=True)
    thread.start()
    thread.join(budget)
    return box.get("scan") if not thread.is_alive() else None


def _field_line(box: dict) -> str | None:
    pick = box.get("pick")
    if not isinstance(pick, str) or not pick:
        return None
    words = box.get("words") if isinstance(box.get("words"), str) else ""
    kind = box.get("kind") if isinstance(box.get("kind"), str) else "text"
    field = {"label": words, "kind": "text" if kind == "textarea" else kind}
    options = box.get("options")
    if isinstance(options, list) and options:
        field["options"] = [o for o in options if isinstance(o, str)]
    field["pick"] = pick
    line = "- " + json.dumps(field, ensure_ascii=False)
    return line + ("  (no words on the page: give it your own label)" if not words else "")


def format_summary(scan: dict) -> str | None:
    """A compact summary of a scan with boxes, or None when there is nothing to show."""
    boxes = scan.get("boxes")
    if not isinstance(boxes, list):
        return None
    lines = [line for line in (_field_line(b) for b in boxes if isinstance(b, dict)) if line]
    if not lines:
        return None
    host = scan.get("host") if isinstance(scan.get("host"), str) else "?"
    kind = scan.get("kind") if isinstance(scan.get("kind"), str) else "unknown"
    head = [
        f"Form on this page (host {host}, kind {kind}). "
        "Build jentera-form from these boxes: copy each as a field (words as the label, kind, options, pick)."
    ]
    tail = []
    submits = scan.get("submits")
    if isinstance(submits, list):
        names = [s for s in submits if isinstance(s, str) and s]
        if names:
            tail.append("Buttons: " + " | ".join(names))
    out = list(head)
    used = len(head[0]) + sum(len(t) + 1 for t in tail)
    shown = 0
    for line in lines:
        # Room for this line and a closing note about the rest.
        if used + len(line) + 1 > SUMMARY_MAX_CHARS - 60:
            break
        out.append(line)
        used += len(line) + 1
        shown += 1
    if shown == 0:
        return None
    if shown < len(lines):
        out.append(f"({len(lines) - shown} more boxes: run jentera-page form for all of them.)")
    out.extend(tail)
    return "\n".join(out)[:SUMMARY_MAX_CHARS]


def _result_ok(result: str):
    """(parsed, ok): the tool result as JSON and whether it reports success."""
    try:
        parsed = json.loads(result)
    except Exception:
        return None, False
    if not isinstance(parsed, dict) or parsed.get("success") is not True:
        return parsed, False
    return parsed, True


def _remember(task: str, page: str) -> bool:
    """Whether ``page`` is new for ``task`` (and record it)."""
    with _LOCK:
        pages = _seen.get(task)
        if pages is None:
            pages = set()
            _seen[task] = pages
            while len(_seen) > MAX_TASKS:
                _seen.popitem(last=False)
        else:
            _seen.move_to_end(task)
        if page in pages:
            return False
        if len(pages) >= MAX_PAGES_PER_TASK:
            pages.clear()
        pages.add(page)
        return True


def _seen_before(task: str, page: str) -> bool:
    with _LOCK:
        pages = _seen.get(task)
        return bool(pages) and page in pages


def augment(tool_name=None, result=None, task_id="", **_kwargs):
    """The browser result with the page's form added, or None to leave it as it is."""
    if tool_name not in BROWSER_TOOLS or not isinstance(result, str) or not result:
        return None
    parsed, ok = _result_ok(result)
    if not ok:
        return None
    with _LOCK:
        if time.monotonic() < _state["unavailable_until"]:
            return None
    task = task_id or "default"
    # A navigate names its page; skip the scan for a page already summarised in this task.
    url = parsed.get("url") if isinstance(parsed.get("url"), str) else ""
    if url and _seen_before(task, "url:" + url):
        return None
    scan = scan_within()
    if not scan:
        return None
    summary = format_summary(scan)
    if not summary:
        # The page answered with no boxes: do not ask again for the same address in this task.
        if url:
            _remember(task, "url:" + url)
        return None
    # A click names no page: the scan's own host and boxes stand for it.
    picks = "|".join(str(b.get("pick")) for b in scan.get("boxes", []) if isinstance(b, dict))
    page_key = "url:" + url if url else f"form:{scan.get('host')}:{picks}"
    if not _remember(task, page_key):
        return None
    if url:
        # The same form reached later by a click is not repeated either.
        _remember(task, f"form:{scan.get('host')}:{picks}")
    parsed["page_form"] = summary
    return json.dumps(parsed, ensure_ascii=False)
