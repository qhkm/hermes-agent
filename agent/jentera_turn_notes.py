"""Per-turn notes from the Jentera control plane.

Jentera's per-message context (the facts retrieved for this message, recent
work, the clock) used to be appended to the system message through
``ephemeral_system_prompt``. It differs on every turn, so the system message
differed on every turn and the provider's prompt cache missed the whole
history behind it.

``turn_notes`` carries that text instead. It is put on the CURRENT turn's user
message when a request is built, on every model call of the turn, after the
history. It is API-call-time only: never written to ``messages``, never into
the ``api_content`` sidecar, so earlier turns replay without it and nothing
stale piles up in the transcript. The notes text is never logged.
"""

import re
from typing import Any

# Same bound as the control plane's ``instructions`` (20,000 characters).
TURN_NOTES_MAX = 20_000


# A block tag as the model could read it: any case, with spaces or newlines
# around the slash and name, with or without its closing bracket. The slash and
# the whitespace after it are one optional group: two whitespace runs with only
# an optional slash between them made the engine try every split of a long run
# of spaces after a "<" (seconds on 20,000 of them).
_BLOCK_TAG = re.compile(r"<\s*(?:(/)\s*)?(jentera-context)(\s*>)?", re.IGNORECASE)


def _defang(match: "re.Match[str]") -> str:
    return "\u2039" + (match.group(1) or "") + match.group(2) + ("\u203a" if match.group(3) else "")


def wrap_turn_notes(notes: str) -> str:
    """The marked block the notes travel in.

    The notes carry text outsiders submit (customer and card names from public
    forms), so a block tag inside them is rewritten to a harmless form
    (``</jentera-context>`` becomes ``\u2039/jentera-context\u203a``) and cannot close
    or reopen the block early. Notes without one come out unchanged.
    """
    return f"<jentera-context>\n{_BLOCK_TAG.sub(_defang, notes)}\n</jentera-context>"


def with_turn_notes(content: Any, notes: Any) -> Any:
    """Return ``content`` of a user message with the notes block in front.

    A string gets the block and a blank line before it; a list of content
    parts (text plus images) gets a new leading text part. Empty or missing
    notes, and any other content shape, come back unchanged. The input is
    never mutated.
    """
    if not isinstance(notes, str) or not notes.strip():
        return content
    block = wrap_turn_notes(notes)
    if isinstance(content, str):
        return block + "\n\n" + content
    if isinstance(content, list):
        return [{"type": "text", "text": block}, *content]
    return content
