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

from typing import Any

# Same bound as the control plane's ``instructions`` (20,000 characters).
TURN_NOTES_MAX = 20_000


def wrap_turn_notes(notes: str) -> str:
    """The marked block the notes travel in."""
    return f"<jentera-context>\n{notes}\n</jentera-context>"


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
