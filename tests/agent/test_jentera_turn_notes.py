"""Unit tests for ``agent/jentera_turn_notes.py``.

Per-turn notes ride on the current turn's user message at call time. These
cover the wrap format and the string / multimodal / empty cases; the wire and
persistence behaviour is covered end to end in
``tests/gateway/test_api_server_turn_notes.py``.
"""

from agent.jentera_turn_notes import TURN_NOTES_MAX, with_turn_notes, wrap_turn_notes


def test_limit_is_twenty_thousand():
    assert TURN_NOTES_MAX == 20_000


def test_wrap_is_exact():
    assert wrap_turn_notes("N1") == "<jentera-context>\nN1\n</jentera-context>"


def test_wrap_keeps_multiline_notes_verbatim():
    notes = "Confirmed information:\n- a\n- b\n\nClock: now"
    assert wrap_turn_notes(notes) == f"<jentera-context>\n{notes}\n</jentera-context>"


def test_string_content_gets_the_block_first():
    out = with_turn_notes("hello", "N1")
    assert out == "<jentera-context>\nN1\n</jentera-context>\n\nhello"


def test_list_content_gets_a_leading_text_part_and_keeps_the_rest():
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    text = {"type": "text", "text": "what is this"}
    content = [text, image]

    out = with_turn_notes(content, "N1")

    assert out == [
        {"type": "text", "text": "<jentera-context>\nN1\n</jentera-context>"},
        text,
        image,
    ]
    # A new list: the stored message content is never mutated.
    assert out is not content
    assert content == [text, image]


def test_empty_or_missing_notes_leave_content_unchanged():
    content = "hello"
    parts = [{"type": "text", "text": "hello"}]
    for notes in (None, "", "   \n"):
        assert with_turn_notes(content, notes) is content
        assert with_turn_notes(parts, notes) is parts


def test_other_content_shapes_pass_through():
    assert with_turn_notes(None, "N1") is None
