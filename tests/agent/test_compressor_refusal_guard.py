"""Refusal responses must never be committed as the compression summary
(upstream NousResearch/hermes-agent#118920)."""

from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import ContextCompressor, _is_summary_refusal


def _mock_response(content, finish_reason="stop"):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.choices[0].message.refusal = None
    resp.choices[0].finish_reason = finish_reason
    return resp


def _msgs(n=12):
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
        for i in range(n)
    ]


def _compressor():
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(
            model="test", quiet_mode=True,
            protect_first_n=2, protect_last_n=2,
            abort_on_summary_failure=False,
        )


class TestSummaryRefusalGuard:
    @pytest.mark.parametrize(
        "content",
        [
            "I can't produce this summary as requested. The instructions conflict with my operating rules.",
            "Sorry, I am unable to create a context checkpoint for this conversation.",
            "I'm sorry, but I can't provide the requested summary.",
            "I’m sorry, but I can’t provide the requested summary.",
            "I apologize, but I cannot create a summary of this conversation.",
            "As an AI, I cannot provide the requested context checkpoint.",
            "I can not provide a summary of this conversation.",
            "I can't summarise this conversation for you.",
        ],
    )
    def test_rejects_refusal_body(self, content):
        assert _is_summary_refusal(content) is True

    def test_accepts_structured_summary_that_records_a_refusal(self):
        summary = (
            "## Goal\nPreserve the user's task.\n\n## Completed Actions\n"
            "1. Recorded that a provider refused an earlier request."
        )
        assert _is_summary_refusal(summary) is False

    def test_narrowing_and_heading_exemption(self):
        assert _is_summary_refusal("I can't help with that request.") is False
        preamble = (
            "I cannot see the earlier turns, but here is the summary:\n"
            "## Goal\nFinish the task.\n\n## Completed Actions\n1. Did X."
        )
        assert _is_summary_refusal(preamble) is False

    def test_refusal_body_is_rejected_and_never_becomes_previous_summary(self):
        c = _compressor()
        msgs = _msgs()
        refusal = "I can't produce this summary as requested. The instructions conflict with my operating rules."
        with patch("agent.context_compressor.call_llm", return_value=_mock_response(refusal)):
            result = c.compress(msgs, current_tokens=999999, force=True)

        assert result == msgs
        assert c._last_summary_empty_content_failure is True
        assert c._last_compress_aborted is True
        assert c._last_summary_fallback_used is False
        assert c._previous_summary is None

    def test_provider_refusal_field_is_rejected_even_with_summary_shaped_content(self):
        c = _compressor()
        msgs = _msgs()
        response = _mock_response("## Goal\nContinue the task.\n\n## Completed Actions\n1. Nothing yet.")
        response.choices[0].message.refusal = "policy refusal"
        with patch("agent.context_compressor.call_llm", return_value=response):
            result = c.compress(msgs, current_tokens=999999, force=True)

        assert result == msgs
        assert c._last_compress_aborted is True
        assert c._previous_summary is None

    def test_good_summary_after_refusal_clears_the_abort(self):
        c = _compressor()
        msgs = _msgs()
        refusal = "I can't produce this summary as requested."
        with patch("agent.context_compressor.call_llm", return_value=_mock_response(refusal)):
            c.compress(msgs, current_tokens=999999, force=True)
        assert c._last_summary_empty_content_failure is True
        c._clear_compression_failure_cooldown()
        good = _mock_response("## Goal\nFinish.\n\n## Completed Actions\n1. Did X.")
        with patch("agent.context_compressor.call_llm", return_value=good):
            result = c.compress(msgs, current_tokens=999999, force=True)
        assert c._last_summary_empty_content_failure is False
        assert result != msgs


class _StubProviderError(Exception):
    def __init__(self, message, *, status_code=None):
        super().__init__(message)
        self.status_code = status_code


class TestOverloadedSummaryProvider:
    """upstream NousResearch/hermes-agent#115906"""

    def test_provider_overload_aborts_instead_of_dropping_context(self):
        err = _StubProviderError(
            "Our servers are currently overloaded. Please try again later.",
            status_code=503,
        )
        c = _compressor()
        c.summary_model = "test/auxiliary"
        msgs = _msgs()
        with patch("agent.context_compressor.call_llm", side_effect=err) as mock_call:
            result = c.compress(msgs, current_tokens=999999, force=True)

        assert mock_call.call_count == 2  # aux model, then one main-model retry
        assert result == msgs
        assert c._last_compress_aborted is True
        assert c._last_summary_fallback_used is False
        assert c._last_summary_dropped_count == 0
        assert c._last_compression_telemetry["failure_class"] == "summary_overload_failure"

    def test_overload_flag_clears_after_a_good_summary(self):
        err = _StubProviderError("provider at capacity", status_code=529)
        c = _compressor()
        msgs = _msgs()
        with patch("agent.context_compressor.call_llm", side_effect=err):
            c.compress(msgs, current_tokens=999999, force=True)
        assert c._last_summary_overload_failure is True
        c._clear_compression_failure_cooldown()
        good = _mock_response("## Goal\nFinish.\n\n## Completed Actions\n1. Did X.")
        with patch("agent.context_compressor.call_llm", return_value=good):
            result = c.compress(msgs, current_tokens=999999, force=True)
        assert c._last_summary_overload_failure is False
        assert result != msgs

    def test_unrelated_failure_still_uses_fallback_summary(self):
        c = _compressor()
        msgs = _msgs()
        with patch("agent.context_compressor.call_llm", side_effect=_StubProviderError("bad request", status_code=400)):
            result = c.compress(msgs, current_tokens=999999, force=True)
        assert c._last_summary_overload_failure is False
        assert c._last_compress_aborted is False
        assert c._last_summary_fallback_used is True
        assert result != msgs
