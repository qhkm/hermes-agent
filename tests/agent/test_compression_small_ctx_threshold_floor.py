"""Compression hygiene: small-context threshold floor, reasoning-trace
exclusion, and bounded summary size.

Covers the July 2026 compression tuning pass:

1. Reasoning traces (native ``reasoning`` field AND inline ``<think>``-style
   blocks) must never reach the summarizer prompt, and traces emitted BY the
   summarizer model must never be stored in the summary.
2. Head/tail protection budgets stay proportionate (tail = 20% of threshold).
3. Summary token budget is bounded to the 1K-10K envelope.
4. Models with context windows below 512K get their compression threshold
   floored at 75% (raise-only — a higher configured value always wins).
"""

from unittest.mock import patch

import agent.context_compressor as cc
from agent.context_compressor import ContextCompressor


def _make(ctx: int, pct: float = 0.50) -> ContextCompressor:
    with patch.object(cc, "get_model_context_length", return_value=ctx):
        return ContextCompressor(
            model="test/model", threshold_percent=pct, quiet_mode=True,
        )


class TestSmallContextThresholdFloor:
    def test_sub_512k_floors_to_75_percent(self):
        for ctx in (128_000, 200_000, 262_144, 511_999):
            comp = _make(ctx, pct=0.50)
            assert comp.threshold_percent == 0.75, ctx
            assert comp.threshold_tokens == int(ctx * 0.75), ctx

    def test_512k_and_above_keep_configured_percent(self):
        for ctx in (512_000, 1_000_000):
            comp = _make(ctx, pct=0.50)
            assert comp.threshold_percent == 0.50, ctx
            assert comp.threshold_tokens == int(ctx * 0.50), ctx

    def test_raise_only_higher_config_wins(self):
        # Explicit 85% (user config or Codex gpt-5.5 autoraise) is not lowered.
        comp = _make(128_000, pct=0.85)
        assert comp.threshold_percent == 0.85

    def test_degenerate_minimum_window_still_uses_85(self):
        # 64K window: the MINIMUM_CONTEXT_LENGTH floor pushes the threshold
        # to/over the window, so the 85% degenerate-window guard still rules.
        comp = _make(64_000, pct=0.50)
        assert comp.threshold_tokens == 54_400  # 85% of 64000

    def test_update_model_rederives_floor_both_directions(self):
        comp = _make(128_000, pct=0.50)
        assert comp.threshold_percent == 0.75
        # small -> large: back to the configured 50%
        comp.update_model("big", 1_000_000)
        assert comp.threshold_percent == 0.50
        assert comp.threshold_tokens == 500_000
        # large -> small: floor re-applies
        comp.update_model("small", 200_000)
        assert comp.threshold_percent == 0.75
        assert comp.threshold_tokens == 150_000


class TestReasoningExcludedFromSummarizer:
    def test_serializer_drops_inline_think_blocks(self):
        comp = _make(128_000)
        turns = [
            {"role": "user", "content": "do the thing"},
            {"role": "assistant", "content": "<think>INLINE_TRACE</think>visible answer"},
            {"role": "assistant", "content": "<reasoning>VARIANT_TRACE</reasoning>other answer"},
        ]
        ser = comp._serialize_for_summary(turns)
        assert "INLINE_TRACE" not in ser
        assert "VARIANT_TRACE" not in ser
        assert "visible answer" in ser
        assert "other answer" in ser

    def test_serializer_excludes_native_reasoning_field(self):
        comp = _make(128_000)
        turns = [{"role": "assistant", "content": "done", "reasoning": "NATIVE_TRACE"}]
        ser = comp._serialize_for_summary(turns)
        assert "NATIVE_TRACE" not in ser
        assert "done" in ser

    def test_summarizer_output_think_block_stripped_before_store(self):
        comp = _make(128_000)

        class FakeMsg:
            content = "<think>OUTPUT_TRACE</think>\n## Active Task\nUser asked X"

        class FakeChoice:
            message = FakeMsg()

        class FakeResp:
            choices = [FakeChoice()]

        with patch.object(cc, "call_llm", return_value=FakeResp()):
            out = comp._generate_summary([{"role": "user", "content": "hi"}])
        assert out is not None
        assert "OUTPUT_TRACE" not in out
        assert "## Active Task" in out
        # The iterative-update seed must be clean too, or the trace compounds
        # across every subsequent compaction.
        assert "OUTPUT_TRACE" not in (comp._previous_summary or "")

    def test_thinking_only_summarizer_response_not_blanked(self):
        # If stripping removes everything (degenerate model output), keep the
        # raw content instead of storing an empty summary.
        comp = _make(128_000)

        class FakeMsg:
            content = "<think>only reasoning, no body</think>"

        class FakeChoice:
            message = FakeMsg()

        class FakeResp:
            choices = [FakeChoice()]

        with patch.object(cc, "call_llm", return_value=FakeResp()):
            out = comp._generate_summary([{"role": "user", "content": "hi"}])
        # Falls back to unstripped content rather than an empty summary body.
        assert out is not None and out.strip()


class TestSummaryBudgetEnvelope:
    def test_no_max_tokens_wire_cap_on_summary_call(self):
        """The summary budget is PROMPT GUIDANCE only ("Target ~N tokens").

        A wire-level max_tokens cap truncates summaries mid-section on the
        Anthropic Messages / NVIDIA NIM paths (which forward the param), and
        thinking models burn the cap on reasoning before emitting the summary
        body — producing truncated or thinking-only summaries and compaction
        loops. The call must NOT carry max_tokens.
        """
        comp = _make(128_000)
        captured = {}

        class FakeMsg:
            content = "## Active Task\nUser asked X"

        class FakeChoice:
            message = FakeMsg()

        class FakeResp:
            choices = [FakeChoice()]

        def fake_call_llm(**kw):
            captured.update(kw)
            return FakeResp()

        with patch.object(cc, "call_llm", side_effect=fake_call_llm):
            out = comp._generate_summary([{"role": "user", "content": "hi"}])
        assert out is not None
        assert "max_tokens" not in captured
        # The budget still lands as prompt guidance, within the envelope.
        prompt = captured["messages"][0]["content"]
        import re
        m = re.search(r"Target ~(\d+) tokens", prompt)
        assert m, "prompt-level token target guidance missing"
        assert 1_000 <= int(m.group(1)) <= 10_000

    def test_budget_capped_at_10k_even_on_1m_window(self):
        comp = _make(1_000_000)
        huge = [{"role": "assistant", "content": "x" * 8000} for _ in range(200)]
        assert comp._compute_summary_budget(huge) <= 10_000
        assert comp.max_summary_tokens <= 10_000

    def test_budget_floor_stays_in_envelope(self):
        comp = _make(1_000_000)
        tiny = [{"role": "user", "content": "hi"}]
        budget = comp._compute_summary_budget(tiny)
        assert 1_000 <= budget <= 10_000

    def test_ceiling_constant_within_envelope(self):
        assert 1_000 <= cc._SUMMARY_TOKENS_CEILING <= 10_000
        assert 1_000 <= cc._MIN_SUMMARY_TOKENS <= 10_000


class TestTailBudgetProportionality:
    def test_tail_budget_is_target_ratio_of_threshold(self):
        comp = _make(128_000)
        assert comp.tail_token_budget == int(comp.threshold_tokens * comp.summary_target_ratio)
        # Sanity: tail protection stays a modest slice of the window (<= 20%).
        assert comp.tail_token_budget <= comp.context_length * 0.20


class TestTailWindowShareCap:
    """upstream NousResearch/hermes-agent#117957"""

    def test_tail_budget_never_exceeds_window_share(self):
        from agent.context_compressor import TAIL_MAX_CONTEXT_FRACTION

        for ctx in (8_192, 16_384, 32_768, 128_000):
            with patch.object(cc, "get_model_context_length", return_value=ctx):
                comp = ContextCompressor(
                    model="test/model", threshold_percent=0.85,
                    summary_target_ratio=0.80, quiet_mode=True,
                )
            assert 0 < comp.tail_token_budget <= ctx * TAIL_MAX_CONTEXT_FRACTION, ctx

    def test_update_model_reapplies_cap(self):
        from agent.context_compressor import TAIL_MAX_CONTEXT_FRACTION

        with patch.object(cc, "get_model_context_length", return_value=1_000_000):
            comp = ContextCompressor(
                model="big", threshold_percent=0.85, summary_target_ratio=0.80, quiet_mode=True,
            )
        comp.update_model("small", 32_768)
        assert comp.tail_token_budget <= 32_768 * TAIL_MAX_CONTEXT_FRACTION

    def test_soft_ceiling_is_capped_but_never_below_budget(self):
        comp = _make(16_384)
        assert comp._tail_soft_ceiling(3_000) == int(16_384 * 0.20)
        assert comp._tail_soft_ceiling(1_000) == 1_500
        assert comp._tail_soft_ceiling(5_000) == 5_000  # budget itself always wins

    def test_small_window_compress_leaves_a_real_middle(self):
        from agent.context_compressor import TAIL_MAX_CONTEXT_FRACTION, _estimate_msg_budget_tokens

        ctx = 8_192
        comp = _make(ctx)
        msgs: list = [{"role": "system", "content": "sys"}]
        for i in range(12):
            msgs.append({"role": "user", "content": f"step {i}"})
            msgs.append({
                "role": "assistant", "content": "",
                "tool_calls": [{"id": f"c{i}", "type": "function",
                                "function": {"name": "terminal", "arguments": "{}"}}],
            })
            msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "x" * 4_000})
            msgs.append({"role": "assistant", "content": f"done {i}"})
        head_end = comp.protect_first_n
        cut = comp._find_tail_cut_by_tokens(msgs, head_end)
        tail_tokens = sum(_estimate_msg_budget_tokens(m) for m in msgs[cut:])
        one_turn = sum(_estimate_msg_budget_tokens(m) for m in msgs[-4:])
        assert tail_tokens <= ctx * TAIL_MAX_CONTEXT_FRACTION + one_turn
        assert cut - head_end >= (len(msgs) - head_end) // 2, (head_end, cut, len(msgs))
