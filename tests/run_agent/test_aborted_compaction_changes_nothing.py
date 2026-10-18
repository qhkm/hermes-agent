"""An aborted compaction (refused or failed summary) leaves the live transcript untouched, even if
the engine mutated the caller's list before aborting (upstream NousResearch/hermes-agent#136083,
the uncommitted-compaction part only)."""

import os
from unittest.mock import MagicMock, patch


def _agent():
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent
        return AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
            quiet_mode=True, skip_context_files=True, skip_memory=True,
        )


def test_aborted_compaction_restores_mutated_input_and_skips_commit():
    agent = _agent()
    messages = [{"role": "user", "content": f"m{i}"} for i in range(10)]
    snapshot = [dict(m) for m in messages]

    def _mutating_abort(msgs, **_kw):
        msgs.append({"role": "user", "content": "pad"})
        msgs[0]["content"] = "mutated"
        return msgs

    compressor = MagicMock()
    compressor.compress.side_effect = _mutating_abort
    compressor._last_compress_aborted = True
    compressor._last_summary_error = "Context compression LLM returned refusal content"
    agent.context_compressor = compressor
    agent.commit_memory_session = MagicMock()
    sid = agent.session_id

    out, _sp = agent._compress_context(messages, "sys", approx_tokens=10_000)

    assert out == snapshot
    assert messages == snapshot
    assert agent.session_id == sid
    agent.commit_memory_session.assert_not_called()
