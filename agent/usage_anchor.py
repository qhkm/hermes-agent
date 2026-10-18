"""Usage-anchored token accounting: the provider's real ``usage.prompt_tokens`` is the authoritative
context size; the local ``bytes/4`` estimate covers ONLY messages appended since.

Ported from NousResearch/hermes-agent#104555 (in-process part only: this fork keeps no persisted
anchor on the session row).

An anchor = provider usage at capture + a snapshot of the transcript position it priced:
``base_count`` (len(messages) at capture; the assistant reply is not yet appended and is covered by
``completion_tokens``, so the delta walk skips an assistant row at that index), ``base_last_role``
and ``base_last_fp`` (content fingerprint of the last priced message). Compaction, splices and
rewinds replace that message, so the anchor fails closed and callers fall back to full estimation
until the next real reading. The fingerprint (not ``id()``) is the identity because the gateway
re-reads the transcript from the DB every turn, so object identity is never stable.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

# Provider-visible fields that round-trip the session DB byte-for-byte. Display/persistence
# metadata (timestamps, row ids) is rewritten on reload and would only ever fail the match closed.
_FINGERPRINT_KEYS = ("role", "content", "api_content", "tool_call_id", "tool_calls")


def message_fingerprint(msg: Any) -> Optional[str]:
    """Stable digest of one transcript message over its provider-visible, persisted fields."""
    if not isinstance(msg, dict):
        return None
    payload = {k: msg.get(k) for k in _FINGERPRINT_KEYS if msg.get(k) is not None}
    try:
        raw = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True, separators=(",", ":"))
    except (TypeError, ValueError):
        raw = repr(sorted(payload.items()))
    return hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()


def context_fingerprint(system_prompt: Any, tools: Any) -> str:
    """Digest of the non-message part of a request: the system prompt and the tool schema list.
    The provider's prompt_tokens priced these too, so a change to either (prompt rewrite on
    fallback, a toolset toggle, MCP tools appearing) makes an anchor taken before it a lie."""
    try:
        tools_raw = json.dumps(tools or [], sort_keys=True, default=str, ensure_ascii=True, separators=(",", ":"))
    except (TypeError, ValueError):
        tools_raw = repr(tools)
    h = hashlib.sha256()
    h.update(str(system_prompt or "").encode("utf-8", "replace"))
    h.update(b"\x00")
    h.update(tools_raw.encode("utf-8", "replace"))
    return h.hexdigest()


def agent_context_fingerprint(agent: Any) -> str:
    """``context_fingerprint`` of what ``agent`` would send now (cached prompt + ephemeral + tools)."""
    prompt = getattr(agent, "_cached_system_prompt", None) or ""
    ephemeral = getattr(agent, "ephemeral_system_prompt", None)
    if ephemeral:
        prompt = f"{prompt}\n\n{ephemeral}"
    return context_fingerprint(prompt, getattr(agent, "tools", None))


def capture_usage_anchor(
    prompt_tokens: Any, completion_tokens: Any, messages: List[Dict[str, Any]],
    context_fp: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Build a usage anchor from provider-reported usage, or None when usage is unusable."""
    try:
        pt = int(prompt_tokens or 0)
        ct = int(completion_tokens or 0)
    except (TypeError, ValueError):
        return None
    if pt <= 0 or not isinstance(messages, list) or not messages:
        return None  # some endpoints omit usage; the caller keeps its previous anchor
    last = messages[-1]
    return {
        "prompt_tokens": pt,
        "completion_tokens": max(0, ct),
        "base_count": len(messages),
        "base_last_role": last.get("role") if isinstance(last, dict) else None,
        "base_last_fp": message_fingerprint(last),
        "context_fp": context_fp,
    }


def _anchor_matches(messages: List[Dict[str, Any]], anchor: Dict[str, Any]) -> bool:
    try:
        base_count = int(anchor.get("base_count") or 0)
    except (TypeError, ValueError):
        return False
    if base_count <= 0 or len(messages) < base_count:
        return False
    base_msg = messages[base_count - 1]
    if not isinstance(base_msg, dict) or base_msg.get("role") != anchor.get("base_last_role"):
        return False
    fp = anchor.get("base_last_fp")
    return isinstance(fp, str) and bool(fp) and message_fingerprint(base_msg) == fp


def anchored_context_tokens(
    messages: List[Dict[str, Any]], anchor: Optional[Dict[str, Any]],
) -> Optional[int]:
    """Anchored prompt+completion tokens plus a rough estimate of ONLY the messages appended since;
    None when the anchor is missing or stale. The anchored response's own reply is skipped (already
    in completion_tokens)."""
    if not isinstance(anchor, dict) or not isinstance(messages, list) or not _anchor_matches(messages, anchor):
        return None
    from agent.model_metadata import estimate_messages_tokens_rough

    total = int(anchor["prompt_tokens"]) + int(anchor.get("completion_tokens") or 0)
    delta = messages[int(anchor["base_count"]):]
    if delta and isinstance(delta[0], dict) and delta[0].get("role") == "assistant":
        delta = delta[1:]
    if delta:
        total += estimate_messages_tokens_rough(delta)
    return total


def set_usage_anchor(agent: Any, anchor: Optional[Dict[str, Any]]) -> None:
    """Install ``anchor`` on the agent (``None`` clears it)."""
    agent._usage_anchor = anchor


def clear_usage_anchor(agent: Any) -> None:
    """Drop the anchor: the provider that priced it is no longer the one answering."""
    agent._usage_anchor = None


def agent_anchored_tokens(agent: Any, messages: List[Dict[str, Any]]) -> Optional[int]:
    """The agent's anchored context figure for ``messages``, or None (no/stale anchor).

    An anchor that recorded a system-prompt/tool-schema digest is stale once either changed."""
    anchor = getattr(agent, "_usage_anchor", None)
    if isinstance(anchor, dict) and anchor.get("context_fp") is not None:
        if anchor["context_fp"] != agent_context_fingerprint(agent):
            return None
    return anchored_context_tokens(messages, anchor)
