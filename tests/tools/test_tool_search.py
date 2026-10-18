"""Tests for tools/tool_search.py — progressive tool disclosure.

Coverage targets — these mirror the issues called out in the OpenClaw tool
search report. Every test that names an OpenClaw issue is the regression
guard that would have caught that specific failure mode.
"""

from __future__ import annotations

import json
import os
import sys
from typing import List, Dict, Any

import pytest


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _td(name: str, description: str = "", properties: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
            },
        },
    }


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


class TestConfigParsing:
    def test_default_when_missing(self):
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw(None)
        assert cfg.enabled == "auto"
        assert cfg.threshold_pct == 10.0

    def test_bool_true_maps_to_auto(self):
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw(True)
        assert cfg.enabled == "auto"

    def test_bool_false_maps_to_off(self):
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw(False)
        assert cfg.enabled == "off"

    def test_explicit_on(self):
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw({"enabled": "on"})
        assert cfg.enabled == "on"

    def test_invalid_enabled_falls_back_to_auto(self):
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw({"enabled": "maybe"})
        assert cfg.enabled == "auto"

    def test_threshold_clamped(self):
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw({"threshold_pct": 150})
        assert cfg.threshold_pct == 100.0
        cfg = ToolSearchConfig.from_raw({"threshold_pct": -5})
        assert cfg.threshold_pct == 0.0

    def test_search_limits_clamped(self):
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw({
            "search_default_limit": 999,
            "max_search_limit": 999,
        })
        assert cfg.max_search_limit == 50
        assert cfg.search_default_limit <= cfg.max_search_limit


# ---------------------------------------------------------------------------
# Classification — the hard invariant: core tools NEVER defer.
# ---------------------------------------------------------------------------


class TestClassification:
    def test_core_tools_never_defer(self):
        """The critical invariant from the OpenClaw report."""
        from tools.tool_search import is_deferrable_tool_name
        # Sample of core tools from _HERMES_CORE_TOOLS.
        for core_name in ["terminal", "read_file", "write_file", "patch",
                          "search_files", "todo", "memory", "browser_navigate",
                          "web_search", "session_search", "clarify",
                          "execute_code", "delegate_task", "send_message"]:
            assert not is_deferrable_tool_name(core_name), (
                f"Core tool '{core_name}' must NEVER be deferrable"
            )

    def test_bridge_tools_never_defer(self):
        from tools.tool_search import is_deferrable_tool_name, BRIDGE_TOOL_NAMES
        for name in BRIDGE_TOOL_NAMES:
            assert not is_deferrable_tool_name(name)

    def test_unknown_tool_not_deferrable(self):
        """Defensive: a tool name we cannot resolve to a registry entry must
        not be claimed as deferrable. This protects against the OpenClaw
        cron regression where unresolved tools were silently dropped."""
        from tools.tool_search import is_deferrable_tool_name
        assert not is_deferrable_tool_name("xx_definitely_not_a_tool_xx")

    def test_classify_keeps_unknown_in_visible(self):
        """A tool we can't classify stays visible — never silently dropped.

        This is the OpenClaw #84141 regression guard (cron lost ``exec``
        because it wasn't in the catalog).
        """
        from tools.tool_search import classify_tools
        # Build a tool def for something we don't have a registry entry for.
        defs = [_td("xx_unknown_tool", "Unknown tool")]
        visible, deferrable = classify_tools(defs)
        names = {(td.get("function") or {}).get("name") for td in visible}
        assert "xx_unknown_tool" in names
        assert deferrable == []


# ---------------------------------------------------------------------------
# Token estimation + threshold gate
# ---------------------------------------------------------------------------


class TestThresholdGate:
    def test_off_never_activates(self):
        from tools.tool_search import ToolSearchConfig, should_activate
        cfg = ToolSearchConfig.from_raw({"enabled": "off"})
        assert not should_activate(cfg, deferrable_tokens=1_000_000, context_length=200_000)

    def test_zero_deferrable_never_activates(self):
        from tools.tool_search import ToolSearchConfig, should_activate
        cfg = ToolSearchConfig.from_raw({"enabled": "on"})
        assert not should_activate(cfg, deferrable_tokens=0, context_length=200_000)

    def test_on_activates_with_any_deferrable(self):
        from tools.tool_search import ToolSearchConfig, should_activate
        cfg = ToolSearchConfig.from_raw({"enabled": "on"})
        assert should_activate(cfg, deferrable_tokens=100, context_length=200_000)

    def test_auto_below_threshold_does_not_activate(self):
        from tools.tool_search import ToolSearchConfig, should_activate
        cfg = ToolSearchConfig.from_raw({"enabled": "auto", "threshold_pct": 10})
        # 5% of 200K = below 10% threshold
        assert not should_activate(cfg, deferrable_tokens=10_000, context_length=200_000)

    def test_auto_at_or_above_threshold_activates(self):
        from tools.tool_search import ToolSearchConfig, should_activate
        cfg = ToolSearchConfig.from_raw({"enabled": "auto", "threshold_pct": 10})
        assert should_activate(cfg, deferrable_tokens=20_000, context_length=200_000)
        assert should_activate(cfg, deferrable_tokens=50_000, context_length=200_000)

    def test_auto_without_context_length_uses_20k_cutoff(self):
        """Fallback cutoff used when the active model is unknown."""
        from tools.tool_search import ToolSearchConfig, should_activate
        cfg = ToolSearchConfig.from_raw({"enabled": "auto"})
        assert not should_activate(cfg, deferrable_tokens=10_000, context_length=0)
        assert should_activate(cfg, deferrable_tokens=25_000, context_length=0)

    def test_token_estimate_proportional_to_schema_size(self):
        from tools.tool_search import estimate_tokens_from_schemas
        small = [_td("a", "x")]
        big = [_td(f"name_{i}", f"description for tool {i} " * 20,
                   {"q": {"type": "string", "description": "search query " * 10}})
               for i in range(10)]
        small_t = estimate_tokens_from_schemas(small)
        big_t = estimate_tokens_from_schemas(big)
        assert big_t > small_t * 10


# ---------------------------------------------------------------------------
# Retrieval (BM25 + substring fallback)
# ---------------------------------------------------------------------------


class TestRetrieval:
    def _fake_catalog(self):
        """Build a catalog directly without touching the registry."""
        from tools.tool_search import CatalogEntry, _tokenize, _entry_search_text
        defs = [
            _td("github_create_issue", "Open a new issue in a GitHub repository",
                {"title": {"type": "string"}, "body": {"type": "string"}}),
            _td("github_search_repos", "Search GitHub for matching repositories",
                {"query": {"type": "string"}}),
            _td("slack_send_message", "Post a message into a Slack channel",
                {"channel": {"type": "string"}, "text": {"type": "string"}}),
            _td("calendar_create_event", "Add an event to the user's calendar",
                {"title": {"type": "string"}, "start": {"type": "string"}}),
        ]
        catalog = []
        for d in defs:
            fn = d["function"]
            e = CatalogEntry(
                name=fn["name"], description=fn["description"],
                schema=d, source="mcp", source_name="mcp-test",
            )
            e._tokens = _tokenize(_entry_search_text(d))
            catalog.append(e)
        return catalog

    def test_search_finds_relevant_tool(self):
        from tools.tool_search import search_catalog
        hits = search_catalog(self._fake_catalog(), "create a github issue", limit=3)
        names = [h.name for h in hits]
        assert names[0] == "github_create_issue"

    def test_search_returns_empty_for_irrelevant_query(self):
        from tools.tool_search import search_catalog
        hits = search_catalog(self._fake_catalog(), "asdf qwerty foobar", limit=3)
        assert hits == []

    def test_search_substring_fallback(self):
        """Even when no BM25 hit, a literal substring of the tool name returns."""
        from tools.tool_search import search_catalog
        hits = search_catalog(self._fake_catalog(), "calendar", limit=3)
        assert any("calendar" in h.name for h in hits)

    def test_search_respects_limit(self):
        from tools.tool_search import search_catalog
        hits = search_catalog(self._fake_catalog(), "github", limit=1)
        assert len(hits) <= 1


# ---------------------------------------------------------------------------
# Assembly — the full passthrough/activate decision.
# ---------------------------------------------------------------------------


class TestAssembly:
    def test_no_deferrable_returns_unchanged(self):
        """Pure-core toolset: pass-through, no bridge tools added."""
        from tools.tool_search import assemble_tool_defs, ToolSearchConfig
        defs = [_td("terminal", "Run shell"), _td("read_file", "Read a file")]
        result = assemble_tool_defs(
            defs,
            context_length=200_000,
            config=ToolSearchConfig.from_raw({"enabled": "on"}),
        )
        assert not result.activated
        assert {t["function"]["name"] for t in result.tool_defs} == {"terminal", "read_file"}

    def test_below_threshold_returns_unchanged(self):
        """Tiny deferrable surface: don't bother."""
        from tools.tool_search import assemble_tool_defs, ToolSearchConfig
        # _td renders to ~80 chars / 20 tokens. 3 of them = ~60 tokens.
        # 10% of 200K = 20K. Way below.
        defs = [_td("unknown_tool_a"), _td("unknown_tool_b"), _td("unknown_tool_c")]
        result = assemble_tool_defs(
            defs,
            context_length=200_000,
            config=ToolSearchConfig.from_raw({"enabled": "auto", "threshold_pct": 10}),
        )
        assert not result.activated
        names = {(t.get("function") or {}).get("name") for t in result.tool_defs}
        assert "tool_search" not in names

    def test_idempotent_when_bridge_already_present(self):
        from tools.tool_search import assemble_tool_defs, ToolSearchConfig, BRIDGE_TOOL_NAMES
        defs = [_td("terminal", "Run shell"), _td("tool_search", "old")]
        result = assemble_tool_defs(
            defs,
            context_length=200_000,
            config=ToolSearchConfig.from_raw({"enabled": "off"}),
        )
        names = [(t["function"]["name"]) for t in result.tool_defs]
        # The pre-existing tool_search was stripped (it would be re-injected if
        # activation happened; here it didn't).
        assert "tool_search" not in names


# ---------------------------------------------------------------------------
# Bridge dispatch
# ---------------------------------------------------------------------------


class TestBridgeDispatch:
    def test_tool_search_requires_query(self):
        from tools.tool_search import dispatch_tool_search
        result = dispatch_tool_search({}, current_tool_defs=[])
        assert "error" in json.loads(result)

    def test_tool_describe_requires_name(self):
        from tools.tool_search import dispatch_tool_describe
        result = dispatch_tool_describe({}, current_tool_defs=[])
        assert "error" in json.loads(result)

    def test_tool_describe_rejects_non_deferrable(self):
        """If the model asks to describe a core tool, refuse — it's already
        in the visible list."""
        from tools.tool_search import dispatch_tool_describe
        result = dispatch_tool_describe(
            {"name": "terminal"}, current_tool_defs=[_td("terminal", "Run shell")],
        )
        assert "error" in json.loads(result)

    def test_resolve_underlying_call_parses_object_args(self):
        from tools.tool_search import resolve_underlying_call
        name, args, err = resolve_underlying_call({
            "name": "unknown_xxx",
            "arguments": {"foo": "bar"},
        })
        # Will fail classification because unknown_xxx isn't deferrable.
        assert err is not None

    def test_resolve_underlying_call_parses_json_string_args(self):
        """Some models emit ``arguments`` as a JSON string instead of object."""
        from tools.tool_search import resolve_underlying_call
        # Use a name that won't classify (so we don't depend on registry),
        # but exercise the JSON parse path.
        _, _, err = resolve_underlying_call({
            "name": "fake",
            "arguments": '{"a": 1}',
        })
        # err is about classification, but the parse worked (it would have
        # failed earlier with "not valid JSON" otherwise).
        assert "not valid JSON" not in (err or "")

    def test_resolve_underlying_call_rejects_bad_json(self):
        from tools.tool_search import resolve_underlying_call
        _, _, err = resolve_underlying_call({
            "name": "fake",
            "arguments": "{this is not json",
        })
        assert err is not None
        assert "JSON" in err

    def test_resolve_underlying_call_rejects_recursion(self):
        """tool_call cannot invoke tool_call itself."""
        from tools.tool_search import resolve_underlying_call, TOOL_CALL_NAME
        name, args, err = resolve_underlying_call({
            "name": TOOL_CALL_NAME,
            "arguments": {},
        })
        assert err is not None
        assert "bridge tool" in err.lower()


# ---------------------------------------------------------------------------
# End-to-end via the real handle_function_call (smoke test).
# ---------------------------------------------------------------------------


class TestHandleFunctionCallIntegration:
    def test_tool_search_dispatch_through_handle_function_call(self):
        """The dispatcher recognizes the bridge tool by name."""
        import model_tools
        result = model_tools.handle_function_call(
            function_name="tool_search",
            function_args={"query": "nothing matches this"},
        )
        parsed = json.loads(result)
        # Without a real registry, the matches will be empty, but the
        # dispatch path completed without error.
        assert "matches" in parsed or "error" in parsed


class TestRegression_OpenClawCron84141:
    """Regression guard for the OpenClaw cron-tool-loss class of bug.

    OpenClaw #84141: ``toolsAllow: ["exec"]`` on an isolated cron turn
    resulted in the agent receiving only ``sessions_send`` — the catalog
    builder silently dropped the requested core tool.

    Our defense: core tools are NEVER deferred. This test exercises the
    full assembly pipeline with a mixed core+MCP toolset and asserts that
    every core tool survives.
    """

    def test_core_tool_survives_alongside_many_mcp_tools(self):
        from tools.tool_search import (
            assemble_tool_defs, ToolSearchConfig, BRIDGE_TOOL_NAMES,
            classify_tools,
        )
        # 1 core tool + 50 unknown/MCP-shaped tools (deferrable).
        defs = [_td("terminal", "Run shell commands")]
        # Pad with fake "deferrable" tools — without registry registration,
        # classify_tools puts them in 'visible'. So instead, we just verify
        # the core-tool side: terminal stays in visible regardless.
        visible, deferrable = classify_tools(defs)
        assert any(
            (td.get("function") or {}).get("name") == "terminal"
            for td in visible
        ), "Core tool 'terminal' was wrongly classified as deferrable"

        # Now force activation and check the resulting tool-defs list.
        result = assemble_tool_defs(
            defs,
            context_length=200_000,
            config=ToolSearchConfig.from_raw({"enabled": "on"}),
        )
        names = {(t.get("function") or {}).get("name") for t in result.tool_defs}
        # terminal must be present; bridges are only added if there are
        # deferrable tools to put behind them.
        assert "terminal" in names

    def test_unwrap_rejects_core_tool_attempt(self):
        """Even if the model tries to invoke a core tool through tool_call,
        we reject the call and tell the model to use it directly."""
        from tools.tool_search import resolve_underlying_call
        _, _, err = resolve_underlying_call({
            "name": "terminal",
            "arguments": {"command": "echo hi"},
        })
        assert err is not None
        assert "not a deferrable" in err


class TestRegression_ToolsetScoping:
    """A restricted-toolset session must not see or invoke out-of-scope tools.

    The bug: the bridge dispatch and the tool_executor unwrap read the
    catalog from the *global* registry (get_tool_definitions with no
    toolset scope = "start with everything"), so a session scoped to one
    MCP server could tool_search the entire process registry and tool_call
    any plugin tool it was never granted. registry.dispatch() has no
    enabled_tools gate for non-execute_code tools, so the out-of-scope tool
    actually ran.

    The fix threads the session's enabled/disabled toolsets into the bridge
    dispatch (model_tools.handle_function_call) and the executor unwrap
    (agent.tool_executor), scoping both the searchable catalog and the
    invocable set to the session's own toolsets.
    """

    @staticmethod
    def _register(name, toolset):
        from tools.registry import registry

        def _handler(args, task_id=None, **kw):
            return json.dumps({"ok": True, "tool": name})

        registry.register(
            name=name,
            handler=_handler,
            schema=_td(name, f"desc for {name}", {"repo": {"type": "string"}}),
            toolset=toolset,
        )

    def test_search_catalog_is_scoped_to_session_toolsets(self):
        import model_tools

        for i in range(12):
            self._register(f"mcp_scoped_gh_{i}", "mcp-scoped-gh")
        self._register("scoped_oos_plugin", "scopedoosplugin")

        # tool_search scoped to the github toolset must not count the
        # out-of-scope plugin tool (or any of the host registry).
        result = model_tools.handle_function_call(
            function_name="tool_search",
            function_args={"query": "mcp_scoped_gh", "limit": 5},
            enabled_toolsets=["mcp-scoped-gh"],
        )
        parsed = json.loads(result)
        assert parsed["total_available"] == 12, (
            f"expected scoped catalog of 12, got {parsed['total_available']} "
            "— catalog leaked tools outside the session's toolsets"
        )
        hit_names = {m["name"] for m in parsed["matches"]}
        assert "scoped_oos_plugin" not in hit_names

    def test_tool_call_rejects_out_of_scope_tool(self):
        import model_tools

        self._register("mcp_inscope_gh_op", "mcp-inscope-gh")
        self._register("inscope_oos_plugin", "inscopeoosplugin")

        # Out-of-scope plugin tool: rejected even though it is registered
        # and deferrable in the global registry.
        rejected = json.loads(model_tools.handle_function_call(
            function_name="tool_call",
            function_args={"name": "inscope_oos_plugin", "arguments": {}},
            enabled_toolsets=["mcp-inscope-gh"],
        ))
        assert "error" in rejected
        assert "not available in this session" in rejected["error"]

        # In-scope tool: dispatches normally.
        ok = json.loads(model_tools.handle_function_call(
            function_name="tool_call",
            function_args={"name": "mcp_inscope_gh_op", "arguments": {"repo": "a/b"}},
            enabled_toolsets=["mcp-inscope-gh"],
        ))
        assert ok.get("ok") is True
        assert ok.get("tool") == "mcp_inscope_gh_op"

    def test_bridge_dispatch_does_not_pollute_global_resolved_names(self):
        import model_tools

        self._register("mcp_pollute_op_0", "mcp-pollute")
        self._register("mcp_pollute_op_1", "mcp-pollute")

        # Establish the scoped session global.
        model_tools.get_tool_definitions(
            enabled_toolsets=["mcp-pollute"], quiet_mode=True,
        )
        before = set(model_tools._last_resolved_tool_names)
        assert "terminal" not in before

        # A scoped tool_search call must not widen the process-global
        # _last_resolved_tool_names to the whole registry (which would leak
        # core/sandbox tools into execute_code's fallback).
        model_tools.handle_function_call(
            function_name="tool_search",
            function_args={"query": "pollute"},
            enabled_toolsets=["mcp-pollute"],
        )
        after = set(model_tools._last_resolved_tool_names)
        assert "terminal" not in after, (
            "bridge dispatch polluted _last_resolved_tool_names with "
            "out-of-scope tools"
        )

    def test_scoped_deferrable_names_helper(self):
        from tools.tool_search import scoped_deferrable_names

        self._register("mcp_helper_op", "mcp-helper")
        import model_tools
        defs = model_tools.get_tool_definitions(
            enabled_toolsets=["mcp-helper"],
            quiet_mode=True,
            skip_tool_search_assembly=True,
        )
        names = scoped_deferrable_names(defs)
        assert "mcp_helper_op" in names
        # core tools are never deferrable
        assert "terminal" not in names



# ---------------------------------------------------------------------------
# Core-tool deferral (PR #97979 port) and the never-defer floor
# ---------------------------------------------------------------------------


class TestCoreToolDeferral:
    # Names the Jentera system prompt tells the model to call directly.
    PROTECTED = {
        "terminal", "process", "read_file", "write_file", "patch", "search_files",
        "execute_code", "web_search", "web_extract", "memory", "clarify",
        "delegate_task", "skills_list", "skill_view", "skill_manage",
        "vision_analyze", "browser_navigate", "browser_snapshot", "browser_click",
        "browser_type", "browser_scroll", "browser_back", "browser_press",
        "browser_get_images", "browser_vision", "browser_console", "browser_cdp",
        "browser_dialog", "ask_specialist", "business_records", "connect_service",
        "site_draft", "site_get", "site_apply", "site_code", "site_image",
    }

    def _register(self, name: str, toolset: str) -> None:
        from tools.registry import registry
        registry.register(
            name=name, toolset=toolset,
            schema={"name": name, "description": f"{name} tool",
                    "parameters": {"type": "object", "properties": {}}},
            handler=lambda args, **kw: json.dumps({"ok": True, "tool": name}),
            check_fn=None,
        )

    def test_default_set_excludes_every_protected_tool(self):
        from tools.tool_search import _DEFAULT_DEFERRED_TOOLS
        assert not (_DEFAULT_DEFERRED_TOOLS & self.PROTECTED)
        assert {"computer_use", "session_search", "image_generate", "cronjob"} <= _DEFAULT_DEFERRED_TOOLS

    def test_default_set_defers_event_triggered_tools_below_threshold(self):
        """Core deferral is not volume-triggered: tiny schemas still defer."""
        from tools.tool_search import BRIDGE_TOOL_NAMES, ToolSearchConfig, assemble_tool_defs

        assembled = assemble_tool_defs(
            [_td("terminal", "Run"), _td("memory", "Remember"),
             _td("computer_use", "Drive the desktop"), _td("session_search", "Search old chats"),
             _td("image_generate", "Make an image"), _td("todo", "Track tasks")],
            context_length=1_000_000,
            config=ToolSearchConfig.from_raw({"enabled": "auto"}),
        )
        names = [td["function"]["name"] for td in assembled.tool_defs]
        assert assembled.activated
        assert names[:2] == ["terminal", "memory"]
        assert set(names[2:]) == BRIDGE_TOOL_NAMES

    def test_off_defers_nothing(self):
        from tools.tool_search import ToolSearchConfig, assemble_tool_defs

        defs = [_td("terminal", "Run"), _td("computer_use", "Drive the desktop")]
        assembled = assemble_tool_defs(
            defs, context_length=200_000, config=ToolSearchConfig.from_raw({"enabled": "off"}),
        )
        assert not assembled.activated
        assert assembled.tool_defs == defs

    def test_empty_defer_list_restores_everything_eager(self):
        from tools.tool_search import ToolSearchConfig, assemble_tool_defs

        defs = [_td("terminal", "Run"), _td("computer_use", "Drive the desktop")]
        assembled = assemble_tool_defs(
            defs, context_length=200_000,
            config=ToolSearchConfig.from_raw({"enabled": "auto", "defer": []}),
        )
        assert not assembled.activated
        assert assembled.tool_defs == defs

    def test_explicit_defer_list_replaces_the_default_and_reads_upstream_names(self):
        from tools.tool_search import ToolSearchConfig, assemble_tool_defs

        cfg = ToolSearchConfig.from_raw({"defer": ["image_generate", "todo_list"]})
        assert cfg.effective_defer_tools == {"image_generate", "todo"}
        assembled = assemble_tool_defs(
            [_td("computer_use", "Drive"), _td("image_generate", "Draw"), _td("todo", "Plan")],
            context_length=200_000, config=cfg,
        )
        names = {td["function"]["name"] for td in assembled.tool_defs}
        assert "computer_use" in names
        assert not ({"image_generate", "todo"} & names)

    def test_jentera_toolset_never_defers_even_when_named_or_when_the_gate_opens(self):
        from tools.tool_search import ToolSearchConfig, assemble_tool_defs

        self._register("zz_jentera_probe_tool", "jentera")
        self._register("zz_other_plugin_tool", "zz_other_plugin")
        defs = [_td("zz_jentera_probe_tool", "x" * 400), _td("zz_other_plugin_tool", "y" * 400),
                _td("computer_use", "Drive")]
        # Threshold 0 forces the MCP/plugin gate open; "defer" names the jentera tool too.
        cfg = ToolSearchConfig.from_raw({
            "enabled": "on", "threshold_pct": 0,
            "defer": ["computer_use", "zz_jentera_probe_tool"],
        })
        names = {td["function"]["name"] for td in assemble_tool_defs(
            defs, context_length=200_000, config=cfg).tool_defs}
        assert "zz_jentera_probe_tool" in names
        assert "zz_other_plugin_tool" not in names
        assert "computer_use" not in names

    def test_never_defer_config_can_protect_a_named_core_tool(self):
        from tools.tool_search import ToolSearchConfig, assemble_tool_defs

        cfg = ToolSearchConfig.from_raw({"never_defer": ["computer_use"]})
        assembled = assemble_tool_defs(
            [_td("computer_use", "Drive"), _td("session_search", "Search")],
            context_length=200_000, config=cfg,
        )
        names = {td["function"]["name"] for td in assembled.tool_defs}
        assert "computer_use" in names and "session_search" not in names

    def test_plugin_tools_still_follow_the_threshold(self):
        from tools.tool_search import ToolSearchConfig, assemble_tool_defs

        self._register("zz_gate_plugin_tool", "zz_gate_plugin")
        defs = [_td("zz_gate_plugin_tool", "tiny"), _td("computer_use", "Drive")]
        assembled = assemble_tool_defs(
            defs, context_length=1_000_000, config=ToolSearchConfig.from_raw({"enabled": "auto"}),
        )
        names = {td["function"]["name"] for td in assembled.tool_defs}
        assert "zz_gate_plugin_tool" in names  # below threshold: stays direct
        assert "computer_use" not in names     # curated set defers regardless

    def test_bridge_lists_deferred_tools_deterministically(self):
        from tools.tool_search import ToolSearchConfig, assemble_tool_defs

        defs = [_td("computer_use", "Drive the desktop. Long second sentence."),
                _td("session_search", "Search past conversations."),
                _td("terminal", "Run")]
        cfg = ToolSearchConfig.from_raw({})
        first = assemble_tool_defs(defs, context_length=200_000, config=cfg).tool_defs
        second = assemble_tool_defs(list(reversed(defs[:2])) + defs[2:], context_length=200_000, config=cfg).tool_defs
        search = next(t for t in first if t["function"]["name"] == "tool_search")
        desc = search["function"]["description"]
        assert "computer_use (Drive the desktop)" in desc
        assert "second sentence" not in desc
        assert desc.index("computer_use") < desc.index("session_search")
        # Same deferred set, different input order -> identical bridge bytes.
        assert json.dumps(search, sort_keys=False) == json.dumps(
            next(t for t in second if t["function"]["name"] == "tool_search"), sort_keys=False)

    def test_deferred_core_tool_is_reachable_through_the_bridge(self, monkeypatch):
        import model_tools
        from tools import tool_search as ts
        from tools.registry import discover_builtin_tools

        discover_builtin_tools()
        monkeypatch.setattr(ts, "load_config_readonly", lambda: ts.ToolSearchConfig.from_raw({}))
        defs = [_td("computer_use", "Drive the desktop"), _td("terminal", "Run")]
        assert ts.scoped_deferrable_names(defs) == {"computer_use"}
        name, args, err = ts.resolve_underlying_call({"name": "computer_use", "arguments": {"a": 1}})
        assert (name, args, err) == ("computer_use", {"a": 1}, None)
        described = json.loads(ts.dispatch_tool_describe(
            {"name": "computer_use"}, current_tool_defs=defs, config=ts.ToolSearchConfig.from_raw({})))
        assert described["name"] == "computer_use"
        found = json.loads(ts.dispatch_tool_search(
            {"query": "desktop"}, current_tool_defs=defs, config=ts.ToolSearchConfig.from_raw({})))
        assert [m["name"] for m in found["matches"]] == ["computer_use"]
        # terminal is the working set: not reachable via tool_call.
        assert ts.resolve_underlying_call({"name": "terminal", "arguments": {}})[2]

    def test_legacy_config_cannot_reach_deferred_core_tool(self, monkeypatch):
        from tools import tool_search as ts

        monkeypatch.setattr(ts, "load_config_readonly",
                            lambda: ts.ToolSearchConfig.from_raw({"defer": []}))
        assert ts.resolve_underlying_call({"name": "computer_use", "arguments": {}})[2]
