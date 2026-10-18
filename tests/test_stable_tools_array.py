"""tools[] is deterministic for an unchanged tool set, and membership is never frozen.

``tools[]`` heads every request, so unstable bytes re-prefill the whole cached
prompt (port of NousResearch/hermes-agent#120289, adapted: this fork recomputes
membership on every turn, so only order and serialisation are canonical).
Jentera's site tools and ask_specialist are offered only while an availability
probe says so; a long-lived session must gain and lose them as the probe flips.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

import model_tools
from tools import registry as registry_mod
from tools.registry import registry

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOOLSETS = ["terminal", "memory", "zz_stable_plugin"]


@pytest.fixture
def gated_tool(monkeypatch):
    """A plugin-style tool behind a flippable availability probe, with a zero TTL."""
    state = {"on": False}
    monkeypatch.setattr(registry_mod, "_CHECK_FN_TTL_SECONDS", 0.0)
    monkeypatch.setattr(registry_mod, "_CHECK_FN_FAILURE_GRACE_SECONDS", 0.0)
    registry_mod.invalidate_check_fn_cache()
    model_tools._clear_tool_defs_cache()
    registry.register(
        name="zz_stable_gated", toolset="zz_stable_plugin",
        schema={"name": "zz_stable_gated", "description": "Gated by a probe.",
                "parameters": {"type": "object", "properties": {}}},
        handler=lambda args, **kw: "{}", check_fn=lambda: state["on"],
    )
    yield state
    registry.deregister("zz_stable_gated")
    registry_mod.invalidate_check_fn_cache()
    model_tools._clear_tool_defs_cache()


def _tools():
    return model_tools.get_tool_definitions(enabled_toolsets=TOOLSETS, quiet_mode=True)


def _names(tools):
    return [t["function"]["name"] for t in tools]


def _bytes(tools):
    return json.dumps(tools, ensure_ascii=False, separators=(",", ":"))


class TestMembershipIsRecomputedEveryTurn:
    def test_gated_tool_appears_on_the_next_turn_without_any_registry_change(self, gated_tool):
        before = _tools()
        assert "zz_stable_gated" not in _names(before)
        gen = registry._generation

        gated_tool["on"] = True  # the probe now says available
        after = _tools()

        assert registry._generation == gen  # nothing re-registered: only the probe flipped
        assert "zz_stable_gated" in _names(after)
        # Everything that was already there is still there, byte for byte.
        kept = [t for t in after if t["function"]["name"] != "zz_stable_gated"]
        assert _bytes(kept) == _bytes(before)

    def test_gated_tool_disappears_when_the_probe_turns_off(self, gated_tool):
        gated_tool["on"] = True
        assert "zz_stable_gated" in _names(_tools())
        gated_tool["on"] = False
        assert "zz_stable_gated" not in _names(_tools())

    def test_same_available_set_gives_byte_identical_tools_across_turns(self, gated_tool):
        gated_tool["on"] = True
        turns = [_bytes(_tools()) for _ in range(4)]
        assert len(set(turns)) == 1
        gated_tool["on"] = False
        gated_tool["on"] = True  # flapped and came back
        assert _bytes(_tools()) == turns[0]

    def test_gated_tool_sits_after_every_built_in_so_a_flip_only_disturbs_the_tail(self, gated_tool):
        without = _names(_tools())
        gated_tool["on"] = True
        with_it = _names(_tools())
        assert with_it[:-1] == without
        assert with_it[-1] == "zz_stable_gated"


class TestCanonicalToolDefs:
    @staticmethod
    def _td(name, **extra):
        return {"type": "function", "function": {"parameters": {"type": "object"}, "description": "d",
                                                  "name": name, **extra}}

    def test_order_is_independent_of_input_order(self):
        defs = [self._td("terminal"), self._td("memory"), self._td("zz_unknown_plugin"),
                self._td("tool_call"), self._td("tool_search")]
        forward = model_tools.canonical_tool_defs(defs)
        backward = model_tools.canonical_tool_defs(list(reversed(defs)))
        assert _bytes(forward) == _bytes(backward)
        assert _names(forward) == ["memory", "terminal", "zz_unknown_plugin", "tool_call", "tool_search"]

    def test_duplicates_keep_the_first_definition_and_key_order_is_fixed(self):
        first = self._td("terminal", strict=True)
        out = model_tools.canonical_tool_defs([first, self._td("terminal")])
        assert len(out) == 1
        assert list(out[0]["function"]) == ["name", "description", "parameters", "strict"]

    def test_input_is_not_mutated(self):
        defs = [self._td("terminal"), self._td("memory")]
        snapshot = json.dumps(defs)
        model_tools.canonical_tool_defs(defs)
        assert json.dumps(defs) == snapshot


def test_tools_bytes_do_not_depend_on_the_process_hash_seed(tmp_path):
    """Set iteration order varies per process; the serialised array must not."""
    script = (
        "import json, hashlib, model_tools\n"
        "from tools.registry import discover_builtin_tools, registry\n"
        "discover_builtin_tools()\n"
        "for e in registry._snapshot_entries(): e.check_fn = None\n"
        "t = model_tools.get_tool_definitions(quiet_mode=True)\n"
        "print(len(t), hashlib.sha256(json.dumps(t).encode()).hexdigest())\n"
    )
    digests = set()
    for seed in ("1", "2", "31337"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "HERMES_HOME": str(tmp_path), "PYTHONPATH": REPO}
        out = subprocess.run([sys.executable, "-c", script], env=env, cwd=REPO,
                             capture_output=True, text=True, timeout=120)
        assert out.returncode == 0, out.stderr[-2000:]
        digests.add(out.stdout.strip().splitlines()[-1])
    assert len(digests) == 1, digests
