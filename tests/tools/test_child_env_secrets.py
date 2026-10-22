"""Child-process secret scrub (upstream #125755) plus Jentera's keep-list.

Declared adapter secrets and Hermes' own secrets never reach an agent child; JENTERA_* and
owner keys do. The model key (OPENROUTER_*) stays out of the terminal, as before: Jentera's
connector tools read it from $HERMES_HOME/.env, not from their environment.
"""

import glob
import os
import re
import shutil

import pytest

from tools.environments.local import (
    LocalEnvironment,
    _ADAPTER_SECRET_ENV,
    _ALWAYS_STRIP_KEYS,
    _CODE_ONLY_ADAPTER_SECRET_ENV,
    _HERMES_OWN_SECRET_ENV,
    _HERMES_PROVIDER_ENV_BLOCKLIST,
    _sanitize_subprocess_env,
    hermes_subprocess_env,
)

MODEL_ENV = {
    "OPENROUTER_API_KEY": "sk-or-v1-jentera-runtime-key",
    "OPENROUTER_BASE_URL": "https://api.jentera.ai/v1/model",
}
JENTERA_ENV = {
    "JENTERA_RUN_ID": "run-1234",
    "JENTERA_CALENDAR_API": "https://api.jentera.ai/v1/connectors/google-calendar",
    "ACME_API_KEY": "owner-pasted-acme-key-1",
}
HERMES_SECRETS = {
    "API_SERVER_KEY": "api-server-secret-1",
    "TELEGRAM_BOT_TOKEN": "123456:telegram-bot-token-abc",
    "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD": "dash-pass-1",
    # Tier 1 but absent from the provider blocklist: only a Tier-1 check on the terminal paths
    # keeps it out.
    "SLACK_SIGNING_SECRET": "slack-signing-1",
}
# Code-only adapter names: Jentera runs none of those platforms, so on a sprite these are keys an
# owner pasted and must reach the agent's terminal, as on the production pin.
OWNER_PASTED_KEYS = {
    "WHATSAPP_CLOUD_ACCESS_TOKEN": "wa-token-1",
    "WEIXIN_TOKEN": "weixin-token-1",
    "QQ_STT_API_KEY": "qq-stt-key-1",
    "TEAMS_GRAPH_ACCESS_TOKEN": "teams-graph-1",
    "MSGRAPH_CLIENT_SECRET": "msgraph-secret-1",
    "FEISHU_ENCRYPT_KEY": "feishu-enc-1",
    "YUANBAO_APP_SECRET": "yuanbao-secret-1",
}
assert set(OWNER_PASTED_KEYS) <= set(_CODE_ONLY_ADAPTER_SECRET_ENV)


@pytest.fixture
def child_environ(monkeypatch):
    monkeypatch.delenv("HERMES_CHILD_ENV_KEEP", raising=False)
    for k, v in {**JENTERA_ENV, **MODEL_ENV, **HERMES_SECRETS, **OWNER_PASTED_KEYS}.items():
        monkeypatch.setenv(k, v)
    # The run marker comes from the run's context, not the process environment (a
    # gateway that has served a run strips an inherited copy); bind the same value.
    from gateway.session_context import _JENTERA_RUN_ID

    token = _JENTERA_RUN_ID.set(JENTERA_ENV["JENTERA_RUN_ID"])
    yield
    _JENTERA_RUN_ID.reset(token)


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_terminal_child_sees_jentera_vars_and_not_hermes_secrets(child_environ, tmp_path):
    """Run a real LocalEnvironment child and read its environment."""
    env = LocalEnvironment(cwd=str(tmp_path), timeout=30)
    try:
        out = env.execute("env")["output"]
    finally:
        env.cleanup()
    visible = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    for name, value in JENTERA_ENV.items():
        assert visible.get(name) == value, f"{name} must reach the terminal child"
    for name, value in OWNER_PASTED_KEYS.items():
        assert visible.get(name) == value, f"{name} is an owner key and must reach the terminal child"
    for name in HERMES_SECRETS:
        assert name not in visible, f"{name} must not reach the terminal child"
        assert HERMES_SECRETS[name] not in out
    # The model key stays out, as on the pin before this port (its URL is not a secret).
    assert "OPENROUTER_API_KEY" not in visible
    assert MODEL_ENV["OPENROUTER_API_KEY"] not in out


def test_background_process_env_matches(child_environ):
    env = _sanitize_subprocess_env(dict(os.environ), stamp_run_marker=True)
    for name in (*JENTERA_ENV, *OWNER_PASTED_KEYS):
        assert name in env
    for name in (*HERMES_SECRETS, "OPENROUTER_API_KEY"):
        assert name not in env


@pytest.mark.parametrize("inherit", [False, True])
def test_non_terminal_children_never_get_tier1(child_environ, inherit):
    env = hermes_subprocess_env(inherit_credentials=inherit)
    for name in HERMES_SECRETS:
        assert name not in env
    assert env["ACME_API_KEY"] == JENTERA_ENV["ACME_API_KEY"]
    # The keep-list is the terminal's; other spawns still drop the model key.
    assert ("OPENROUTER_API_KEY" in env) is inherit


def test_keep_list_is_configurable_but_cannot_rescue_tier1(child_environ, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-owner-wants-this")
    assert "OPENAI_API_KEY" not in _sanitize_subprocess_env(dict(os.environ))
    monkeypatch.setenv("HERMES_CHILD_ENV_KEEP", "OPENAI_API_KEY, API_SERVER_KEY ,TELEGRAM_BOT_TOKEN")
    env = _sanitize_subprocess_env(dict(os.environ))
    assert env["OPENAI_API_KEY"] == "sk-owner-wants-this"
    assert "API_SERVER_KEY" not in env and "TELEGRAM_BOT_TOKEN" not in env


def test_tier1_covers_declared_adapter_and_hermes_secrets():
    assert _ADAPTER_SECRET_ENV <= _ALWAYS_STRIP_KEYS
    assert _HERMES_OWN_SECRET_ENV <= _ALWAYS_STRIP_KEYS
    assert _ALWAYS_STRIP_KEYS <= _HERMES_PROVIDER_ENV_BLOCKLIST | _ALWAYS_STRIP_KEYS
    for name in ("HERMES_DASHBOARD_BASIC_AUTH_SECRET", "API_SERVER_KEY", "TELEGRAM_BOT_TOKEN"):
        assert name in _ALWAYS_STRIP_KEYS
    # Code-only adapter names are not Tier 1 on Jentera (owner-pasted keys).
    assert not (_CODE_ONLY_ADAPTER_SECRET_ENV & _ALWAYS_STRIP_KEYS)
    assert not (_CODE_ONLY_ADAPTER_SECRET_ENV & _HERMES_PROVIDER_ENV_BLOCKLIST)
    # The names Jentera's tools and owners rely on are not Tier 1.
    for name in ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "JENTERA_RUN_ID", "ACME_API_KEY",
                 "SLACK_USER_TOKEN", "LOCAL_LLM_API_KEY"):
        assert name not in _ALWAYS_STRIP_KEYS


def test_user_variables_are_not_adapter_secrets(monkeypatch):
    monkeypatch.setenv("SLACK_USER_TOKEN", "user-owned-1")
    monkeypatch.setenv("LOCAL_LLM_API_KEY", "user-owned-2")
    env = hermes_subprocess_env(inherit_credentials=True)
    assert env["SLACK_USER_TOKEN"] == "user-owned-1"
    assert env["LOCAL_LLM_API_KEY"] == "user-owned-2"


def test_passthrough_cannot_restore_adapter_secrets(monkeypatch):
    from tools import env_passthrough as ep

    ep.clear_env_passthrough()
    monkeypatch.setattr(ep, "_config_passthrough", frozenset({"API_SERVER_KEY", "ACME_API_KEY"}))
    ep.register_env_passthrough(["TELEGRAM_BOT_TOKEN", "ACME_API_KEY"])
    try:
        assert not ep.is_env_passthrough("API_SERVER_KEY")
        assert not ep.is_env_passthrough("TELEGRAM_BOT_TOKEN")
        assert ep.is_env_passthrough("ACME_API_KEY")
        assert "API_SERVER_KEY" not in ep.get_all_passthrough()
    finally:
        ep.clear_env_passthrough()


def test_every_secret_named_gateway_and_adapter_read_is_covered():
    """A new ``*_TOKEN/_SECRET/_KEY/_PASSWORD`` env read in gateway/config.py or a platform
    adapter must be declared (OPTIONAL_ENV_VARS, a plugin manifest, or the code-only list)."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    files = [os.path.join(root, "gateway", "config.py")] + glob.glob(
        os.path.join(root, "plugins", "platforms", "*", "adapter.py"))
    pattern = re.compile(r"(?:getenv|environ\.get|environ\[)\(?\s*[\"']([A-Z][A-Z0-9_]+)[\"']")
    suffixes = ("_TOKEN", "_SECRET", "_KEY", "_PASSWORD", "_JSON")
    missing = set()
    for path in files:
        with open(path, encoding="utf-8") as fh:
            for name in pattern.findall(fh.read()):
                if (name.endswith(suffixes) and name not in _HERMES_PROVIDER_ENV_BLOCKLIST
                        and name not in _CODE_ONLY_ADAPTER_SECRET_ENV):
                    missing.add(name)
    # Names that are not secrets despite the suffix, or belong to the user.
    assert not (missing - {"SLACK_USER_TOKEN"}), sorted(missing)


def test_tier1_names_outside_provider_blocklist_are_dropped_on_terminal_paths(monkeypatch):
    from tools import env_passthrough as ep
    from tools.environments.local import _make_run_env

    assert "SLACK_SIGNING_SECRET" in _ALWAYS_STRIP_KEYS
    assert "SLACK_SIGNING_SECRET" not in _HERMES_PROVIDER_ENV_BLOCKLIST
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "slack-signing-1")
    monkeypatch.setenv("HERMES_CHILD_ENV_KEEP", "SLACK_SIGNING_SECRET")
    ep.clear_env_passthrough()
    ep.register_env_passthrough(["SLACK_SIGNING_SECRET"])
    try:
        assert "SLACK_SIGNING_SECRET" not in _make_run_env({})
        assert "SLACK_SIGNING_SECRET" not in _make_run_env({"SLACK_SIGNING_SECRET": "x"})
        assert "SLACK_SIGNING_SECRET" not in _sanitize_subprocess_env(dict(os.environ))
        assert "SLACK_SIGNING_SECRET" not in _sanitize_subprocess_env({}, {"SLACK_SIGNING_SECRET": "x"})
        forced = {"_HERMES_FORCE_SLACK_SIGNING_SECRET": "x"}
        assert "SLACK_SIGNING_SECRET" not in _sanitize_subprocess_env({}, forced)
        assert "SLACK_SIGNING_SECRET" not in _make_run_env(forced)
    finally:
        ep.clear_env_passthrough()
