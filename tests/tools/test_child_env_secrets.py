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
    "TELEGRAM_WEBHOOK_SECRET": "tg-webhook-secret-1",
    "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD": "dash-pass-1",
    "WEIXIN_TOKEN": "weixin-token-1",
}


@pytest.fixture
def child_environ(monkeypatch):
    monkeypatch.delenv("HERMES_CHILD_ENV_KEEP", raising=False)
    for k, v in {**JENTERA_ENV, **MODEL_ENV, **HERMES_SECRETS}.items():
        monkeypatch.setenv(k, v)


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
    for name in HERMES_SECRETS:
        assert name not in visible, f"{name} must not reach the terminal child"
        assert HERMES_SECRETS[name] not in out
    # The model key stays out, as on the pin before this port (its URL is not a secret).
    assert "OPENROUTER_API_KEY" not in visible
    assert MODEL_ENV["OPENROUTER_API_KEY"] not in out


def test_background_process_env_matches(child_environ):
    env = _sanitize_subprocess_env(dict(os.environ))
    for name in JENTERA_ENV:
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
    for name in ("TELEGRAM_WEBHOOK_SECRET", "WEIXIN_TOKEN", "FEISHU_ENCRYPT_KEY",
                 "PHOTON_SIDECAR_TOKEN", "HERMES_DASHBOARD_BASIC_AUTH_SECRET", "API_SERVER_KEY"):
        assert name in _ALWAYS_STRIP_KEYS
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
    monkeypatch.setattr(ep, "_config_passthrough", frozenset({"WEIXIN_TOKEN", "ACME_API_KEY"}))
    ep.register_env_passthrough(["TELEGRAM_WEBHOOK_SECRET", "ACME_API_KEY"])
    try:
        assert not ep.is_env_passthrough("WEIXIN_TOKEN")
        assert not ep.is_env_passthrough("TELEGRAM_WEBHOOK_SECRET")
        assert ep.is_env_passthrough("ACME_API_KEY")
        assert "WEIXIN_TOKEN" not in ep.get_all_passthrough()
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
                if name.endswith(suffixes) and name not in _HERMES_PROVIDER_ENV_BLOCKLIST:
                    missing.add(name)
    # Names that are not secrets despite the suffix, or belong to the user.
    assert not (missing - {"SLACK_USER_TOKEN"}), sorted(missing)
