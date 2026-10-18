"""A configured ``base_url`` must beat the vendor host everywhere (#121614, #121359, #121347).

The config below has the shape Jentera writes on every sprite: ``provider: openrouter`` with
``base_url`` pointing at a proxy, every auxiliary task pinned to ``provider: custom`` on the same
URL, and extra ``gateway.api_server.extra.model_routes``. The relationship under test is
"configured URL X -> resolved URL is X, and the key is never paired with the vendor host".
Nothing here makes a live request: constructing an SDK client does no I/O, and the catalog
probes are replaced with recorders that fail the test if the vendor host is asked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import yaml

from agent import auxiliary_client as ac
from hermes_cli import models
from hermes_cli import runtime_provider as rp

PROXY = "https://proxy.jentera.invalid/v1/model"
KEY = "sk-jentera-issued-test-key"
VENDOR_HOST = "openrouter.ai"
AUX_TASKS = (
    "compression", "title_generation", "session_search", "vision",
    "web_extract", "approval", "skills_hub", "mcp",
)


def _jentera_config(*, fallback=None):
    cfg = {
        "model": {
            "default": "vendor/model-a",
            "provider": "openrouter",
            "base_url": PROXY,
            "api_key": "${OPENROUTER_API_KEY}",
            "api_mode": "chat_completions",
        },
        "auxiliary": {
            task: {
                "provider": "custom",
                "base_url": PROXY,
                "api_key": "${OPENROUTER_API_KEY}",
                "model": "vendor/model-a",
                "api_mode": "chat_completions",
            }
            for task in AUX_TASKS
        },
        "gateway": {"api_server": {"extra": {"model_routes": {
            "vendor/model-b": {
                "model": "vendor/model-b",
                "provider": "openrouter",
                "base_url": PROXY,
                "api_key": "${OPENROUTER_API_KEY}",
            },
        }}}},
    }
    if fallback is not None:
        cfg["fallback_providers"] = fallback
    return cfg


@pytest.fixture
def write_config(tmp_path, monkeypatch):
    for name in ("OPENROUTER_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                 "ANTHROPIC_TOKEN", "CUSTOM_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def _write(cfg):
        (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))

    return _write


def _assert_proxy(client):
    url = str(client.base_url)
    assert url.rstrip("/") == PROXY
    assert VENDOR_HOST not in url


# -- main chat -----------------------------------------------------------------------------

@pytest.mark.parametrize("env_also_set", [True, False])
def test_main_runtime_resolves_configured_url(write_config, monkeypatch, env_also_set):
    """With only ``model.base_url`` (no OPENROUTER_BASE_URL) the main path must still use it."""
    write_config(_jentera_config())
    if env_also_set:
        monkeypatch.setenv("OPENROUTER_BASE_URL", PROXY)

    resolved = rp.resolve_runtime_provider(requested="openrouter")

    assert resolved["provider"] == "openrouter"
    assert resolved["base_url"] == PROXY
    assert resolved["api_key"] == KEY


def test_env_only_still_works(write_config, monkeypatch):
    """OPENROUTER_BASE_URL alone (no model.base_url) keeps selecting the endpoint."""
    cfg = _jentera_config()
    cfg["model"].pop("base_url")
    write_config(cfg)
    monkeypatch.setenv("OPENROUTER_BASE_URL", PROXY)

    resolved = rp.resolve_runtime_provider(requested="openrouter")

    assert resolved["base_url"] == PROXY
    assert resolved["api_key"] == KEY


def test_base_url_saved_under_another_provider_is_ignored(write_config):
    """A relay URL left behind by a different ``model.provider`` must not redirect OpenRouter."""
    cfg = _jentera_config()
    cfg["model"]["provider"] = "deepseek"
    write_config(cfg)

    resolved = rp.resolve_runtime_provider(requested="openrouter")

    assert VENDOR_HOST in resolved["base_url"]


def test_model_routes_carry_the_configured_url(write_config):
    write_config(_jentera_config())
    from hermes_cli.config import load_config

    routes = load_config()["gateway"]["api_server"]["extra"]["model_routes"]
    assert routes["vendor/model-b"]["base_url"] == PROXY


# -- auxiliary tasks -----------------------------------------------------------------------

@pytest.mark.parametrize("task", AUX_TASKS)
def test_every_aux_task_uses_configured_url(write_config, task):
    write_config(_jentera_config())

    client, _model = ac.get_text_auxiliary_client(task)

    _assert_proxy(client)
    assert client.api_key == KEY


def test_aux_openrouter_provider_without_pin_uses_configured_url(write_config):
    """The workaround (``provider: custom``) is no longer needed for the URL."""
    cfg = _jentera_config()
    cfg["auxiliary"] = {task: {"provider": "openrouter"} for task in AUX_TASKS}
    write_config(cfg)

    for task in AUX_TASKS:
        client, _model = ac.get_text_auxiliary_client(task)
        _assert_proxy(client)


# -- fallback ------------------------------------------------------------------------------

def test_fallback_openrouter_entry_without_url_uses_configured_url(write_config):
    write_config(_jentera_config())

    client, _model = ac.resolve_provider_client("openrouter", "vendor/model-c")

    _assert_proxy(client)
    assert client.api_key == KEY


def test_fallback_openrouter_entry_url_wins(write_config):
    write_config(_jentera_config())
    entry_url = "http://127.0.0.1:9002/openrouter/v1"

    client, _model = ac.resolve_provider_client(
        "openrouter", "vendor/model-c", explicit_base_url=entry_url,
    )

    assert str(client.base_url).rstrip("/") == entry_url


def test_openrouter_canonical_host_when_nothing_configured(write_config):
    """No relay anywhere: unchanged behaviour."""
    write_config({"model": {"provider": "openrouter", "default": "vendor/model-a"}})

    client, _model = ac.resolve_provider_client("openrouter", "vendor/model-c")

    assert VENDOR_HOST in str(client.base_url)


def _build_anthropic(explicit_base_url=None):
    with (
        patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)),
        patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="sk-ant-test"),
        patch("agent.anthropic_adapter.build_anthropic_client") as build,
    ):
        build.return_value = MagicMock()
        client, _model = ac._try_anthropic(explicit_base_url=explicit_base_url)
    return client, build


def test_fallback_anthropic_entry_url_reaches_the_client(write_config):
    write_config(_jentera_config())
    entry_url = "https://api.anthropic.com/entry-specific"

    client, build = _build_anthropic(entry_url)

    assert client is not None
    assert build.call_args.args[1] == entry_url


def test_fallback_anthropic_foreign_host_is_refused(write_config):
    """A non-Anthropic host is refused, not demoted to api.anthropic.com with the key."""
    write_config(_jentera_config())

    client, build = _build_anthropic(PROXY)

    assert client is None
    build.assert_not_called()


# -- model listing -------------------------------------------------------------------------

def test_listing_probes_configured_url_never_the_vendor(write_config, monkeypatch):
    write_config(_jentera_config())
    seen = []

    def fake_fetch(api_key, base_url, **_kw):
        seen.append((api_key, base_url))
        return ["relay-only/model"]

    def no_vendor(*_a, **_kw):
        raise AssertionError("vendor catalog must not be fetched when a relay is configured")

    monkeypatch.setattr(models, "fetch_api_models", fake_fetch)
    monkeypatch.setattr(models, "_urlopen_model_catalog_request", no_vendor)

    assert models.provider_model_ids("openrouter") == ["relay-only/model"]
    assert seen == [(KEY, PROXY)]


def test_listing_failed_probe_degrades_locally(write_config, monkeypatch):
    write_config(_jentera_config())

    def no_vendor(*_a, **_kw):
        raise AssertionError("vendor catalog must not be fetched when a relay is configured")

    monkeypatch.setattr(models, "fetch_api_models", lambda *a, **k: None)
    monkeypatch.setattr(models, "_urlopen_model_catalog_request", no_vendor)
    monkeypatch.setattr(models, "_openrouter_curated_ids", lambda: ["curated/one"])

    assert models.provider_model_ids("openrouter") == ["curated/one"]


def test_listing_without_relay_is_unchanged(write_config, monkeypatch):
    write_config({"model": {"provider": "openrouter", "default": "vendor/model-a"}})
    monkeypatch.setattr(models, "model_ids", lambda force_refresh=False: ["canonical/one"])

    assert models.provider_model_ids("openrouter") == ["canonical/one"]


# -- validation ----------------------------------------------------------------------------

def test_validation_probes_the_configured_url(write_config, monkeypatch):
    write_config(_jentera_config())
    probed = []

    def fake_probe(api_key, base_url, **_kw):
        probed.append(base_url)
        return {"models": ["relay-only/model"], "probed_url": base_url + "/models"}

    monkeypatch.setattr(models, "probe_api_models", fake_probe)

    result = models.validate_requested_model(
        "relay-only/model", "openrouter", api_key=KEY, base_url=PROXY,
    )

    assert result["accepted"] and result["recognized"]
    assert probed == [PROXY]


# -- xAI pool row --------------------------------------------------------------------------

class _Entry:
    access_token = "pool-token"
    source = "env:XAI_API_KEY"
    runtime_base_url = None
    runtime_api_key = None

    def __init__(self, url):
        self.base_url = url


class _Pool:
    def __init__(self, url):
        self._entry = _Entry(url)

    def has_credentials(self):
        return True

    def select(self, **_kwargs):
        return self._entry


def _xai_pool(url):
    return _Pool(url)


@pytest.mark.parametrize("row_url, expected", [
    ("https://api.x.ai/v1", "http://127.0.0.1:8765/v1"),   # registry host: relay wins
    ("http://127.0.0.1:7777/v1", "http://127.0.0.1:7777/v1"),  # explicit row endpoint wins
])
def test_xai_pool_row_honours_model_base_url(monkeypatch, row_url, expected):
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "xai")
    monkeypatch.setattr(rp, "load_pool", lambda provider: _xai_pool(row_url))
    monkeypatch.setattr(rp, "_get_model_config", lambda: {
        "provider": "xai", "default": "grok-4", "base_url": "http://127.0.0.1:8765/v1/"})

    resolved = rp.resolve_runtime_provider(requested="xai")

    assert resolved["base_url"] == expected


def test_xai_pool_row_ignores_base_url_of_another_provider(monkeypatch):
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "xai")
    monkeypatch.setattr(rp, "load_pool", lambda provider: _xai_pool("https://api.x.ai/v1"))
    monkeypatch.setattr(rp, "_get_model_config", lambda: {
        "provider": "deepseek", "default": "grok-4", "base_url": "http://127.0.0.1:8765/v1"})

    resolved = rp.resolve_runtime_provider(requested="xai")

    assert resolved["base_url"] == "https://api.x.ai/v1"


# -- OPENAI_API_KEY must not follow the OpenRouter ladder to another host (#120299) -----------

@pytest.mark.parametrize("openai_base_url, expected_key", [
    ("https://proxy.corp.example/v1", ""),
    ("proxy.corp.example:8080/v1", ""),  # scheme-less still names a foreign host
    ("https://openrouter.ai/api/v1", "sk-openai-fallback"),
    ("", "sk-openai-fallback"),
])
def test_openai_key_bound_to_another_host_never_reaches_openrouter(
    monkeypatch, openai_base_url, expected_key,
):
    monkeypatch.setattr(rp, "_get_model_config", lambda: {})
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-fallback")
    if openai_base_url:
        monkeypatch.setenv("OPENAI_BASE_URL", openai_base_url)
    else:
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    resolved = rp._resolve_openrouter_runtime(requested_provider="openrouter")

    assert resolved["base_url"] == "https://openrouter.ai/api/v1"
    assert resolved["api_key"] == expected_key


def test_jentera_key_precedence_unchanged_with_foreign_openai_base_url(write_config, monkeypatch):
    """OPENROUTER_API_KEY still wins; a bound OPENAI_API_KEY is never the fallback."""
    write_config(_jentera_config())
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-elsewhere")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.corp.example/v1")

    resolved = rp.resolve_runtime_provider(requested="openrouter")

    assert resolved["base_url"] == PROXY
    assert resolved["api_key"] == KEY
